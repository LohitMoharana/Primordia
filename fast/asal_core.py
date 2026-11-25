"""
ASAL on Primordia: shared pieces (CLIP, scores, pictures, one search generation).

Built on two projects:
  ASAL   Kumar et al. 2024, "Automating the Search for Artificial Life with Foundation
         Models" (github.com/SakanaAI/asal, Apache-2.0). Taken over: the supervised-target,
         softmax-target, open-endedness and illumination scores (asal_* functions, same
         maths as asal_metrics.py, ported from JAX to NumPy), Sep-CMA-ES search, averaging
         over several initial states (bs), the illumination genetic algorithm.
  ASAL++ Baid et al. 2025, "Guiding Evolution of Artificial Life Using Vision-Language
         Models" (github.com/fredericowieser/ASALPlusPlus). Taken over: the evolver loop
         (a vision-language model watches the best world and proposes the next target;
         the search warm-starts from the previous best), EST and ETT (see asal_evolve.py).

Added for a world of evolving creatures (not in either project):
  - pictures of the creatures only, in two views: where they stand and where their
    genes put them (asal_world.render), so food or soil moving cannot look like novelty;
  - a noise floor for open-endedness (novelty): two random halves of the creatures are
    pictured at every moment; a moment only counts as new by how much further it is from
    every earlier moment than its own two halves are from each other. Without it,
    near-dead worlds win: a handful of scattered dots looks different in every picture.
  - every picture shows the same number of creatures (two samples of 100), so a growing
    or shrinking population cannot look like change (found in testing: without this,
    dying worlds won, because ever-thinner pictures look new);
  - a 3000-tick settling period (only extinction counts, no pictures); after it,
    collapse (< 200 creatures) and explosion freeze a world (no new pictures), so they earn nothing.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import asal_world as AW


# ----------------------------------------------------------------------------- CLIP
class Clip:
    MODEL = "ViT-B-32-quickgelu"

    def __init__(self, weights=None, device=None):
        import torch
        import open_clip
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cpu":
            torch.set_num_threads(4)
        pre = weights if weights else "laion400m_e32"  # downloaded once (about 600 MB) if no file is given
        self.model, _, self.pre = open_clip.create_model_and_transforms(self.MODEL, pretrained=pre, device=self.device)
        self.model.eval()
        self.tok = open_clip.get_tokenizer(self.MODEL)
        norm = self.pre.transforms[-1]
        self.mean = torch.tensor(norm.mean, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(norm.std, device=self.device).view(1, 3, 1, 1)

    def images(self, frames, batch=64):
        """frames uint8 [N,H,W,3] -> unit vectors [N,512] (numpy)."""
        torch = self.torch
        out = []
        with torch.no_grad():
            for i in range(0, len(frames), batch):
                x = torch.from_numpy(np.ascontiguousarray(frames[i:i + batch])).to(self.device)
                x = x.permute(0, 3, 1, 2).float() / 255.0
                x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", antialias=True,
                                                    align_corners=False)
                z = self.model.encode_image((x - self.mean) / self.std).float()
                out.append(torch.nn.functional.normalize(z, dim=-1).cpu().numpy())
        return np.concatenate(out)

    def texts(self, prompts):
        torch = self.torch
        with torch.no_grad():
            z = self.model.encode_text(self.tok(list(prompts)).to(self.device)).float()
            return torch.nn.functional.normalize(z, dim=-1).cpu().numpy()


# ----------------------------------------------------------------------------- ASAL's scores (ported)
def _repeat_prompts(z, z_txt):
    """ASAL: the prompts take equal turns along the run (T frames, P prompts, T % P == 0).
    Here any T works: frame t belongs to prompt floor(t * P / T)."""
    T, P = len(z), len(z_txt)
    return z_txt[(np.arange(T) * P) // T]


def asal_supervised_target(z, z_txt):
    """ASAL calc_supervised_target_score: -(mean similarity of each frame to its prompt). Minimise."""
    return -float((z * _repeat_prompts(z, z_txt)).sum(1).mean())


def asal_softmax_target(z, z_txt, temperature=0.01):
    """ASAL calc_supervised_target_softmax_score: each frame should match its own prompt better
    than the other frames do, and the other way round. Minimise."""
    zt = _repeat_prompts(z, z_txt)
    k = zt @ z.T / temperature
    def lsm(a, axis):
        m = a.max(axis=axis, keepdims=True)
        return a - m - np.log(np.exp(a - m).sum(axis=axis, keepdims=True))
    return float((-np.diag(lsm(k, -1)).mean() - np.diag(lsm(k, -2)).mean()) / 2)


def asal_open_endedness(z):
    """ASAL calc_open_endedness_score, exactly: mean over frames of the highest similarity to
    an earlier frame (the first frame counts as 0, as tril(k=-1) leaves its row at 0).
    Minimise. Kept for comparison; the search uses novelty()."""
    k = np.tril(z @ z.T, k=-1)
    return float(k.max(axis=-1).mean())


def asal_illumination(zs):
    """ASAL calc_illumination_score: mean over worlds of the similarity to the nearest other world. Minimise."""
    k = zs @ zs.T
    np.fill_diagonal(k, -np.inf)
    return float(k.max(axis=-1).mean())


# ----------------------------------------------------------------------------- ours
def novelty(zA, zB):
    """Open-endedness with a noise floor (higher = more open-ended).
    zA, zB: embeddings of two random halves of the creatures at each moment.
        new_t = max(0, sim(A_t, B_t) - max_{t' < t} sim(A_t, B_t'))
    The mean of new_t over the run. A frozen (dead) world earns nothing; random scatter in a
    tiny population makes A_t unlike B_t too, so it earns nothing either."""
    k = zA @ zB.T
    return float(np.mean([max(0.0, k[t, t] - k[t, :t].max()) for t in range(1, len(zA))]))


def embed_run(clip, r):
    """All embeddings one score needs, from one rollout."""
    return dict(full=clip.images(r["full"]), sA=clip.images(r["sA"]), sB=clip.images(r["sB"]),
                gA=clip.images(r["gA"]), gB=clip.images(r["gB"]))


def score(z, objective="open", z_txt=None, coef_prompt=1.0, coef_softmax=0.0, coef_oe=0.0):
    """Loss to minimise, plus its parts.
    open:   -(mean novelty of the space and gene views)
    target: ASAL's supervised target on the full picture (+ softmax term) + coef_oe * open loss"""
    nov_s, nov_g = novelty(z["sA"], z["sB"]), novelty(z["gA"], z["gB"])
    parts = dict(novelty_space=nov_s, novelty_genes=nov_g, asal_oe=asal_open_endedness(z["full"]))
    oe_loss = -0.5 * (nov_s + nov_g)
    if objective == "open":
        return oe_loss, parts
    t = asal_supervised_target(z["full"], z_txt)
    sm = asal_softmax_target(z["full"], z_txt) if coef_softmax else 0.0
    parts.update(target=-t, softmax=sm)
    return coef_prompt * t + coef_softmax * sm + coef_oe * oe_loss, parts


def final_embedding(z):
    """One vector per world for illumination: the last moment, both views, both halves."""
    v = np.concatenate([z["sA"][-1] + z["sB"][-1], z["gA"][-1] + z["gB"][-1]])
    return v / (np.linalg.norm(v) + 1e-12)


# ----------------------------------------------------------------------------- pictures
def strip(frames, n=8, gframes=None):
    """Pictures across the run, left to right; with gframes, the gene view as a second row."""
    pick = np.linspace(0, len(frames) - 1, n).astype(int)
    pad = np.full((frames.shape[1], 4, 3), 255, np.uint8)
    def row(fr):
        r = []
        for i in pick:
            r += [fr[i], pad]
        return np.concatenate(r[:-1], 1)
    top = row(frames)
    if gframes is None:
        return top
    return np.concatenate([top, np.full((4, top.shape[1], 3), 255, np.uint8), row(gframes)], 0)


def save_png(arr, path):
    from PIL import Image
    Image.fromarray(arr).save(path)


def save_gif(frames, path, ms=150):
    from PIL import Image
    ims = [Image.fromarray(f) for f in frames]
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=ms, loop=0)


# ----------------------------------------------------------------------------- evaluating worlds
def jobs_for(U, seeds, ticks, every, pop_cap, min_pop, burn_in=3000):
    return [dict(kw=AW.decode(u), seed=int(s), ticks=ticks, every=every, pop_cap=pop_cap, min_pop=min_pop,
                 burn_in=burn_in) for u in U for s in seeds]


def evaluate(pool, clip, U, seeds, ticks, every, pop_cap=12000, min_pop=200, burn_in=3000, **score_kw):
    """Run every world u on every seed (in parallel), score each run, average over seeds (ASAL's bs).
    Returns losses [len(U)], per-world info, and the first seed's rollout + embeddings of each world."""
    res = pool.map(AW.rollout, jobs_for(U, seeds, ticks, every, pop_cap, min_pop, burn_in), chunksize=1)
    losses, infos, firsts = [], [], []
    nb = len(seeds)
    for i in range(len(U)):
        ls, ps = [], []
        for j in range(nb):
            r = res[i * nb + j]
            z = embed_run(clip, r)
            l, p = score(z, **score_kw)
            ls.append(l); ps.append(p)
            if j == 0:
                firsts.append((r, z))
        losses.append(float(np.mean(ls)))
        info = {k: float(np.mean([p[k] for p in ps])) for k in ps[0]}
        rr = [res[i * nb + j] for j in range(nb)]
        info.update(status=";".join(x["status"] for x in rr), final_pop=int(np.mean([x["pops"][-1] for x in rr])),
                    max_gen=int(np.mean([x["gens"] for x in rr])), secs=float(np.mean([x["secs"] for x in rr])))
        infos.append(info)
    return losses, infos, firsts


def make_pool(jobs):
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
    import multiprocessing as mp
    return mp.get_context("spawn").Pool(jobs, initializer=AW.worker_init)
