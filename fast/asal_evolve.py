"""
ASAL++ on Primordia: an evolver watches the best world and proposes the next target.

After Baid et al. 2025 (github.com/fredericowieser/ASALPlusPlus). Each iteration:
  1. search world settings for the current target(s) (asal_search.run_search, ASAL's
     supervised target), warm-started from the previous iteration's best world;
  2. picture the best world (8 moments; top row where creatures stand, bottom row
     their genes);
  3. the evolver looks at the picture and the targets so far and proposes the next target.

  --mode ett  Evolved Temporal Targets: the run must follow the whole target history in
              order (early moments like target 1, late moments like the newest). For an
              evolving world this asks for a history: one stage leading to the next.
  --mode est  Evolved Supervised Targets: only the newest target counts.

  --evolver vlm   a vision-language model (Hugging Face transformers; default
                  Qwen2.5-VL-3B-Instruct, not gated). ASAL++ used Gemma-3-4b-it
                  (gated: needs a Hugging Face login): --vlm-model google/gemma-3-4b-it
  --evolver file  write the picture and question to <out>/evolver/ and wait for a reply
                  file (a person, or Claude, can be the evolver)

    python fast/asal_evolve.py --out fast/runs/asal_ett --first "scattered single cells" --iters 8 --gens 25

Resumable: history.json keeps every target and best world; each iteration's search resumes.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np

import asal_core as AC
import asal_search as AS
import asal_world as AW

INSTRUCTION = """You are guiding the evolution of an artificial world. The picture shows one run of it, \
left to right in time. Top row: where every creature stands; each creature's colour comes from its genes, \
so a new colour is a new kind of creature; red dots are predators. Bottom row: the same creatures placed \
by their genes, so separate clouds are separate species.

The world was tuned to follow these targets, in order: {history}.
Treat them as stages already reached. You are at iteration {i}. Propose the NEXT TARGET: a short \
description (a few words) of what this world of creatures should look like next, clearly different \
from the targets so far, lifelike, and a step a living world could really take. \
Output only the target, nothing else."""


class VLMEvolver:
    def __init__(self, model_id, device=None):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map=device or "auto")
        self.model.eval()

    def propose(self, png, text):
        from PIL import Image
        im = Image.open(png).convert("RGB")
        if im.width > 1280:
            im = im.resize((1280, int(im.height * 1280 / im.width)))
        msgs = [{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": text}]}]
        inp = self.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                            return_dict=True, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**inp, max_new_tokens=32, do_sample=True, temperature=0.8)
        return self.proc.decode(out[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True)


class FileEvolver:
    def __init__(self, folder, poll=10):
        self.folder, self.poll = folder, poll
        os.makedirs(folder, exist_ok=True)

    def propose(self, png, text, i):
        req = os.path.join(self.folder, f"request_{i}.txt")
        rep = os.path.join(self.folder, f"reply_{i}.txt")
        with open(req, "w") as f:
            f.write(f"picture: {png}\n\n{text}\n")
        print(f"waiting for the evolver: read {req}, write the next target to {rep}", flush=True)
        while not os.path.exists(rep):
            time.sleep(self.poll)
        return open(rep).read()


def clean(t):
    t = t.strip().splitlines()[0] if t.strip() else ""
    for p in ("NEXT TARGET:", "Next target:", "next target:"):
        t = t.replace(p, "")
    return t.strip().strip('"\'*`. ').strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "asal_ett"))
    ap.add_argument("--first", default="scattered single cells", help="the first target")
    ap.add_argument("--iters", type=int, default=8, help="ASAL++ used 8")
    ap.add_argument("--mode", choices=["ett", "est"], default="ett")
    ap.add_argument("--evolver", choices=["vlm", "file"], default="vlm")
    ap.add_argument("--vlm-model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--x0", default=None, help="first search starts here (JSON list), e.g. an open-endedness winner")
    AS.add_args(ap)
    ap.set_defaults(gens=25, coef_softmax=1.0)  # softmax term keeps the stages distinct (ASAL's extension)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    hist_path = os.path.join(a.out, "history.json")
    if os.path.exists(hist_path):
        hist = json.load(open(hist_path))
    else:
        hist = dict(targets=[a.first], best=[], config=vars(a))
    clip = AC.Clip(a.clip_weights, a.device)
    evolver = None
    log = lambda s: print(s, flush=True)
    with AC.make_pool(a.jobs) as pool:
        while len(hist["best"]) < a.iters:
            i = len(hist["best"])
            targets = hist["targets"][: i + 1]
            prompts = targets if a.mode == "ett" else targets[-1:]
            x0 = hist["best"][-1]["u"] if hist["best"] else (json.loads(a.x0) if a.x0 else None)
            log(f"=== iteration {i}: targets {prompts}")
            best = AS.run_search(a, os.path.join(a.out, f"iter_{i:02d}"), clip, pool, "target", prompts, x0, log)
            png = os.path.join(a.out, f"iter_{i:02d}", "best_strip.png")
            hist["best"].append(dict(u=[float(v) for v in best["u"]], loss=best["loss"], world=AW.describe(best["u"]),
                                     info={k: v for k, v in best["info"].items()}))
            if len(hist["targets"]) <= i + 1 and i + 1 < a.iters:
                text = INSTRUCTION.format(history="; ".join(f"'{t}'" for t in hist["targets"]), i=i + 1)
                if a.evolver == "file":
                    nxt = clean(FileEvolver(os.path.join(a.out, "evolver")).propose(png, text, i + 1))
                else:
                    evolver = evolver or VLMEvolver(a.vlm_model)
                    nxt = clean(evolver.propose(png, text))
                hist["targets"].append(nxt or "a different kind of creature")
                log(f"evolver proposes: '{hist['targets'][-1]}'")
            with open(hist_path + ".tmp", "w") as f:
                json.dump(hist, f, indent=1)
            os.replace(hist_path + ".tmp", hist_path)
    log("done. targets: " + " -> ".join(hist["targets"]))


if __name__ == "__main__":
    main()
