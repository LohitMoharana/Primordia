"""
Compiled kernels (Numba) for the fast engine: sensing + brain + action
sampling fused into one parallel loop over agents.

For each agent it reads the 5x5 patches straight from the world grids and
adds weight rows only for inputs that are non-zero (most are zero: empty
cells, no signals, no poison), so it never builds the 227-wide input vector.
Same maths as engine.FastSim.sense + think, including the stage-13 input
layout, so results are identical up to float rounding.

If Numba is missing, HAVE_NUMBA is False and the engine uses its NumPy path.
"""
import numpy as np

try:
    from numba import njit, prange
    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False

if HAVE_NUMBA:
    @njit(parallel=True, fastmath=False, cache=True)
    def sense_think(idx, xs, ys, energy, age, food, occ, sig, haz, cof,
                    social, w1, b1, w2, b2, u, max_energy, max_age, out_act,
                    memory, wh, hstate, sig2, ws, bs, u2, sig0, nsym, out_sym2, wall, has_wall):
        S = food.shape[0]
        n = idx.shape[0]
        H = b1.shape[1]
        O = b2.shape[1]
        # scratch rows allocated once (allocating inside the parallel loop is slow)
        h_all = np.empty((n, H), np.float32)
        out_all = np.empty((n, O), np.float64)
        acc_all = np.empty((n, O), np.float32)
        cx_all = np.empty((n, 25), np.int64)
        cy_all = np.empty((n, 25), np.int64)
        rare = np.empty((S, S), np.bool_)
        for p in range(S):
            for q in range(S):
                rare[p, q] = sig[p, q] != 0.0 or haz[p, q] != 0.0 or cof[p, q] > 0.0 or sig2[p, q] != 0.0
        for k in prange(n):
            a = idx[k]
            x = xs[k]
            y = ys[k]
            h = h_all[k]
            b1r = b1[a]
            for j in range(H):
                h[j] = b1r[j]
            cxs = cx_all[k]
            cys = cy_all[k]
            c = 0
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    cxs[c] = (x + dx) % S
                    cys[c] = (y + dy) % S
                    c += 1
            # one channel at a time, so each channel's 25 weight rows are read in
            # one contiguous run (and in the same order as the NumPy path)
            for c in range(25):
                v = np.float32(food[cxs[c], cys[c]])
                if v != 0.0:
                    wr = w1[a, c]
                    for j in range(H):
                        h[j] += v * wr[j]
            if social:
                for c in range(25):
                    v = occ[cxs[c], cys[c]]
                    if c == 12:
                        v = v - 1.0
                        if v < 0.0:
                            v = 0.0
                    if v != 0.0:
                        vf = np.float32(v)
                        wr = w1[a, 25 + c]
                        for j in range(H):
                            h[j] += vf * wr[j]
            # signals, poison and shared food are rare: one combined check per
            # cell, then the three channels only where something is there.
            # Channel order (signal, then poison, then shared food) is kept.
            for c in range(25):
                if rare[cxs[c], cys[c]]:
                    v = sig[cxs[c], cys[c]]
                    if v != 0.0:
                        vf = np.float32(v)
                        wr = w1[a, 50 + c]
                        for j in range(H):
                            h[j] += vf * wr[j]
            for c in range(25):
                if rare[cxs[c], cys[c]]:
                    v = haz[cxs[c], cys[c]]
                    if v != 0.0:
                        vf = np.float32(v)
                        wr = w1[a, 75 + c]
                        for j in range(H):
                            h[j] += vf * wr[j]
            if has_wall:  # terrain: walls nearby (inputs 125-149, the unused predator slot)
                for c in range(25):
                    if wall[cxs[c], cys[c]] != 0.0:
                        wr = w1[a, 125 + c]
                        for j in range(H):
                            h[j] += wr[j]
            for c in range(25):
                if rare[cxs[c], cys[c]]:
                    v = sig2[cxs[c], cys[c]]
                    if v != 0.0:
                        vf = np.float32(v)
                        wr = w1[a, 175 + c]
                        for j in range(H):
                            h[j] += vf * wr[j]
            for c in range(25):
                if rare[cxs[c], cys[c]]:
                    if cof[cxs[c], cys[c]] > 0.0:
                        wr = w1[a, 200 + c]
                        for j in range(H):
                            h[j] += wr[j]
            if memory:
                # working memory: last tick's hidden state feeds back through wh
                hp = hstate[a]
                for i in range(H):
                    hpi = hp[i]
                    if hpi != 0.0:
                        whr = wh[a, i]
                        for j in range(H):
                            h[j] += hpi * whr[j]
            e = np.float32(energy[k] / max_energy)
            g = np.float32(age[k] / max_age)
            for j in range(H):
                # tanh via exp (same value to ~1 ulp, several times cheaper here)
                z = h[j] + e * w1[a, 225, j] + g * w1[a, 226, j]
                h[j] = np.float32(1.0) - np.float32(2.0) / (np.exp(np.float32(2.0) * z) + np.float32(1.0))
            if memory:
                for j in range(H):
                    hstate[a, j] = h[j]
            # output layer + softmax sampling
            out = out_all[k]
            acc = acc_all[k]
            for o in range(O):
                acc[o] = b2[a, o]
            # j outer, o inner: 14 independent running sums (each still adds
            # its terms in j order, so the result is bit-identical)
            for j in range(H):
                hj = h[j]
                w2r = w2[a, j]
                for o in range(O):
                    acc[o] += hj * w2r[o]
            m = -1e300
            for o in range(O):
                s = acc[o]
                out[o] = s
                if s > m:
                    m = s
            tot = 0.0
            for o in range(O):
                out[o] = np.exp(out[o] - m)
                tot += out[o]
            r = u[k] * tot
            cum = 0.0
            act = O - 1
            for o in range(O):
                cum += out[o]
                if cum >= r:
                    act = o
                    break
            out_act[k] = act
            # second symbol of a message: sampled from the second-symbol head
            sym = 0
            if act >= sig0 and act < sig0 + nsym:
                Q = bs.shape[1]
                mq = -1e300
                lg = np.empty(Q, np.float64)
                for q in range(Q):
                    s2 = np.float32(bs[a, q])
                    for j in range(H):
                        s2 += h[j] * ws[a, j, q]
                    lg[q] = s2
                    if s2 > mq:
                        mq = s2
                tq = 0.0
                for q in range(Q):
                    lg[q] = np.exp(lg[q] - mq)
                    tq += lg[q]
                rq = u2[k] * tq
                cq = 0.0
                sym = Q
                for q in range(Q):
                    cq += lg[q]
                    if cq >= rq:
                        sym = q + 1
                        break
            out_sym2[k] = sym


if HAVE_NUMBA:
    @njit(cache=True)
    def act_phase(act, xs, ys, e, age, food, hazard, keys, su, S,
                  A_STAY, A_N, A_S, A_E, A_W, A_EAT, A_REP, A_MATE, A_MARK, A_SHARE, SIG0, NSYM,
                  IDLE, MOVE, SIGC, MARKC, EAT_GAIN, REP_MIN, REP_COST, SHARE_COST, SHARE_AMT,
                  HAZ, MAX_E, MAX_AGE, rep_out, dead_out, recv_out, wall):
        """Everything an agent's chosen action does this tick, for all agents.

        Inputs are local arrays (one entry per alive agent, in idx order).
        xs, ys are updated in place to the new positions; e (energy) in place;
        food in place. keys: random tie-break order for eating in a shared cell;
        su: random number used to pick a gift's receiver. Outputs: rep_out
        (1 = reproduces), dead_out (1 = dies), recv_out (local index of the gift
        receiver, -1 = none). Same rules and same order as the NumPy path.
        """
        n = act.shape[0]
        x0 = xs.copy()
        y0 = ys.copy()
        for k in range(n):
            a = act[k]
            if a == A_STAY or a == A_EAT or a == A_REP or a == A_MATE:
                e[k] -= IDLE
            elif a >= A_N and a <= A_W:
                if a == A_E:
                    xs[k] = (x0[k] + 1) % S
                elif a == A_W:
                    xs[k] = (x0[k] - 1) % S
                elif a == A_S:
                    ys[k] = (y0[k] + 1) % S
                else:
                    ys[k] = (y0[k] - 1) % S
                if wall[xs[k], ys[k]] != 0.0:  # walking into a wall: stay put (still costs the move)
                    xs[k] = x0[k]
                    ys[k] = y0[k]
                e[k] -= MOVE
            elif a >= SIG0 and a < SIG0 + NSYM:
                e[k] -= SIGC
            elif a == A_MARK:
                e[k] -= MARKC
        # eating: in random-key order, each eater takes up to EAT_GAIN from its cell
        ne = 0
        for k in range(n):
            if act[k] == A_EAT:
                ne += 1
        eaters = np.empty(ne, np.int64)
        ekeys = np.empty(ne, np.float64)
        j = 0
        for k in range(n):
            if act[k] == A_EAT:
                eaters[j] = k
                ekeys[j] = keys[k]
                j += 1
        order = np.argsort(ekeys)
        for q in range(ne):
            k = eaters[order[q]]
            avail = food[x0[k], y0[k]]
            g = avail if avail < EAT_GAIN else EAT_GAIN
            if g < 0.0:
                g = 0.0
            food[x0[k], y0[k]] = avail - g
            e[k] += g
        # reproduction
        for k in range(n):
            rep_out[k] = 0
            if act[k] == A_REP and e[k] >= REP_MIN:
                e[k] -= REP_COST
                rep_out[k] = 1
        # sharing: everyone pays the cost; a gift needs a neighbour (radius 1,
        # start-of-tick positions) and energy above the gift amount
        nshare = 0
        for k in range(n):
            recv_out[k] = -1
            if act[k] == A_SHARE:
                e[k] -= SHARE_COST
                nshare += 1
        if nshare > 0:
            cnt = np.zeros(S * S + 1, np.int64)
            for k in range(n):
                cnt[x0[k] * S + y0[k] + 1] += 1
            for c in range(S * S):
                cnt[c + 1] += cnt[c]
            fill = cnt[:-1].copy()
            members = np.empty(n, np.int64)
            for k in range(n):  # increasing k: same order as a stable sort by cell
                c = x0[k] * S + y0[k]
                members[fill[c]] = k
                fill[c] += 1
            cands = np.empty(n, np.int64)
            for k in range(n):
                if act[k] != A_SHARE or e[k] <= SHARE_AMT:
                    continue
                m = 0
                for ddx in range(-1, 2):
                    for ddy in range(-1, 2):
                        c = ((x0[k] + ddx) % S) * S + (y0[k] + ddy) % S
                        for p in range(cnt[c], cnt[c + 1]):
                            b = members[p]
                            if b != k:
                                cands[m] = b
                                m += 1
                if m > 0:
                    recv_out[k] = cands[int(su[k] * m)]
            for k in range(n):
                if recv_out[k] >= 0:
                    e[k] -= SHARE_AMT
        # poison where the agent ended up; cap; deaths
        for k in range(n):
            if hazard[xs[k], ys[k]] > 0.0:
                e[k] -= HAZ
            if e[k] > MAX_E:
                e[k] = MAX_E
            dead_out[k] = 1 if (e[k] <= 0.0 or age[k] >= MAX_AGE) else 0


if HAVE_NUMBA:
    @njit(cache=True)
    def prepare(idx, x, y, last_action, age, S, communication, SIG0, NSYM, occ, sig, xs, ys, ages,
                language, last_sym2, sig2):
        """Occupancy and last tick's signals from alive agents, ages += 1, and the
        per-agent position/age arrays the other kernels use (one pass, no temporaries).
        Signals: when several agents in one cell signalled, the last one (in idx
        order) wins, as with NumPy fancy assignment."""
        for c in range(S):
            for d in range(S):
                occ[c, d] = 0.0
                sig[c, d] = 0.0
                sig2[c, d] = 0.0
        for k in range(idx.shape[0]):
            a = idx[k]
            xs[k] = x[a]
            ys[k] = y[a]
            occ[x[a], y[a]] += 1.0
            age[a] += 1
            ages[k] = age[a]
            if communication:
                la = last_action[a]
                if la >= SIG0 and la < SIG0 + NSYM:
                    sig[x[a], y[a]] = la - SIG0 + 1
                    if language:
                        sig2[x[a], y[a]] = last_sym2[a]

    @njit(cache=True)
    def resolve_cofood(eat_slots, ex, ey, alive, energy, cof, S, max_e, stats):
        """Shared food: in a cell with 2+ alive eaters this tick, they split it.
        stats: [eaten, co_eaters, solo_attempts] increments."""
        n = eat_slots.shape[0]
        cnt = np.zeros(S * S, np.int64)
        touched = np.empty(n, np.int64)
        nt = 0
        for k in range(n):
            if cof[ex[k], ey[k]] > 0.0 and alive[eat_slots[k]]:
                c = ex[k] * S + ey[k]
                if cnt[c] == 0:
                    touched[nt] = c
                    nt += 1
                cnt[c] += 1
        for q in range(nt):
            c = touched[q]
            m = cnt[c]
            cx, cy = c // S, c % S
            if m >= 2:
                share = cof[cx, cy] / m
                for k in range(n):
                    if ex[k] == cx and ey[k] == cy and alive[eat_slots[k]]:
                        v = energy[eat_slots[k]] + share
                        energy[eat_slots[k]] = v if v < max_e else max_e
                cof[cx, cy] = 0.0
                stats[0] += 1
                stats[1] += m
            else:
                stats[2] += m


if HAVE_NUMBA:
    @njit(cache=True)
    def food_decay(food, decay_r, decay_p):
        S = food.shape[0]
        for p in range(S):
            for q in range(S):
                if decay_r[p, q] < decay_p:
                    food[p, q] *= 0.5

    @njit(cache=True)
    def food_regrow(food, regrow_r, regrow_p, max_food):
        S = food.shape[0]
        for p in range(S):
            for q in range(S):
                f = food[p, q]
                if f < 0.05 and regrow_r[p, q] < regrow_p:
                    f = max_food
                if f < 0.0:
                    f = 0.0
                elif f > max_food:
                    f = max_food
                food[p, q] = f


if HAVE_NUMBA:
    @njit(cache=True)
    def food_regrow_map(food, regrow_r, pmap, factor, max_food):
        """Regrowth with a per-cell probability map (sparse world): a cell regrows
        when it is nearly empty and its random draw is below pmap * factor."""
        S = food.shape[0]
        for p in range(S):
            for q in range(food.shape[1]):
                f = food[p, q]
                if f < 0.05 and regrow_r[p, q] < pmap[p, q] * factor:
                    f = max_food
                if f < 0.0:
                    f = 0.0
                elif f > max_food:
                    f = max_food
                food[p, q] = f
