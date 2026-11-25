"""
World modes for the fast engine: the kept stages of the roadmap, each with the
same settings as its original code, plus stage 13. Each mode is a function of
the arm ('on' = feature on) returning FastSim keyword arguments.

    stage1  social sensing on/off   (alife/: 16 hidden units, 7 actions, no signals)
    stage2  communication on/off    (alife_comm/: 24 hidden, 11 actions)
    stage3  seasons on/off          (stages/03_seasons: communication on)
    stage5  hazards on/off          (stages/05_hazards: communication + seasons on)
    stage13 shared food on/off      (stages/13_cofood: stage-5 founders)

Stages 1-5 start from 80 random brains and have no shared-food energy at all.
"""

BASE = dict(cofood=False, cofood_spawn_prob=0.0)

MODES = {
    "stage1": dict(name="01_social_sensing", prefix="ablation", ticks=20000, founders="random",
                   kw=lambda on: dict(BASE, social_sensing=on, communication=False, seasons=False,
                                      hazards=False, hidden=16, n_actions=7, init_inputs=52)),
    "stage2": dict(name="02_communication", prefix="comm", ticks=20000, founders="random",
                   kw=lambda on: dict(BASE, social_sensing=True, communication=on, seasons=False,
                                      hazards=False, hidden=24, n_actions=11, init_inputs=77)),
    "stage3": dict(name="03_seasons", prefix="seasons", ticks=25000, founders="random",
                   kw=lambda on: dict(BASE, social_sensing=True, communication=True, seasons=on,
                                      hazards=False, hidden=24, n_actions=11, init_inputs=77)),
    "stage5": dict(name="05_hazards", prefix="hazards", ticks=25000, founders="random",
                   kw=lambda on: dict(BASE, social_sensing=True, communication=True, seasons=True,
                                      hazards=on, hidden=24, n_actions=11, init_inputs=102)),
    "stage13": dict(name="13_cofood", prefix="cofood", ticks=25000, founders="stage5",
                    kw=lambda on: dict(social_sensing=True, communication=True, seasons=True,
                                       hazards=True, cofood=on)),
}

# ---------------------------------------------------------------------------
# SPARSE WORLD (128 x 128, food only in 16 drifting clusters; sensing radius 2).
# Base: social sensing and communication on, no seasons, no hazards; founders:
# the 300 evolved stage-5 agents, placed near the clusters. Each experiment
# switches one thing on/off.
SPARSE = dict(size=128, food_mode="patches", n_patches=16, patch_sigma=4.0, patch_speed=0.03,
              patch_turn=0.05, patch_peak=0.057, founders_near_patches=True,
              social_sensing=True, communication=True, seasons=False, hazards=False,
              cofood=False, cofood_spawn_prob=0.0, memory=False)

def _sparse(**over):
    d = dict(SPARSE); d.update(over); return d

MODES.update({
    "sparse_memory": dict(name="20_sparse_memory", prefix="memory", ticks=25000, founders="stage5",
                          kw=lambda on: _sparse(memory=on)),
    "sparse_comm": dict(name="21_sparse_comm", prefix="comm", ticks=25000, founders="stage5",
                        kw=lambda on: _sparse(communication=on)),
    "sparse_cofood": dict(name="22_sparse_cofood", prefix="cofood", ticks=25000, founders="stage5",
                          kw=lambda on: _sparse(cofood=on, cofood_spawn_prob=0.5)),
})

MODES.update({
    # language: every message has a second symbol; ON delivers it, OFF does not (same cost and choice)
    "sparse_language": dict(name="23_sparse_language", prefix="language", ticks=25000, founders="stage5",
                            kw=lambda on: _sparse(language=on)),
})

# ---------------------------------------------------------------------------
# SPARSE WORLD v2: information is worth more. 8 richer clusters (same total
# food), drifting faster, each vanishing at random (mean life 400 ticks) and
# reappearing elsewhere, so where food is has to be found again and again.
SPARSE2 = dict(SPARSE, n_patches=8, patch_peak=0.114, patch_speed=0.05, patch_life=400.0)

def _sparse2(**over):
    d = dict(SPARSE2); d.update(over); return d

MODES.update({
    "sparse2_memory": dict(name="30_sparse2_memory", prefix="memory", ticks=100000, founders="stage5",
                           kw=lambda on: _sparse2(memory=on)),
    "sparse2_comm": dict(name="31_sparse2_comm", prefix="comm", ticks=100000, founders="stage5",
                         kw=lambda on: _sparse2(communication=on)),
    "sparse2_language": dict(name="32_sparse2_language", prefix="language", ticks=100000, founders="stage5",
                             kw=lambda on: _sparse2(language=on)),
    # shared food worth more (8 energy) and brief (30 ticks): two agents must get there together
    "sparse2_cofood": dict(name="33_sparse2_cofood", prefix="cofood", ticks=100000, founders="stage5",
                           kw=lambda on: _sparse2(cofood=on, cofood_spawn_prob=0.5, cofood_value=8.0,
                                                  cofood_life=30)),
})

MODES.update({
    # fair memory test: founders' memory weights start at zero in BOTH arms
    "sparse2_memory0": dict(name="34_sparse2_memory0", prefix="memory", ticks=100000, founders="stage5",
                            kw=lambda on: _sparse2(memory=on, zero_memory_weights=True)),
})

# ---------------------------------------------------------------------------
# TERRAIN (sparse world v2 + walls): 16 rooms of 31 x 31 joined by 6-cell doorways.
# Food clusters vanish and reappear, often in a room an agent cannot see into.
# calibrated (6 seed-tests): 16 clusters (same total food), 8-cell doorways, clusters last ~600 ticks;
# with 8 clusters / 6-cell doorways the founders died out within ~3,000 ticks
TERRAIN = dict(SPARSE2, walls=True, wall_spacing=32, door_width=8, n_patches=16, patch_peak=0.057,
               patch_life=600.0)

def _terrain(**over):
    d = dict(TERRAIN); d.update(over); return d

MODES.update({
    "terrain_language": dict(name="35_terrain_language", prefix="language", ticks=100000, founders="stage5",
                             kw=lambda on: _terrain(language=on)),
})

# ---------------------------------------------------------------------------
# ARMS RACE (sparse world v2 + a second species): predators with their own
# evolving brains hunt the prey (predators.py). ON: predators evolve (mutation);
# OFF: predators reproduce as exact copies, so only the prey can change.
# Both arms: same founders (300 evolved stage-5 prey, 60 random-brain predators).
ARMS = dict(SPARSE2, predators=True, pred_delay=3000, catch_p=0.2)

def _arms(**over):
    d = dict(ARMS); d.update(over); return d

MODES.update({
    "arms_race": dict(name="40_arms_race", prefix="predevolve", ticks=1000000, founders="stage5",
                      kw=lambda on: _arms(pred_evolve=on)),
})

# ---------------------------------------------------------------------------
# ARMS RACE ROUND 2: predators evolve in both arms; newcomers (when predators fall
# below 10) are copies of recently dead predators, not random brains.
# ON: 3% of cells are shelters (predators cannot catch prey there; no food grows there;
# prey sense them in inputs 100-124). OFF: no shelters.
ARMS2 = dict(ARMS, pred_evolve=True, pred_immigrants="bank")

def _arms2(**over):
    d = dict(ARMS2); d.update(over); return d

MODES.update({
    "arms2_shelter": dict(name="41_arms2_shelter", prefix="shelter", ticks=1000000, founders="stage5",
                          kw=lambda on: _arms2(shelter_frac=0.03 if on else 0.0)),
    "arms2_shelter_c30": dict(name="42_arms2_shelter_c30", prefix="shelter", ticks=1000000, founders="stage5",
                              kw=lambda on: _arms2(shelter_frac=0.03 if on else 0.0, catch_p=0.3)),
})

# ---------------------------------------------------------------------------
# SEVERAL WAYS TO LIVE (ways.py): sparse world v2 plus buried food in fixed patches of
# rich soil, reached only by digging, and a heritable digestion trade-off (gut g:
# surface (1-g)^1.5, buried 0.5 + 0.5 g^1.5). ON: buried food exists. OFF: same trade-off, no
# buried food. Question: does the population split into surface eaters and diggers?
WAYS = dict(SPARSE2, buried=True, buried_base=0.5, gut_init_max=0.2, gut_mut=0.05, gut_q=1.5)

def _ways(**over):
    d = dict(WAYS); d.update(over); return d

MODES.update({
    "ways": dict(name="50_ways", prefix="buried", ticks=100000, founders="stage5",
                 kw=lambda on: _ways(buried=on)),
})

# ARMS RACE ROUND 2b: as round 2, but newcomers arrive slowly (one every 200 ticks while
# predators are below 10) and a catch is worth 2 energy instead of 3, so a predator crash gives prey a breathing window instead of
# keeping 10 skilled hunters on them (round 2 lost 6/20 worlds that way).
ARMS2B = dict(ARMS2, immigrant_every=200, kill_gain=2.0)

def _arms2b(**over):
    d = dict(ARMS2B); d.update(over); return d

MODES.update({
    "arms2b_shelter": dict(name="43_arms2b_shelter", prefix="shelter", ticks=1000000, founders="stage5",
                           kw=lambda on: _arms2b(shelter_frac=0.03 if on else 0.0)),
})

# ---------------------------------------------------------------------------
# AGENTS CHANGE THE WORLD, part 1: FOOD STORAGE (storage.py). Sparse world v2 with
# seasons (lean winters make saving worthwhile). ON: "mark" stores up to 0.5 energy in the
# cell, "mate" takes up to 0.5 back; stores persist (spoil 0.1%/tick) and are sensed in
# inputs 150-174. OFF: same actions and costs, no effect.
STORE = dict(SPARSE2, storage=True, seasons=True)

def _store(**over):
    d = dict(STORE); d.update(over); return d

MODES.update({
    "storage": dict(name="60_storage", prefix="storage", ticks=100000, founders="stage5",
                    kw=lambda on: _store(storage=on)),
})

# ---------------------------------------------------------------------------
# AGENTS CHANGE THE WORLD, part 2: BUILDING SHELTERS (building.py). The round-2b arms-race
# world (evolving predators, slow newcomers, 2 energy per catch), no placed shelters.
# ON: the "mark" action builds a shelter on the creature's cell (1.5 energy); shelters
# crumble with probability 1/1000 per tick. OFF: the action does nothing.
BUILD = dict(ARMS2B, build=True)

def _build(**over):
    d = dict(BUILD); d.update(over); return d

MODES.update({
    "build": dict(name="70_build", prefix="build", ticks=300000, founders="stage5",
                  kw=lambda on: _build(build=on)),
})

# ---------------------------------------------------------------------------
# BIGGER BRAINS WHEN NEEDED (brains.py): brain size is heritable. Room for 48 hidden
# units; founders use 24 (their evolved brains, padded with unused zero units); a child
# gains or loses one unit with probability 0.05; each unit above 24 costs 0.0002 energy
# per tick (each unit below 24 saves the same). Brain size evolves in both arms. ON: extra
# units can be wired in by mutation. OFF: extra units stay unwired (pure cost), so brain
# growth there is drift; growth beyond it means bigger brains are needed.
# brain_ways: in the several-ways-to-live world; brain_arms: in the arms race 2b world.
MODES.update({
    "brain_ways": dict(name="80_brain_ways", prefix="brain", ticks=300000, founders="stage5",
                       kw=lambda on: dict(WAYS, brain_evolve=True, brain_wired=on)),
    "brain_arms": dict(name="81_brain_arms", prefix="brain", ticks=300000, founders="stage5",
                       kw=lambda on: dict(ARMS2B, brain_evolve=True, brain_wired=on)),
})

# ---------------------------------------------------------------------------
# COMBINATION: SEVERAL WAYS TO LIVE + EVOLVING PREDATORS. The `ways` world (buried food,
# gut trade-off) with the round-2b predators (evolving, slow newcomers, 2 energy per catch).
# ON: buried food (two ways to live) with predators. OFF: no buried food, with predators.
# Compare with `ways` (no predators): does predation break the split, or strengthen it?
WAYS_PRED = dict(WAYS, predators=True, pred_delay=3000, catch_p=0.2, pred_evolve=True,
                 pred_immigrants="bank", immigrant_every=200, kill_gain=2.0)

MODES.update({
    "ways_pred": dict(name="51_ways_pred", prefix="buried", ticks=300000, founders="stage5",
                      kw=lambda on: dict(WAYS_PRED, buried=on)),
})

# FOOD STORAGE, private versions: public stores were raided (tragedy of the commons: storing
# was selected against). Here only the owner (storage_private) or its family: parent,
# children, siblings (storage_family) can take from a store, and nobody can add to a
# store someone else owns. ON: private storage. OFF: storage does nothing.
MODES.update({
    "storage_private": dict(name="61_storage_private", prefix="storage", ticks=100000, founders="stage5",
                            kw=lambda on: _store(storage=on, store_access="owner")),
    "storage_family": dict(name="62_storage_family", prefix="storage", ticks=100000, founders="stage5",
                           kw=lambda on: _store(storage=on, store_access="family")),
})

# ---------------------------------------------------------------------------
# REPLICATION 1: STEPPING STONES (after Lenski, Ofria, Pennock & Adami 2003, Nature: in Avida
# the complex EQU function evolved only when simpler functions were also rewarded).
# Buried food pays in full only after 4 digs in a row on the same cell (a behaviour
# sequence). ON: each earlier dig pays 1/4 (stepping stones). OFF: only the full sequence pays.
# No digestion trade-off (everyone digests both foods fully), so the only question is
# whether the 4-step sequence evolves. 100k ticks.
# Buried food is sensed on the same 0-1 scale as surface food (buried_sense_scale 0.5; full
# buried cells hold 2.0). At full scale the control died out early on 3/3 test seeds: the
# founders' weights for this sense start at zero, but every child's mutations put noise on
# them, and a strong, useless signal times that noise is a real cost. At half scale seeds 1-2
# survive.
STEPS = dict(SPARSE2, buried=True, buried_base=1.0, gut_init_max=0.0, gut_mut=0.0, dig_steps=4,
             buried_sense_scale=0.5)

MODES.update({
    "steps": dict(name="90_steps", prefix="steps", ticks=100000, founders="stage5",
                  kw=lambda on: dict(STEPS, dig_partial=0.25 if on else 0.0)),
})

# REPLICATION 1b, STEPPING STONES WITH DIFFERENT ACTIONS (our twist on Lenski et al. 2003):
# buried food pays in full only after the sequence dig (mark), stay, take (mate), eat on one cell,
# in that order - four different skills, closer to Avida's EQU than one action repeated.
# ON: each correct step after the first pays 1/4 (the first alone pays nothing, else digging
# forever wins). OFF: only the whole sequence pays.
# Gifts break even here (share_multiplier 1.0; see engine.SHARE_MULTIPLIER).
MODES.update({
    "steps_mixed": dict(name="91_steps_mixed", prefix="steps", ticks=100000, founders="stage5",
                        kw=lambda on: dict(STEPS, dig_steps=1, dig_sequence=("mark", "stay", "mate", "eat"),
                                           dig_partial=0.25 if on else 0.0, share_multiplier=1.0)),
})

# MEMORY UNDER PRESSURE: does working memory evolve once a problem needs it?
# Buried food pays in full only after dig (mark), stay, eat on one cell, in that order. Standing
# on one cell the senses barely change, so a purely reactive brain can only hit the order by
# chance; memory (hidden state carried to the next tick, starting at zero weights) can track
# the step. Two modes x two arms = 4 arms:
#   memseq_mem    memory ON;  ON: stepping stones (step 2 pays 1/4), OFF: only the whole sequence pays
#   memseq_nomem  memory OFF; same two arms
# Gifts break even. 200k ticks.
MEMSEQ = dict(STEPS, dig_steps=1, dig_sequence=("mark", "stay", "eat"), share_multiplier=1.0)

MODES.update({
    "memseq_mem": dict(name="92_memseq_mem", prefix="memseq", ticks=200000, founders="stage5",
                       kw=lambda on: dict(MEMSEQ, memory=True, zero_memory_weights=True,
                                          dig_partial=0.25 if on else 0.0)),
    "memseq_nomem": dict(name="93_memseq_nomem", prefix="memseq", ticks=200000, founders="stage5",
                         kw=lambda on: dict(MEMSEQ, memory=False, dig_partial=0.25 if on else 0.0)),
})
