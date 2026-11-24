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
