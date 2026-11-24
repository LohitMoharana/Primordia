# Autopilot protocol

The user asked: after each evolution experiment, automatically implement and
launch the most appropriate next change. This file is the standing procedure
every scheduled check follows. Read it first, every time.

## Hard rules (never broken, whatever a stage seems to need)
- Everything stays inside this sealed sandbox. No network access for agents,
  no agent-written code, no agents modifying the simulator, nothing on the
  user's own computer. Agents are small neural nets in a grid, nothing more.
- One new variable per stage, always with an on/off ablation flag. The brain
  architecture must be identical in both arms; "off" zeroes the information
  or disables the mechanism, it does not shrink the brain.
- Never edit a stage's code directory while it has live or unfinished runs.
  Each new stage gets a fresh copy: lab/stages/<NN>_<name>/.
- Only 2 CPU cores: run ONE on/off pair at a time.
- Tell the user at every stage transition: result, decision, what launched.

## Each check
1. `python3 /home/claude/lab/supervisor.py status` to see the state of the
   active stage.
2. If runs are dead and unfinished, relaunch them all at once (Bash,
   run_in_background):  `python3 /home/claude/lab/supervisor.py run`
   This resumes each run bit-exactly from its latest checkpoint up to the
   stage's target tick, prunes old checkpoints, and writes DONE when finished.
3. Evaluate once BOTH arms reach generation >= 60:
   `python3 /home/claude/lab/evaluate.py <stage_name>`
   Take at least 2 evaluations at different depths before calling a result,
   because single checks have flipped before.
4. When a stage is decided, or its runs reach the target tick:
   - Append the result to FINDINGS.md (numbers from every check, verdict).
   - Capability stages (sensing, communication, memory, kin recognition...):
     adopt into the base (flag on in all later stages) only if the ON arm
     beats OFF on survival in a consistent direction, with no population
     collapse. Otherwise keep it off and note why.
   - Environment stages (seasons, hazards, predators): these are meant to
     make the world harder, so lower survival is expected. Adopt if the ON
     population stays sustainable (no collapse, no drift to extinction) and
     the arms differ in behaviour, because that pressure is what later
     capability stages are tested against. Reject if it collapses.
   - Mark the stage "done" in stages.json, pick the next roadmap item (adjust
     the order if findings suggest it), build it in a new stage directory,
     smoke-test both arms (about 500 ticks, must run and arms must differ in
     the intended way), register it in stages.json as "running", launch with
     supervisor.py run.
4b. If a run is marked EXTINCT: the stage cannot be compared. If it died
   before selection had time to act (early generations), restart BOTH arms
   with a larger founding population; if it died late, that is a result
   (the condition is unsustainable): record it and decide the stage on it.
5. Schedule the next check in about 25 minutes (send_later), pointing back to
   this file.

## Founding populations
Stages 1-5 started from random brains. From stage 6 on, each stage is seeded
with the SAME evolved agents (identical in both arms) from the previous
stage's adopted arm, with new inputs wired in at zero weight and new actions
starting rare. Reason: the world is now harsh enough (seasons + poison) that
random founders went extinct three times in stage 6. Generation counters
restart at 0 per stage, so "generation 60" still means within the stage.

## Roadmap (ordered; revisable with justification in FINDINGS.md)
1. Social sensing (see crowding)           — DONE, adopted
2. Communication channel (4 symbols)       — DONE, adopted
3. Seasons (food regrowth cycles, same average food in both arms) — DONE, adopted (environment)
4. Recurrent memory (hidden state carried between ticks) — DONE, rejected (retry with a gated design later): tests whether
   agents can anticipate seasons instead of just reacting
5. Hazards (poison cells that drain energy) — DONE, adopted (environment)
6. Kin recognition + share action — DONE, inconclusive (share cost >= benefit, so altruism could not evolve)
6b. Kin recognition + productive share (receiver gets 2x what giver gives) — DONE, not adopted (no consistent effect; retry with chosen recipient later)
7. Sexual reproduction (two parents, crossover) — DONE, not adopted (crossover breaks co-adapted weights)
8. Predators (a second species that eats agents) — DONE, not adopted (sustainable, but no behavioural response at mild predation)

Added by the user's request (21 Nov, 20:57):
9. World markers / external memory (CAPABILITY) — DONE, not adopted (no trails; marks carried no information). A "mark" action drops a
   lasting trace on the agent's cell that fades slowly (for example, halves
   every ~200 ticks); a new 5x5 marker sensing patch. OFF = marking allowed
   and costed, but the patch reads zero. Measure: do marks cluster near food,
   and do agents move toward marks? Internal memory failed in stage 4; this
   tests the ant-trail alternative.
10. Lifetime learning (CAPABILITY) — DONE, not adopted (survival inconsistent; learning rate selected to stay low). During its life, an agent nudges the
    output weights of the action it just took up or down by a small amount
    depending on whether its energy rose or fell (a simple reward-modulated
    rule, learning rate itself heritable). Children inherit only the genome
    weights, not what the parent learned. OFF = learning rate forced to 0.
    Measure: survival, and whether the heritable learning rate rises in ON.
11. Self-prediction — running (CAPABILITY, marker for self-modelling, NOT a test of
    consciousness). An extra output predicts the agent's own energy next tick.
    Prediction accuracy has to earn its place: ON pays a small energy cost
    proportional to prediction error. OFF = the output exists but has no cost
    and no effect. Measure: does prediction error fall over generations in ON
    relative to OFF?
12. Richer language (CAPABILITY). Signals become two-symbol messages (4x4 = 16
    combinations); receivers sense both symbols. OFF = same channel, but only
    the first symbol is delivered. Measure: do two-symbol messages carry
    information beyond the first symbol (does the second symbol depend on the
    sender's situation, e.g. direction of food)?
13. Two food types (ENVIRONMENT). Add a high-value food that can only be
    eaten when two agents eat it on the same tick, alongside the normal food.
    OFF = only normal food (same average energy available). Measure: does
    cooperative eating evolve?

Measurement tools (infrastructure, not stages; build into every new stage's
code starting with the next one built, never into a stage with live runs):
- Family trees: append (tick, child_id, parent_id(s), generation) to
  births.csv in each run directory, so lineages can be reconstructed.
- Novelty meter: every 1000 ticks, log the population's action-share vector
  and its distance from the previous one; a long stretch near zero means the
  world has plateaued (the signal for moving to a richer world).

Stop and ask the user before anything not on this list, and when the roadmap
runs out.
