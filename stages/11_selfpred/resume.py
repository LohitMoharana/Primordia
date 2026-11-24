"""
Resume a run from a checkpoint, exactly where it left off — same agents,
same brains, same tick count, same evolutionary history. Nothing restarts.

Usage:
    python3 resume.py runs/social1/checkpoint_14000.pkl runs/social1 --ticks 20000

The checkpoint's world and all its agents (with their evolved brains) are
restored as-is. If you also copied the matching log.csv into the same
out_dir, logging continues in that same file rather than starting a new one.
"""
import argparse
from sim import Simulation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="path to the .pkl checkpoint file")
    parser.add_argument("out_dir", help="directory to continue writing logs/checkpoints into "
                                         "(put the matching log.csv here first if you have it)")
    parser.add_argument("--ticks", type=int, default=20000,
                         help="how many MORE ticks to run from this point")
    args = parser.parse_args()

    sim = Simulation.from_checkpoint(args.checkpoint, args.out_dir)
    print(f"Loaded: tick={sim.tick}, population={len(sim.agents)}, "
          f"max_generation={max((a.generation for a in sim.agents), default=0)}")
    sim.run(max_ticks=args.ticks)
