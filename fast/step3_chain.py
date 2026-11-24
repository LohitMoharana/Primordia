"""Terrain seeds 6-10 (100k), then step 3: every surviving terrain run extended to 1,000,000 ticks (saves every 50k)."""
import subprocess, sys, os
H = os.path.dirname(os.path.abspath(__file__)); py = sys.executable
subprocess.run([py, os.path.join(H, "replicate_modes.py"), "--modes", "terrain_language", "--seeds", *map(str, range(6, 11)), "--jobs", "14"], check=True)
print("=== STEP 3: extending surviving terrain runs to 1,000,000 ticks ===", flush=True)
subprocess.run([py, os.path.join(H, "replicate_modes.py"), "--modes", "terrain_language", "--seeds", *map(str, range(1, 11)),
                "--jobs", "14", "--ticks", "1000000", "--checkpoint-every", "50000"], check=True)
print("STEP 3 RUNS DONE", flush=True)
