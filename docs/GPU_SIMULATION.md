# GPU simulation on the RTX 3060

This host uses MuJoCo MJX rather than Isaac Sim. Isaac Sim 5.x officially
requires at least 32 GB system RAM and 16 GB VRAM; this WSL2 machine has 16 GB
RAM and a 12 GB RTX 3060. MJX preserves the articulated G1 dynamics and runs
batched physics on CUDA with much lower overhead.

```bash
scripts/sim/gpu-setup.sh
scripts/sim/gpu-run.sh doctor
scripts/sim/gpu-run.sh benchmark --envs 256 --steps 500
```

The process disables JAX memory preallocation and uses a 50% memory-fraction
hint. Since other processes already consume roughly 3.8 GB, keep total GPU
usage below 10 GB and reduce `--envs` if it rises above that.

The pelvis is welded. MJX 3.3 does not support one collision pair in Unitree's
mesh model, so contacts are disabled. Results are suitable only as a second
opinion for suspended arm motion—not walking, contact, grasp collision, or
automatic real-robot gain deployment.
