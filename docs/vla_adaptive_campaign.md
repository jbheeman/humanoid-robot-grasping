# Adaptive UniFoLM campaign

The campaign controller trains one attempt at a time, evaluates on validation
data, records metrics in SQLite/WAL, and continues only while the frozen
baseline-normalized objective improves. The sealed test is opened only after a
candidate passes validation; any test result permanently terminates the
campaign.

The unattended trainer defaults to GPU 1 because GPU 0 has repeatedly fallen
off the PCIe bus. It varies learning rate, warmup, real/synthetic sampling
weight, augmentation, seed, and fresh versus action-head warm starts. The
default safety fuse is 24 attempts with a patience of three comparable
non-improving attempts. Infrastructure failures pause the campaign and do not
consume tuning patience.

Use the wrapper on the trainer:

```bash
cd /home/aarav/Documents/g1-bunny-vla-workspace
export INITIAL_ACTION_CHECKPOINT=/path/to/steps_N_action_model.pt
scripts/training/g1_vla_campaign.sh start
scripts/training/g1_vla_campaign.sh status
scripts/training/g1_vla_campaign.sh logs
scripts/training/g1_vla_campaign.sh stop-after-attempt
scripts/training/g1_vla_campaign.sh stop-now
scripts/training/g1_vla_campaign.sh resume
```

`stop-after-attempt` is lossless. `stop-now` interrupts training and resumes
only from a full state that the controller verified before the interrupt, so
up to one checkpoint interval can be lost. It never treats a partially written
state as resumable.

The authoritative ledger is
`runs/automation/<campaign>/campaign.sqlite3`. Deterministic CSV/JSONL exports
and PNG/SVG progress graphs are written beside it. Historical reports are
imported into separate comparable groups; the plots do not imply that metrics
from different action or evaluation contracts are directly comparable.

Completed-attempt retention keeps all portable action checkpoints and two full
states for the best attempt, one full state plus the selected action checkpoint
for the next two, and only the selected action checkpoint plus diagnostics for
the rest. Removals first enter a 24-hour quarantine under `runs/.trash`.
