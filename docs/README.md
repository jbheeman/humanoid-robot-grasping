---
title: Find the project documentation
contentType: Landing
---

# Find the project documentation

Use this index to choose between presentation material, current operator guidance, research records, and historical plans. Some older pages describe architectures that the project later replaced.

## Present the project

Start with the [presentation documentation](presentation/README.md). It contains:

- [The complete project journey](presentation/PROJECT_JOURNEY.md)
- [The implementation history across every branch](presentation/IMPLEMENTATION_HISTORY.md)
- [How the complete technical system works](presentation/TECHNICAL_EXPLAINER.md)
- [The VLA experiment history](presentation/VLA_EXPERIMENT_HISTORY.md)
- [Experiments and quantitative results](presentation/EXPERIMENTS_AND_RESULTS.md)
- [Current status flow and final test](presentation/CURRENT_STATUS.md)
- [A 15-slide presentation outline](presentation/SLIDE_OUTLINE.md)
- [The evidence behind each claim](presentation/EVIDENCE_INDEX.md)

## Operate the current system

These pages describe the current or recently validated robot workflow:

- [Runtime architecture](ARCHITECTURE.md)
- [Project commands](COMMANDS.md)
- [Lab launchers](LAB_LAUNCHERS.md)
- [Arm commissioning runbook](ARM_COMMISSIONING_RUNBOOK.md)
- [Arm tracking runbook](ARM_TRACKING_RUNBOOK.md)
- [Manual ROS 2 arm control](MANUAL_ARM_ROS2.md)
- [Disarmed-controller diagnosis](FIX_DISARM.md)
- [Joint tuning guide](TUNING_GUIDE.md)
- [Third-party sources](THIRD_PARTY_SOURCES.md)

The source launchers are authoritative when a transport detail conflicts with prose. Current tracking uses the split CycloneDDS path. Commissioning and some isolated depth/legacy modes use Fast DDS.

## Review active research

These pages record current experiment contracts or results:

- [G1 IK backend evaluation](G1_IK_BACKEND_EVALUATION.md)
- [Lightweight bunny interception](LIGHTWEIGHT_BUNNY_INTERCEPT.md)
- [Morning interception runbook](REALTIME_INTERCEPT_MORNING_RUNBOOK.md)
- [Offline future-position evaluation](OFFLINE_BUNNY_PREDICTION.md)
- [Moving-rabbit VLA pipeline](VLA_MOVING_RABBIT.md)
- [VLA real-data and evaluation runbook](VLA_REAL_DATA_AND_EVAL_RUNBOOK.md)
- [VLA timing-alignment results](VLA_TIMING_ALIGNMENT_RESULTS.md)
- [Dynamic VLA recovery](VLA_DYNAMIC_RECOVERY_IMPLEMENTATION.md)
- [Adaptive UniFoLM campaign](vla_adaptive_campaign.md)
- [Plushie dataset sources](PLUSHIE_DATASETS.md)

## Read historical design records

These pages explain earlier decisions but should not be treated as the current launch procedure:

- [ROS 2 rewrite status](PLAN.md)
- [Tabletop pointing roadmap](ROADMAP.md)
- [Real-time interception implementation plan](REALTIME_INTERCEPT_TONIGHT_PLAN.md)
- [YOLO FastAPI implementation plan](YOLO_FASTAPI_STREAM_PLAN.md)
- [Superseded implementation review](review.md)
- [Early tracking start notes](TRACKING_START.md)
- [Unitree SDK2 investigation notes](UNITREE_SDK2_NOTES.md)
