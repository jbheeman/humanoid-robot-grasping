---
title: Present the humanoid robot grasping project
contentType: Landing
---

# Present the humanoid robot grasping project

This folder turns the repository, robot trials, and Isaac Sim training artifacts into a presentation-ready account of the project. It distinguishes demonstrated behavior from offline results and work awaiting final validation.

## Recommended reading order

1. [Project journey](PROJECT_JOURNEY.md): the story from the first prototype to the current retest phase
2. [Implementation and branch history](IMPLEMENTATION_HISTORY.md): the commit-level chronology across every GitHub branch
3. [Complete system explainer](TECHNICAL_EXPLAINER.md): VLA, IK, perception, simulation, middleware, and safety concepts
4. [VLA experiment history](VLA_EXPERIMENT_HISTORY.md): all 36 train or smoke attempts and what each experiment group established
5. [Experiments and results](EXPERIMENTS_AND_RESULTS.md): dataset scale, metrics, and negative results
6. [Current status](CURRENT_STATUS.md): the project flowchart, what works, and what remains to test
7. [Slide outline](SLIDE_OUTLINE.md): a 15-slide structure with suggested visuals and speaker points
8. [Evidence index](EVIDENCE_INDEX.md): the source behind every important numerical claim

## One-paragraph project summary

The project developed a perception-guided control stack for a Unitree G1 humanoid to track and intercept a moving plush bunny. The team built a 60 FPS camera pipeline, YOLO detection, depth fusion, tabletop calibration, guarded ROS 2 arm control, collision-aware inverse kinematics (IK), research telemetry, Isaac Sim data generation, real demonstration curation, and UniFoLM-VLA training. The robot successfully moved its arm to track the bunny with IK, but the motion was too slow for reliable interception. A faster analytic IK servo, Ruckig trajectory generator, and real-time interception runtime are now isolated on the focused `demo-realtime-intercept-minimal` branch for the final physical retest.

## How to describe the outcome

Use these statements in a presentation:

- The robot demonstrated closed-loop visual tracking and physical arm movement with guarded IK
- The first working controller was accurate enough to follow the bunny, but too slow for the final moving-target task
- The team generated and curated 547 canonical sim-and-real episodes for the latest VLA dataset
- The VLA store records 36 train or smoke attempts, including 32 that produced action-model checkpoints
- Timing-aware VLA targets improved weighted right-hand trajectory error by 21.6% over the previous relative-action pilot
- No learned policy passed the project’s offline promotion gate, so no failing checkpoint was authorized to control the robot
- The current branch removes broad research tooling and focuses on the faster analytic IK, Ruckig, and interception stack

Avoid saying that the robot completed a grasp or that the VLA solved moving-object interception. The robot has no verified grasping hand in this setup, and the learned policy remains an offline research result.
