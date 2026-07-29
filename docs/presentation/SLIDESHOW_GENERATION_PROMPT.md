# Reusable Prompt: Generate the Final Slideshow

Copy the prompt below into a presentation-capable ChatGPT session.

---

Create a polished, fully editable 16:9 `.pptx` for a **15-minute group research presentation plus an optional two-minute Q&A**.

## Identity

- Title: **Real-Time Moving-Object Interception**
- Subtitle: **with a Unitree G1**
- Program: **UC Santa Cruz Summer AI Academy**
- Preserve four editable identity lines:
  - `[Presenter 1 · School]`
  - `[Presenter 2 · School]`
  - `[Presenter 3 · School]`
  - `[Presenter 4 · School]`
- Do not include mentor names.

## Audience and voice

The audience is intelligent but not necessarily specialized in robotics. Use formal, neutral, plain language. Explain the physical problem before technical components. Avoid dense equations and long paragraphs.

## Visual direction

Use a minimal research-conference style:

- off-white background, black type, two neutral grays
- Avenir Next for prose and DIN Condensed for numbers/status
- square corners, thin rules, direct-labeled charts, generous whitespace
- no gradients, shadows, glowing HUD effects, generic stock imagery, decorative icons, or rounded-card grids
- only real project photographs or real demonstration frames in the visible deck
- label dataset frames **REAL XR DEMONSTRATION DATA**
- state that these frames are demonstrations, not autonomous execution
- use editable PowerPoint shapes for diagrams and charts

## Exact evidence

The project built a perception-guided Unitree G1 system for controlled palm contact with a moving plush bunny.

Live stack:

- Intel RealSense D435I RGB and depth
- YOLO bunny detection
- RGB-depth fusion to a tabletop 3D point
- alpha-beta position and velocity tracking
- plane-crossing interception prediction
- Pinocchio inverse kinematics
- joint, collision, and tabletop checks
- Ruckig jerk-limited motion shaping
- ROS 2 / DDS transport
- robot-local 250 Hz safety controller as final command authority

Representative perception dry run:

- about 57 camera FPS
- 49–51 YOLO FPS

Dataset:

- 547 episodes
- 480 Isaac Sim + 67 real XR-teleoperation episodes
- 32,051 frames
- real-data curation: 127 audited → 97 accepted → 67 canonical
- split by capture session

VLA experiments:

- 36 training or smoke attempts
- 32 produced checkpoints
- 0 authorized for robot execution
- weighted right-hand ADE, lower is better:
  - absolute action: 0.12517 m
  - relative to current pose: 0.08535 m, a 31.8% improvement
  - achieved future state: 0.06695 m, a further 21.6% improvement
- longer training did not beat the original pilot
- best timing-aligned model still failed deployment gates:
  - did not beat hold-pose or mean-action baselines
  - 2.14× predicted displacement versus target
  - 8.8% normalized-output saturation
- no learned Cartesian proposal was authorized for robot execution

Canonical offline IK benchmark:

- bounded SciPy global IK: 64/64 successes, 1.13 mm mean position error, 230.3 ms median
- MuJoCo DLS reference: 0/64 convergence, 9.53 mm, 15.2 ms, no unsafe output
- finite-difference local DLS: 64/64 accepted, 0.30 mm error per step, 0.136 ms median
- analytic Pinocchio local DLS: 64/64 accepted, 0.30 mm error per step, 0.080 ms median
- 230.3 / 0.080 = 2,878.75, approximately 2,879×
- this is a **per-call gap between different contracts**, not end-to-end robot speedup
- 0.136 → 0.080 ms is 41% less local Jacobian compute

Secondary appendix evidence:

- 9,312 simulation controller candidates
- 2.38% failure rate in the selected simulation campaign
- 22 attempts before the v29 adaptive campaign paused
- 77.6% active-motion windows in real training data
- 754 GB research store
- 25,756 lines removed from the focused demo branch across 183 files

Evidence boundaries:

- demonstrated: live perception, guarded arm movement, physical bunny tracking
- offline/artifact-verified: dataset, VLA ablations, IK benchmark
- pending: repeatable fast tracking, fault-response validation, bounded interception outcome
- no standardized physical success rate
- no verified fingered grasp

## Required 11-slide main deck

1. **Real-Time Moving-Object Interception**  
   Research-conference title, program, identity placeholders, one real project image. Status: tracking demonstrated; interception attempt pending.

2. **Moving targets turn latency into the central problem**  
   Timeline: see → locate → predict → move → verify. Define controlled palm contact and show the representative perception FPS.

3. **Each result exposed the next bottleneck**  
   Journey: vision → physical IK → simulation/data → VLA → real-time pivot. Takeaway: the project used negative results to choose the shortest defensible path.

4. **The robot retains final authority**  
   Draw the full live pipeline and show the Unitree G1 robot-local 250 Hz guard as the only command publisher. Put Isaac Sim, MuJoCo/MJX, RLDS/TFDS, and UniFoLM-VLA in a dashed offline-research band.

5. **A sim-to-real dataset made the VLA question testable**  
   Use three real XR frames as an approach sequence. Show 480 + 67 = 547, the 127 → 97 → 67 funnel, and 32,051 frames.

6. **Action semantics mattered more than training longer**  
   Direct-labeled zero-based bar chart for the three ADE values. Show 36 → 32 → 0 and highlight the 31.8% and 21.6% improvements.

7. **Lower offline error still produced inconsistent motion**  
   Large zero authorized count. Show baseline failure, 2.14× displacement, and 8.8% saturation. End with **NOT PROMOTED**.

8. **Local servoing changed the timing regime**  
   Dark climax slide. Compare 230.3 ms global solve with 0.080 ms analytic local step and show approximately 2,879× per-call gap. State clearly that the contracts differ and this is not end-to-end speedup. Also show 0.136 → 0.080 ms and 41% less local Jacobian compute. Leave this slide projected for the live demo. Add the cue: **guarded tracking + one bounded interception attempt**.

9. **The live path remains bounded by deterministic safety**  
   State path: acquire → preview → commit → hold → expire. List rejection conditions: stale RGB/depth, missing arm state, joint/discontinuity limits, collision/swept path, tabletop clearance, heartbeat/following error.

10. **The remaining question is physical, not computational**  
    Three columns: demonstrated, offline evidence, pending. Explicitly say no verified fingered grasp, no standardized physical success rate, and no learned policy authorized.

11. **Evidence—not optimism—determined the final system**  
    Summarize: physical tracking established feasibility; VLA experiments corrected assumptions; local control created a practical final hypothesis. Close with: **Can it follow fast enough—with evidence and safety?**

## Q&A appendix

12. Technology map  
13. The 36 VLA attempts grouped by purpose  
14. Canonical IK benchmark table  
15. Research scale and secondary evidence  
16. Evidence labels and anticipated questions

## Timing and notes

- Plan the main deck for 13:40–14:15 including a 75–90 second live demo.
- Put concise speaker notes on every slide.
- Include presenter handoff cues.
- Keep slide 8 projected during the demo; do not create a separate demo slide.
- Add the benchmark caveat in both visible text and speaker notes.
- Generate an editable `.pptx`, not a collection of images.

Do not invent citations, names, affiliations, success rates, hardware capabilities, or autonomous results.
