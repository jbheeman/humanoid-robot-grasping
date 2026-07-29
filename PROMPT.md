# Prompt for ChatGPT: Create the Final Research Presentation

You are an expert research storyteller, presentation designer, and PowerPoint author. Create a polished, fully editable **16:9 PowerPoint presentation (`.pptx`)** for a group presentation at the **UC Santa Cruz Summer AI Academy**.

The presentation must fit within:

- **15 minutes total**, including the live demonstration
- **Optional two-minute Q&A**
- **11 main slides**
- **Five appendix slides** that are opened only when relevant during Q&A

Do not merely write an outline. Produce the actual downloadable `.pptx`, with editable text, charts, diagrams, speaker notes, and image placements. Also export a PDF backup if the environment supports it.

## Source restrictions

Use only information found in:

- `docs/`
- `docs/presentation/`

Do not use source code, filenames, branch names, commit history, or external claims as project evidence unless the same information is explicitly documented in those two locations.

Before creating the deck, inspect the relevant documents and reconcile duplicate figures. Prefer final or canonical results over exploratory values. Never invent names, affiliations, citations, measurements, success rates, or robot capabilities.

## Identity

Title:

**Real-Time Moving-Object Interception**

Subtitle:

**with a Unitree G1**

Program:

**UC Santa Cruz Summer AI Academy**

Preserve these four editable lines:

- `[Presenter 1 · School]`
- `[Presenter 2 · School]`
- `[Presenter 3 · School]`
- `[Presenter 4 · School]`

Do not include mentor names. Do not infer presenter identities from project files.

## Audience and writing style

The audience may include students, instructors, researchers, and visitors who do not specialize in robotics.

Use:

- clear, formal, neutral language
- short declarative headlines that communicate the slide’s conclusion
- plain-language explanations before technical terminology
- concise direct labels instead of paragraphs
- speaker notes for detail that does not belong visibly on the slide

Avoid:

- dense equations
- unexplained acronyms
- long bullet lists
- marketing language
- exaggerated novelty or success claims
- vague titles such as “Methodology,” “Results,” or “Our Journey”

Each slide must communicate one main idea and should be understandable within several seconds.

## Visual system

Create a minimalist black-and-white research-conference deck.

### Color

- warm off-white background: approximately `#F4F3EF`
- nearly black text: approximately `#090909`
- medium neutral gray: approximately `#6C6B67`
- light neutral gray: approximately `#D9D8D2`
- white where needed for contrast

Do not use accent colors, gradients, glows, shadows, glass effects, or a futuristic robot-HUD aesthetic.

### Typography

Use:

- **Avenir Next** for headlines, body text, and diagrams
- **DIN Condensed** for large numbers and short status labels

Suggested hierarchy:

- hero title: 48–52 pt
- slide title: 34–38 pt
- major takeaway: 28–32 pt
- body text: 20–22 pt
- diagram labels: 16–18 pt
- data/status labels: 13–15 pt
- metadata and footers: 10–12 pt

Keep important numbers on one line. Use sentence case except for short evidence labels. Maintain strong contrast and generous whitespace.

### Composition

- 16:9 widescreen
- consistent left title edge
- square corners
- thin rules and precise alignment
- direct-labeled charts rather than legends
- asymmetry and scale contrast instead of decoration
- one dominant visual thesis per slide
- small slide number at bottom right
- evidence qualifier at bottom left when needed

Do not create repetitive grids of rounded cards.

## Image policy

Use only **real-life project imagery** in the visible presentation.

Prioritize the real XR demonstration frames under:

`docs/presentation/assets/real/`

The current monochrome presentation-ready files include:

- `real_xr_demo_060_mono.jpg`
- `real_xr_demo_105_mono.jpg`
- `real_xr_demo_150_mono.jpg`

Use them as:

- one strong title-slide image
- a three-frame approach sequence on the dataset slide

Label these images:

**REAL XR DEMONSTRATION DATA**

Include a small qualifier where appropriate:

**Frames show demonstration data, not autonomous execution.**

Do not use Isaac Sim screenshots, generated robot imagery, generic stock robots, or unrelated web images.

## Central narrative

The presentation should tell one causal story:

> Physical tracking made the timing bottleneck visible. The VLA experiments clarified action semantics but did not pass deployment gates. Those negative results led to a deterministic local-control path with a much smaller per-call compute cost and a final safety-bounded physical interception test.

This is not a generic “AI versus traditional robotics” story. It is a story about using evidence to decide which approach was ready for the robot.

## Exact technical facts

The team developed a perception-guided Unitree G1 system for controlled palm contact with a moving plush bunny.

### Live perception and control stack

- Intel RealSense D435I RGB and depth
- YOLO bunny detection
- RGB-depth fusion to recover a tabletop 3D point
- alpha-beta position and velocity tracking
- plane-crossing interception prediction
- Pinocchio inverse kinematics
- joint, self-collision, swept-path, and tabletop checks
- Ruckig jerk-limited trajectory shaping
- ROS 2 / DDS transport
- robot-local **250 Hz** safety controller as the final command authority

Only the robot-side guarded process publishes Unitree arm commands. No learned Cartesian proposal directly controls the motors.

### Representative perception dry run

- approximately **57 camera FPS**
- **49–51 YOLO FPS**

These values describe representative perception throughput. They do not establish successful interception.

### Dataset

- **547 episodes**
- **480 Isaac Sim episodes**
- **67 real XR-teleoperation episodes**
- **32,051 frames**
- real-data curation: **127 audited → 97 accepted → 67 canonical**
- data splits separated by capture session

### VLA experiments

- **36** training or smoke attempts
- **32** checkpoint-producing attempts
- **0** policies authorized for robot execution

Weighted right-hand average displacement error, where lower is better:

- absolute action: **0.12517 m**
- relative to current pose: **0.08535 m**
- absolute-to-relative improvement: **31.8%**
- achieved future state: **0.06695 m**
- relative-to-future-aligned improvement: **21.6%**

Longer training did not improve the original pilot. The most important improvement came from changing the action definition so the target matched motion achieved after sensing, inference, transport, and actuation delay.

The best timing-aligned VLA still failed promotion:

- it did not beat required hold-pose or mean-action baselines
- predicted displacement was **2.14×** the target displacement on real validation
- approximately **8.8%** of normalized outputs saturated
- robot-execution authorization remained **No**

Do not describe “zero authorized” as “zero successful.” The system produced useful offline results, but it did not satisfy the consistency and safety requirements for hardware execution.

### Canonical inverse-kinematics benchmark

| Backend | Success / acceptance | Mean position error | Median latency | Unsafe |
| --- | ---: | ---: | ---: | ---: |
| Bounded SciPy global IK | 64 / 64 | 1.13 mm | 230.3 ms | 0 |
| MuJoCo DLS reference | 0 / 64 | 9.53 mm | 15.2 ms | 0 |
| Finite-difference local DLS | 64 / 64 accepted | 0.30 mm per step | 0.136 ms | 0 |
| Analytic Pinocchio local DLS | 64 / 64 accepted | 0.30 mm per step | 0.080 ms | 0 |

Important interpretation:

- `230.3 ÷ 0.080 = 2,878.75`, or approximately **2,879×**
- this is a **per-call latency gap between different contracts**
- the global solver searches for a complete pose
- the local solver performs one bounded correction from measured state
- this is **not** an end-to-end claim that the robot became 2,879 times faster
- replacing the finite-difference Jacobian with the analytic Pinocchio Jacobian reduced local median compute from **0.136 ms to 0.080 ms**
- that is approximately **41% less local Jacobian compute**

Repeat the benchmark caveat in both visible text and speaker notes.

### Secondary evidence for the appendix

- **9,312** controller candidates evaluated in six hours of simulation
- **2.38%** failure rate in the selected simulation campaign
- **22** attempts before the v29 adaptive campaign paused for no progress
- **77.6%** active-motion windows in real training data
- **754 GB** research store
- **25,756** lines removed from the focused demo branch across **183 files**

These figures provide research context. They are not an end-to-end physical score.

## Evidence grammar

Use visible labels that distinguish:

- **DEMONSTRATED** — observed on real physical hardware
- **OFFLINE / ARTIFACT-VERIFIED** — recorded in a manifest, report, dataset, or canonical benchmark
- **PENDING** — implementation exists, but physical validation remains
- **NOT PROMOTED** — failed the required deployment gates

Never combine results from different evidence classes into one implied success metric.

Current evidence boundaries:

### Demonstrated

- live perception
- guarded arm movement
- physical bunny tracking

### Offline or artifact-verified

- 547-episode dataset
- VLA ablations
- analytic IK benchmark

### Pending

- repeatable fast tracking
- fault-response validation
- bounded interception outcome

State explicitly:

- no verified fingered grasp
- no standardized physical tracking success rate
- no learned policy authorized for robot execution

Use **interception** or **controlled palm contact**, not “verified grasp.”

## Required main deck

### Slide 1 — Real-Time Moving-Object Interception

Use a research-conference title composition with one large real XR demonstration image.

Include:

- title and subtitle
- four presenter/school placeholders
- UC Santa Cruz Summer AI Academy
- short thesis: **From VLA experimentation to safety-bounded geometric control**
- compact status labels:
  - **TRACKING DEMONSTRATED**
  - **INTERCEPTION ATTEMPT PENDING**

Do not begin with a generic agenda.

Target speaking time: **0:40**

### Slide 2 — Moving targets turn latency into the central problem

Show a target moving across the complete loop:

**SEE → LOCATE → PREDICT → MOVE → VERIFY**

Explain:

- the target continues moving while the system senses and computes
- the goal is controlled palm contact
- motion must remain within joint, collision, tabletop, freshness, and operator limits

Show the representative perception numbers:

- approximately 57 camera FPS
- 49–51 YOLO FPS

Add the qualifier that perception speed alone does not establish interception.

Target speaking time: **0:55**

### Slide 3 — Each result exposed the next bottleneck

Create a horizontal research journey:

1. **VISION** — approximately 60-FPS-class camera and YOLO
2. **PHYSICAL IK** — guarded tracking demonstrated
3. **SIM + DATA** — 9,312 controller trials and 547 episodes
4. **VLA** — 36 attempts and 0 authorized
5. **REAL-TIME PIVOT** — local analytic IK and Ruckig

End with the takeaway:

**The project did not abandon learning. It used negative results to choose the shortest defensible live path.**

Target speaking time: **1:00**

### Slide 4 — The robot retains final authority

Draw the live pipeline:

**D435I → YOLO → RGB–D → alpha-beta tracking → crossing prediction → Pinocchio → checks → Ruckig → Unitree G1**

Give each component a short plain-language function.

Make the final Unitree G1 block large and black:

- **250 Hz**
- robot-local safety controller
- only this process publishes Unitree arm commands

Place the following inside a dashed **OFFLINE RESEARCH** band:

- Dual RTX 3090
- Isaac Sim
- MuJoCo/MJX
- RLDS/TFDS
- UniFoLM-VLA evaluation

Target speaking time: **1:20**

### Slide 5 — A sim-to-real dataset made the VLA question testable

Use the three real XR demonstration frames as an approach sequence.

Show:

**480 simulation + 67 real = 547 episodes**

Also show:

- **127 audited → 97 accepted → 67 canonical real**
- **32,051 frames**
- session-held-out train, validation, and sealed-test splits

Clearly label the imagery as demonstration data, not autonomous execution.

Target speaking time: **1:15**

### Slide 6 — Action semantics mattered more than training longer

Start with:

**36 attempts → 32 checkpoint-producing → 0 authorized**

Create a direct-labeled, zero-based horizontal bar chart:

- absolute action — 0.12517 m
- relative to current pose — 0.08535 m
- achieved future state — 0.06695 m

Highlight:

- **31.8%** improvement from absolute to relative
- **21.6%** further improvement from relative to future-aligned

Visible conclusion:

**Longer training did not improve the original pilot. The label definition was the breakthrough.**

Target speaking time: **1:20**

### Slide 7 — Lower offline error still produced inconsistent motion

Use a large:

**0 robot-authorized checkpoints**

Show three promotion failures:

- **BASELINE** — did not beat holding the current pose or predicting the mean action
- **2.14×** — predicted displacement exceeded the real-validation target motion
- **8.8%** — normalized outputs saturated at the clipping limit

End with a black **NOT PROMOTED** bar.

Target speaking time: **1:00**

### Slide 8 — Local servoing changed the timing regime

Make this the visual climax. Use a black background and white typography.

Compare:

- **230.3 ms** — bounded global solve
- **0.080 ms** — bounded analytic local step
- **approximately 2,879×** — per-call latency gap

Show the caveat prominently:

**Different contracts. This is not an end-to-end robot speedup.**

Also show:

- **0.136 → 0.080 ms**
- **41% less local Jacobian compute**

At the bottom, include the live-demonstration cue:

**LIVE DEMONSTRATION FOLLOWS**

**guarded tracking + one bounded interception attempt**

Leave this slide projected during the live demo. Do not create a separate demo slide.

Target explanation time: **1:10**  
Target demonstration time: **1:15–1:30**

### Slide 9 — The live path remains bounded by deterministic safety

Show:

**ACQUIRE → PREVIEW → COMMIT → HOLD → EXPIRE**

Explain that a target is rejected if any boundary fails:

- fresh RGB and paired depth
- measured arm state
- joint-step and discontinuity limits
- self-collision and swept-path checks
- table-plane clearance
- heartbeat and following-error limits

Show the robot-local **250 Hz** guard again as the final release authority.

Target speaking time: **1:00**

### Slide 10 — The remaining question is physical, not computational

Create three visibly distinct evidence columns.

**DEMONSTRATED**

- live perception
- guarded arm movement
- physical bunny tracking

**OFFLINE EVIDENCE**

- 547-episode data contract
- VLA ablations
- analytic IK benchmark

**PENDING**

- repeatable fast tracking
- fault-response validation
- bounded interception outcome

Below the columns, state:

- no verified fingered grasp
- no standardized physical success rate
- no learned policy authorized

Conclude:

**The final hypothesis: faster local control can preserve tracking quality while reducing enough lag for interception.**

Target speaking time: **1:00**

### Slide 11 — Evidence—not optimism—determined the final system

Use a strong, minimal conclusion slide.

Summarize:

- **PHYSICAL TRACKING** established that the guarded loop was feasible
- **VLA EXPERIMENTS** corrected assumptions about actions, timing, and consistency
- **LOCAL CONTROL** created a practical final real-time hypothesis

Close with:

**Can it follow fast enough—with evidence and safety?**

Include a small **Questions** cue and the UC Santa Cruz Summer AI Academy name.

Target speaking time: **0:40**

## Required Q&A appendix

### Slide 12 — Appendix · Technology map

Map each technology to its purpose:

- RealSense D435I
- YOLO
- RGB-depth fusion
- alpha-beta filter
- plane crossing
- Pinocchio
- Ruckig
- ROS 2 / DDS
- Isaac Sim
- MuJoCo / MJX
- RLDS / TFDS
- UniFoLM-VLA

### Slide 13 — Appendix · The 36 VLA attempts

Group all 36 attempts by purpose:

- initial full training
- timing-aligned targets
- v28 staged schedule
- loss-weight controls
- dynamic-action pilots
- motion-focused data
- Pose23 representation
- v29 automated campaign
- longer relative training
- v30 active-horizon attempt

State that 32 attempts produced checkpoints and 0 were authorized for robot execution.

### Slide 14 — Appendix · Canonical IK benchmark

Show the exact four-row benchmark table supplied earlier.

Include:

**230.3 ÷ 0.080 = 2,878.75**

And:

**Global solves and local servo steps have different contracts.**

### Slide 15 — Appendix · Research scale and secondary evidence

Show:

- 9,312 controller candidates
- 2.38% selected-campaign failure rate
- 22 v29 attempts
- 77.6% active-motion windows
- 754 GB research store
- 25,756 lines removed across 183 files

Clearly state that these values are secondary context, not one end-to-end score.

### Slide 16 — Appendix · Evidence map and anticipated questions

Define:

- demonstrated
- artifact-verified
- source-verified
- pending

Prepare short answers for:

- Why not deploy the best VLA?
- Was inverse kinematics really 2,879× faster?
- Why use simulation?
- Why interception rather than grasping?
- What remains?

## Live-demo handling

The presentation includes an in-person demonstration.

The intended scope is:

**guarded tracking plus one bounded interception attempt**

Do not imply a repeatable success rate from one attempt.

Leave slide 8 projected during the demonstration. In the speaker notes, include this sequence:

1. announce the exact scope
2. begin only after the operator and spotter confirm the approved setup
3. demonstrate acquisition and guarded tracking
4. perform one bounded interception attempt
5. stop on success, safety rejection, or the rehearsed cutoff
6. transition with:

> The point is not a perfect catch; it is a bounded physical test of whether reduced control latency preserves tracking quality.

Do not add operational robot instructions that are not present in the approved project runbook.

## Speaker notes

Add concise speaker notes to every slide.

Notes must include:

- target time
- plain-language explanation
- evidence caveats
- presenter handoff cue
- exact demo transition on slide 8

The complete main presentation, including the demonstration and handoffs, should rehearse at approximately **13:40–14:15**. Do not exceed 15 minutes.

## Final quality requirements

Before returning the files:

1. Render every slide and visually inspect it.
2. Verify that no text is clipped, overlapping, or too small.
3. Verify that every important number remains on one line.
4. Check that all charts start at zero when bar length encodes magnitude.
5. Confirm that all visible imagery is real project imagery.
6. Confirm that no slide implies autonomous execution from XR demonstration frames.
7. Confirm that the 2,879× figure is never described as end-to-end speedup.
8. Confirm that the VLA is described as **0 authorized**, not 0 successful.
9. Confirm that no claim of verified grasp or standardized physical success rate appears.
10. Confirm that presenter/school placeholders remain editable.
11. Confirm that every element in the `.pptx` remains editable whenever practical.
12. Return the `.pptx` and, if possible, a PDF backup.

The final result should look like a carefully edited robotics research presentation: restrained, clear, highly legible, evidence-led, and ready to present tomorrow.
