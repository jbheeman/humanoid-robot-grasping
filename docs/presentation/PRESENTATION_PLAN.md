# Final Presentation Plan

## Outcome

Deliver an editable, evidence-backed group presentation for the **UC Santa Cruz Summer AI Academy**:

- 11 spoken slides in no more than 15 minutes
- five reactive appendix slides for the optional two-minute Q&A
- minimalist black-and-white visual system
- only real-life imagery
- PowerPoint, Keynote, and PDF formats

The story is:

> Physical tracking made the timing bottleneck visible. VLA experiments clarified action semantics but did not pass promotion gates. A deterministic local-control path produced a much smaller per-call compute cost and now supports one bounded physical interception test.

## Main deck

1. **Real-Time Moving-Object Interception** — team/school placeholders and program.
2. **Moving targets turn latency into the central problem** — see, locate, predict, move, verify.
3. **Each result exposed the next bottleneck** — vision → physical IK → data → VLA → real-time pivot.
4. **The robot retains final authority** — D435I, YOLO, RGB–D, alpha–beta tracking, plane crossing, Pinocchio, checks, Ruckig, and the robot-local 250 Hz guard.
5. **A sim-to-real dataset made the VLA question testable** — 480 sim + 67 real = 547 episodes; 32,051 frames; 127 → 97 → 67 real-data curation.
6. **Action semantics mattered more than training longer** — 36 attempts, 32 checkpoint-producing, 0 authorized; 31.8% and 21.6% ADE improvements.
7. **Lower offline error still produced inconsistent motion** — baseline failure, 2.14× displacement overshoot, 8.8% saturation, not promoted.
8. **Local servoing changed the timing regime** — 230.3 ms global solve versus 0.080 ms local step; approximately 2,879× per-call gap, explicitly not end-to-end speedup. Leave this slide projected for the live demo.
9. **The live path remains bounded by deterministic safety** — acquire, preview, commit, hold, expire; reject stale, discontinuous, collision, table, heartbeat, or following-error cases.
10. **The remaining question is physical, not computational** — demonstrated, offline, and pending evidence separated.
11. **Evidence—not optimism—determined the final system** — conclusion and questions.

## Timing

- Slides 1–7: about 8:15
- Slide 8 explanation: about 1:10
- Live demo while slide 8 remains projected: 1:15–1:30
- Slides 9–11: about 3:00
- Planned total: approximately 13:40–14:15

## Appendix

12. Technology map  
13. All 36 VLA attempts grouped by purpose  
14. Canonical IK benchmark table  
15. Research scale and secondary evidence  
16. Evidence labels and anticipated questions

## Claim rules

- Say **interception** or **controlled palm contact**, not verified grasp.
- Say **0 authorized**, not 0 successful.
- The 2,879× figure compares different call contracts; it is not an end-to-end robot speedup.
- The VLA results are offline; no learned Cartesian proposal was authorized for robot execution.
- Real XR frames are demonstration data, not autonomous-success evidence.
- No standardized physical success rate has been measured.

## Final preparation

- Replace the four presenter/school placeholders.
- Assign slide owners, demo operator, spotter, and e-stop responsibility.
- Rehearse the exact live-demo cutoff and recorded fallback.
- Run once with a visible timer; cut words if the main deck exceeds 14:15.
