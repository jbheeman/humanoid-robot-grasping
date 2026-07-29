# Presentation Design: Monochrome Robotics Evidence

## Direction

The deck is a research-conference presentation, not a product pitch. It uses an off-white field, black typography, thin rules, square geometry, and real laboratory imagery. Evidence creates the visual interest.

Avoid gradients, shadows, glowing “robot HUD” effects, decorative icons, rounded-card grids, and generic stock imagery.

## Canvas and grid

- 16:9 widescreen, 13.333 × 7.5 inches
- 0.65–0.82 inch outer margins
- consistent left title edge
- slide number at bottom right; evidence qualifier at bottom left
- one dominant visual thesis per slide

## Color

- Paper: `#F4F3EF`
- Ink: `#090909`
- Mid gray: `#6C6B67`
- Light gray: `#D9D8D2`
- White: `#FFFFFF`

Status never depends on color alone:

- **DEMONSTRATED:** solid black
- **OFFLINE / ARTIFACT-VERIFIED:** solid outline
- **PENDING:** dashed outline
- **NOT PROMOTED:** black stop band

## Typography

- Primary family: **Avenir Next**
- Numeric/status family: **DIN Condensed**
- Hero: 50 pt
- Slide title: 36 pt
- Takeaway: 30 pt
- Body: 21 pt
- Diagram text: 17 pt
- Data labels: 14 pt
- Metadata/footer: 11.5 pt

Use sentence case except short evidence labels. Keep numbers on one line. Never place a paragraph on a slide when a direct label or one-sentence takeaway will do.

## Imagery

- Use only real project photographs or real demonstration frames in the visible deck.
- Current frames come from the real XR teleoperation dataset and are labeled **REAL XR DEMONSTRATION DATA**.
- These frames illustrate the task and data collection; they do not imply autonomous execution.
- Do not show Isaac Sim screenshots in the main deck.

## Charts and diagrams

- Direct-label every mark; avoid legends.
- Start bars at zero when length encodes magnitude.
- Use black for the focal result and gray for comparison.
- Put the evidence class next to the result: representative dry run, offline evaluation, or offline canonical benchmark.
- Never convert a per-call compute ratio into an end-to-end robot speedup.

## Motion and live demo

- The deck must work without animation.
- Leave slide 8 projected during the in-person demonstration.
- Demo scope: guarded tracking plus one bounded interception attempt.
- Keep a recorded fallback available, but do not embed simulated footage.

## Accessibility

- Maintain at least 4.5:1 contrast.
- Body copy is at least 18 pt.
- Define uncommon acronyms in speech.
- Use line style, fill, and wording—not color alone—to communicate status.
