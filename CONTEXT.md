# CONTEXT — the vocabulary of the sprite pipeline

A glossary, and nothing else. No implementation detail lives here: how a thing works belongs
in `README.md`, why it came to be belongs in `DEVLOG.md`, and whether it is done belongs in
OpenProject. This file exists because several of the words below name *different* things that
sound like the same thing, and a reader who conflates them draws the wrong conclusion from
otherwise clear code.

## Camera words — three axes that are easy to confuse

**view** — a named camera angle on a *standing character*, drawn off the accepted base by
Qwen-Image-Edit. There are eight, plus the front. A view is a **reference image**: a statement
about what the character looks like from that side. It is not a frame of anything.

**front** — the accepted base itself. It is a view by name, and it is never rendered, because
it already exists.

**direction** (also **cell**, also a **compass cell**) — one of the 16 facings an *action* can
be rendered at: eight yaws × two elevations, named `s0`…`s315` and `t0`…`t315`. A direction is
where the camera stands for a piece of **motion**. The Animate tab works in directions; it
*consumes* views as identity references, one view per direction.

**pose** — a skeleton: OpenPose keypoints extracted per frame, used to drive motion. Nothing
to do with a camera angle. A "pose pass" is a skeleton render, not a view.

The one-line test: a **view** says *what the character looks like*; a **direction** says *where
the camera is for an action*; a **pose** says *how the body is arranged*.

**mirrorable** — a view, or a whole animation at one direction, derivable from its partner by a
horizontal flip, because a flip takes a camera at yaw *t* to the camera at yaw *180 − t*. Three
pairs: right↔left, back_right↔back_left, front_left↔front_right. `back`, `top` and `front` look
straight down the symmetry plane and have no partner. A mirror costs no GPU, and it **swaps
handedness**: a sword in the right hand comes back in the left. That is why mirroring is always
an explicit choice a person makes for that one view or that one animation, never something a
batch does on its own.

## core / main / advanced — three overlapping sets, deliberately not aligned

Three names group the same eight views, and they answer different questions. They are **not**
the same set and are not meant to be:

| Set | Members | Answers |
|---|---|---|
| **core** | back, left, right, top | what a bare `anim concept --views` renders |
| **main** | back, left, top | which views get a top-level button on the Views tab, and which count toward that tab's ✓ |
| **advanced** | right, back_right, back_left, front_left, front_right | which views sit behind the tab's fold |

`right` is in **core** *and* in **advanced**. That is not a bug: the CLI's default is "the four
useful angles", while the tab's split is "what a person needs at a glance" versus "what they can
open if they want it" — and the right profile is usually mirrored from the left rather than
drawn, so it does not earn a front-row button. Realigning the two would silently change what a
bare `--views` renders. `main` and `advanced` together cover all eight; `core` is a shorthand
that crosses the fold.

## Generation words

**attempt** — one rendered image on one target, kept forever. Asking for a target again *adds*
attempts; nothing is replaced. A target is the base, one view, or one frame of an animation.

**pick** — the one attempt a person has **accepted** for a target. A view's pick is what the
Animate tab reads for that view's direction; a base's pick is the front. Accepting is always a
human act.

**approved / rejected / to review** — a frame's **verdict**: the person's judgement of the
attempt the frame currently shows. A frame with a picture and no verdict on it is *to review*.
For a frame of an animation, approving is the same human act as a pick, under the name the
Animate tab uses; editing the approved picture takes the approval away.

**base gate** — the rule that no view may be drawn until the base has a pick. A view drawn from
an unaccepted base is a picture of a character nobody has agreed on.

**provisional** — a pick recorded for a rerun, held as pending rather than accepted.

**stale** — a recorded thing whose input has since changed: views drawn from an earlier base, or
a pick edited after it was accepted. Stale is reported, never silently corrected.

## Animation words

**animation** — one action at one direction for one unit: a row of **frames** played in order.
The unit of work on the Animate tab, and the thing a finished cell of the **grid** holds.

**frame** — one picture of an animation, at one place in the row. It has attempts, shows one of
them, and carries a verdict. Its number ("Frame 4") is its place in the row and changes when
frames are added or removed; its id (`f04`) does not.

**keyframe** — a frame marked as a key moment of the action, labelled with its **phase**:
start, wind-up, action, wind-down, end. A keyframe is a statement about *what the character is
doing at that instant*; it is drawn from the direction's view and approved by a person like any
other frame. Not to be confused with a video codec's keyframe.

**phase** — the label a keyframe carries naming which key moment it is (wind-up, action, …).
Every keyframe has one; an in-between has none of its own.

**in-between** — a frame that is not a keyframe: it lies between two keyframes and exists only to
carry the motion from one to the next. It is drawn from the approved keyframes either side of
it, and may not be drawn until both are approved. An animation with no in-betweens is still an
animation; one with no keyframes is not.

**draw from** — what a frame attempt is drawn from, besides its skeleton: the **view** (the
direction's accepted view), the **keys** (the approved keyframes either side of the frame), or
**both**. A fourth answer, **copy**, is not a drawing at all: the frame's picture *is* the
direction's accepted view, unchanged. Only a keyframe can be a copy.

**view key** — a keyframe whose picture is a **copy** of its direction's view, because at that
moment the character stands exactly as the view shows. By default an action's start keyframe is
a view key, and so is its end unless the action loops (a loop's end must flow back into its
start, not repeat it). A view key is still judged like any frame: it arrives *to review*, and it
goes **stale** if the view it copied is re-accepted.

## Batch words

**sweep** — one press that covers a whole set of targets in a single run. A sweep skips targets
that already have a pick, so pressing it twice tops up what is unfinished rather than piling
attempts onto settled work. On the Animate tab a sweep covers several **animations** at once —
whichever cells of the grid are ticked — drawing their keyframes, or their in-betweens once
every ticked cell's keys are approved, as one server run with one GPU lease. A cell the sweep
cannot draw (no accepted view; for in-betweens, a key not yet approved) is **blocked**, not
substituted, same as any other blocked target.

**grid** — the Animate tab's view of every **animation** a unit has or could have: one row per
studio action, one column per side direction, each cell showing that animation's state at a
glance. The grid is what a sweep is ticked against.

**blocked** — a cell or a press that *may not* run, because a prerequisite is missing. Distinct
from **failed**, which was attempted and did not work. A batch reports both separately, and a
blocked direction is never quietly substituted with a fallback.

**warm / cold** — whether the GPU already holds the model. A cold run pays a one-off load of
roughly three minutes on its first render; every render after that is warm. Any duration quoted
to a person says which of the two it means.
