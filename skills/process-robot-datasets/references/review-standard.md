# Exhaustive review standard

Review each source segment from start to end in every required view. A contact sheet or automatic
score is navigation evidence, not a final decision.

Mark PASS only when one or more complete, uninterrupted task attempts can be bounded. Store each
attempt as a separate episode under the same source segment. Exclude setup, human intervention,
reset activity, and trailing cleanup from every episode.

Mark FAIL when no complete attempt exists, the manipulated object is lost, human intervention
overlaps the task, required views are unavailable, or semantic boundaries cannot be established.
Write a concrete failure reason.

For each PASS episode:

- keep start and end inside the source segment;
- order non-overlapping episodes chronologically;
- make atomic boundaries cover the complete episode with no gaps;
- satisfy the task spec's minimum frame count for every atomic action;
- retain the original source frame coordinates in the decision document.

When the head camera observes a QR marker before the manipulation, preserve that continuous prefix
with `context_start_frame`. Keep `episode_start_frame` at the true manipulation start. The context
must remain in the exported episode with atomic action index `-1`; it is calibration evidence and
must not be relabeled as approach or acquisition. Record the measured QR black-square edge length
outside decisions and pass it explicitly to the calibration command.

The reviewer field must identify the human or visual-audit agent. Do not use `HUMAN`, empty names,
or unresolved placeholders in a production decision file.

For cup pick-and-place, confirm stable acquisition, transport with the cup, entry into the drop
zone, release with the cup remaining in the zone, and gripper retreat. A hand may set up the cup
before the robot begins or remove it after completion only when that activity is outside the saved
episode.
Confirm left- and right-hand subtask semantics separately; a stationary, supporting, or carrying
hand remains a distinct subtask stream even when both hands share the same phase boundary.

For three-cup stacking, begin with one stable nested stack, confirm a stable pyramid with two cups
on the bottom and one on top, then confirm that all three cups return to one stable nested stack.
Use the first approach to the nested stack as the episode start, the renewed approach to the stable
pyramid as the build/collapse boundary, and the completed gripper retreat as the exclusive end.
Exclude camera handling and human setup or cleanup outside those bounds.
