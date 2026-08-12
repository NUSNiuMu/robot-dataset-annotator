# Task extension

Add a task spec containing a stable task ID, ordered atomic actions, language instructions, minimum
model context, and a plugin import path. Implement the plugin under `tasks/<task>/`; accept
normalized observations and return either a candidate with boundaries and evidence or an explicit
unresolved reason.

Do not import ROS, a capture dashboard, or a training library from task code. Put raw-data parsing
in `adapters/` and final dataset writing in a separate export adapter.

Build a labeled regression set containing normal successes, incomplete attempts, human
intervention, sensor gaps, short actions, boundary ambiguity, and multiple complete attempts in one
source segment. Measure candidate precision, candidate recall, boundary error, runtime, and scratch
disk usage before promoting the plugin to production.

Keep exhaustive visual review until the user defines and approves a different acceptance standard.
