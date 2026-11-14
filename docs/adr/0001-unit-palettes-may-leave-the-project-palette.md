# Unit palettes may leave the project palette

A unit palette was always a subset of the project palette, so every unit looked like one game.
The 2026-10-11 wanderer test showed the cost: snapping to the nearest project colour turned the
wanderer's hair maroon, and a subset cannot fix that because it still snaps to the nearest
project colour. We now allow a unit palette taken from the unit's own art
(`anim palette --unit X --from-art`) to hold colours the project palette lacks. Such a unit is
**off-palette**: the palette file says so and every post report names it, so cohesion becomes a
visible per-unit choice instead of a rule nobody can see being broken.

Considered: keeping the subset rule (colours stay wrong), and growing the project palette with
each unit's colours (the shared palette grows without limit and every unit inherits colours
meant for one).
