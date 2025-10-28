You are a classical STRIPS planner for the Rush Hour puzzle on a 6×6 grid.

Grid & naming:
- Positions are locR_C (row R, col C), with R,C ∈ {1..6}.
- A car occupies consecutive positions in a row (horizontal movement) per this encoding.
- The predicate ADJACENT(p1, p2, p3) encodes that p2 lies between p1 and p3 along a straight line (used to advance one cell).

Goal:
- Move the red car to the right edge: (containsCar loc5_3 red) and (containsCar loc6_3 red)

Objects:
- Cars: red, car1, car3, car4, car5, car6, car10, car13, car14, car16, car17, car18
- Positions: loc1_1 … loc6_6 (all 36 grid cells)

Initial state (key facts):
- Types:
  (SMALL car1) (SMALL car3) (SMALL car4) (SMALL car5) (SMALL car10) (SMALL car16) (SMALL car17) (SMALL car18) (SMALL red)
  (LARGE car13) (LARGE car14) (LARGE car6)
- Red car initially occupies: (containsCar loc1_3 red) (containsCar loc2_3 red)
- Other cars (examples; full list is provided to the evaluator and should be respected):
  (containsCar loc1_1 car1) (containsCar loc1_2 car1)
  (containsCar loc4_1 car13) (containsCar loc5_1 car13) (containsCar loc6_1 car13)
  (containsCar loc3_2 car14) (containsCar loc4_2 car14) (containsCar loc5_2 car14)
  (containsCar loc1_4 car16) (containsCar loc2_4 car16)
  (containsCar loc3_3 car3) (containsCar loc3_4 car3)
  (containsCar loc4_3 car4) (containsCar loc4_4 car4)
  (containsCar loc5_3 car5) (containsCar loc5_4 car5)
  (containsCar loc6_2 car6) (containsCar loc6_3 car6) (containsCar loc6_4 car6)
  (containsCar loc4_5 car10) (containsCar loc4_6 car10)
  (containsCar loc5_5 car17) (containsCar loc6_5 car17)
  (containsCar loc2_6 car18) (containsCar loc3_6 car18)
- Occupancy facts include, e.g.: (isOccupied loc1_1) (isOccupied loc1_2) … consistent with the placements above.
- ADJACENT facts define straight-line triples along rows and columns to support one-cell advances (available to the evaluator).

Actions you can use (schemas):
```

(move-small ?c ?p1 ?p2 ?p3)
(move-large ?c ?p1 ?p2 ?p3 ?p4)

```

Action semantics (PDDL-style, precise):
```

move-small():
PRE: and (SMALL ?c) (ADJACENT ?p1 ?p2 ?p3) (containsCar ?p1 ?c) (containsCar ?p2 ?c) (not (isOccupied ?p3))
EFF: and (not (isOccupied ?p1)) (not (containsCar ?p1 ?c)) (containsCar ?p3 ?c) (isOccupied ?p3)

move-large():
PRE: and (LARGE ?c) (ADJACENT ?p1 ?p2 ?p3) (ADJACENT ?p2 ?p3 ?p4)
(containsCar ?p1 ?c) (containsCar ?p2 ?c) (containsCar ?p3 ?c) (not (isOccupied ?p4))
EFF: and (not (isOccupied ?p1)) (not (containsCar ?p1 ?c)) (containsCar ?p4 ?c) (isOccupied ?p4)

```

Constraints & guidance:
- Only move a SMALL car with (move-small …) and a LARGE car with (move-large …).
- For a move, the car’s current occupied positions must match the parameters (p1,p2 for small; p1,p2,p3 for large) and the next position (p3 for small; p4 for large) must be unoccupied.
- Respect occupancy: after moving, the freed head position is no longer occupied and no longer contains the car; the new tail position becomes occupied and contains the car.
- Use only object names from this problem (cars and locR_C positions). Do not invent new names.
- Plan should be as short as possible while achieving the goal (avoid redundant moves).

Think privately to ensure every PRE holds before each step and the final state meets the goal. Do NOT print your reasoning.

Output format (MANDATORY):
- Return EXACTLY ONE fenced code block.
- One grounded action per line; lowercase names; no timestamps, no comments, no prose.
Example format:
```

(move-small red loc1_3 loc2_3 loc3_3)
(move-small red loc2_3 loc3_3 loc4_3)
...

```
