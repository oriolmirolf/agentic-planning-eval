# Blocksworld: Domain & 10 Problems
## Domain

Classic STRIPS Blocksworld with predicates `on`, `on-table`, `clear`, `arm-empty`, `holding`, and actions `pick-up`, `put-down`, `stack`, `unstack`.

## Problems (ascending difficulty)

1. **p01 (3 blocks, build tower):** Three blocks A, B and C start on the table; the goal is a tower where A is on B and B is on C.
2. **p02 (3 blocks, reorder):** B starts on A and C is alone; reorder them so that A is on C and B is on A.
3. **p03 (3 blocks, reverse):** Begin with C on B on A; reverse the stack so that A is on B and B is on C.
4. **p04 (4 blocks, merge two 2-stacks):** Two 2‑block towers (A on B and D on C) must be merged into one four‑block tower A-B-C-D.
5. **p05 (4 blocks, reverse 4-stack):** A four‑block tower D-C-B-A must be reversed to A-B-C-D.
6. **p06 (5 blocks, build 5-stack):** Five separate blocks A-E must be stacked into a tower E-D-C-B-A.
7. **p07 (5 blocks, merge 3-stack & 2-stack):** Two towers (C-B-A and E-D) need to be merged and reversed into a single tower A-B-C-D-E.
8. **p08 (5 blocks, merge three piles):** Three stacks (B-A, D-C and E) are combined into one tower C-D-B-A-E.
9. **p09 (6 blocks, build 6-stack):** Six separate blocks A-F are assembled into a six‑block tower F-E-D-C-B-A.
10. **p10 (7 blocks, merge 3-stack & 4-stack):** A 3‑block tower C-B-A and a 4‑block tower G-F-E-D are merged into a seven‑block tower A-B-C-D-E-F-G.
