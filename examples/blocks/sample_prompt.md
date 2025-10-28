You are given a Blocks World problem with 3 blocks (a, b, c).

Initial state (English, but the PDDL is available to the evaluator):
- All three blocks a, b, c are on the table.
- All three blocks are clear.
- The hand is empty.

Goal:
- Stack a on b, and b on c. That is: (on a b) and (on b c)

Actions you can use come from the domain; their schemas (names & argument order) are provided separately.

Output format:
- Return ONLY the plan in a single fenced code block.
- Lowercase action names; lowercase object names a,b,c.
- One grounded action per line, e.g.
```
(pick-up a)
(stack a b)
...
```
