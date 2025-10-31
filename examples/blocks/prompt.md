You are solving a **blocks world** planning task with three distinct blocks: **a**, **b**, and **c**. There is a single robot hand that can hold at most one block at a time.

### World description

* A block can be **on** another block (e.g., “a on b”).
* A block can be **on the table**.
* A block is **clear** if nothing is on top of it.
* The hand is either **empty** or **holding** exactly one block.

### Available actions (each action costs 1)

* **pick-up x**: Pick up block x **from the table**. Allowed only if x is clear and the hand is empty.
* **put-down x**: Put down block x **onto the table**. Allowed only if the hand is holding x.
* **stack x y**: Place the held block x **on top of** block y. Allowed only if the hand is holding x and y is clear.
* **unstack x y**: Lift block x **off** block y. Allowed only if x is on y, x is clear, and the hand is empty.

**Effects (intuitive):**

* Picking up removes the block from the table, makes the hand not empty, and the block not clear.
* Putting down places the block on the table, makes it clear, and frees the hand.
* Stacking frees the hand, makes x clear on top, and y not clear.
* Unstacking makes the hand hold x, clears y, and removes the “on x y” relation.

### Initial state

* f, h, and q are all **on the table**.
* f, h, and q are **clear**.
* The hand is **empty**.

### Goal state

* **f is on h**
* **h is on q**
  (i.e., f on h on q)

### Objective

* Find a sequence of actions that achieves the goal from the initial state.
* Minimize the number of actions (each costs 1).
* Use only the listed actions.
* Assume a deterministic, fully observable environment with no parallel actions.

### Output format

* Return **only** the plan, in a **single fenced code block**.
* Use **lowercase** for all action and object names.
* Include **one grounded action per line**, for example:

```
(pick-up a)
(stack a b)
...
```