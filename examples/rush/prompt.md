You are solving a **Rush Hour puzzle**. The world is represented as a **6×6 grid** of positions labeled `loc1_1` through `loc6_6`. Each position can either be **empty** or **occupied by part of a car**. Cars are either **small** (occupying 2 consecutive cells) or **large** (occupying 3 consecutive cells). Cars can only move **forward or backward along their orientation**—they cannot turn.

Each move slides a car by **one grid cell** into an **empty adjacent position** along its line of movement.

---

### World description

* **Positions:** The grid is composed of positions like `loc3_4` (row 3, column 4). Adjacent positions are those that lie directly next to each other in a row or column.
* **Cars:** There are multiple cars (`car1`, `car3`, …, `red`), each either small or large.
* **Small cars** occupy two consecutive positions.
* **Large cars** occupy three consecutive positions.
* **A position** is considered **occupied** if any car currently occupies it.

---

### Available actions

You can **move one car at a time**, following these rules:

#### For a small car:

* It must occupy two adjacent positions, say `p1` and `p2`.
* It may move forward into a new position `p3` that is adjacent to both `p1` and `p2`, **only if `p3` is empty**.
* When it moves:

  * The trailing position `p1` becomes empty.
  * The car now occupies `p2` and `p3`.

#### For a large car:

* It must occupy three consecutive positions, say `p1`, `p2`, `p3`.
* It may move forward into a new position `p4` that extends from the same direction, **only if `p4` is empty**.
* When it moves:

  * The trailing position `p1` becomes empty.
  * The car now occupies `p2`, `p3`, and `p4`.

Each move counts as **one step**.

---

### Initial configuration

* The grid is populated with the following cars:

  * **Small cars:** `car1`, `car3`, `car4`, `car5`, `car10`, `car16`, `car17`, `car18`, `red`
  * **Large cars:** `car13`, `car14`, `car6`

* Each car’s starting positions (grid cells it occupies):

  ```
  car1:  loc1_1, loc1_2
  car13: loc4_1, loc5_1, loc6_1
  car14: loc3_2, loc4_2, loc5_2
  red:   loc1_3, loc2_3
  car16: loc1_4, loc2_4
  car3:  loc3_3, loc3_4
  car4:  loc4_3, loc4_4
  car5:  loc5_3, loc5_4
  car6:  loc6_2, loc6_3, loc6_4
  car10: loc4_5, loc4_6
  car17: loc5_5, loc6_5
  car18: loc2_6, loc3_6
  ```

* All other positions listed above as “occupied” are filled by these cars; all remaining positions are **empty**.

---

### Goal

* The **red car** must reach the **exit** on the rightmost edge of the grid.
* Specifically, it should end up occupying:

  ```
  loc5_3 and loc6_3
  ```

  meaning the red car slides horizontally to the right until it exits the traffic jam.

---

### Objective

* **Find a sequence of moves** that frees the red car and allows it to reach the goal.
* **Minimize the number of moves.**
* Each move corresponds to sliding one car by one cell.
* Assume a deterministic, fully observable environment with no randomness or parallel actions.

---

### Output format

* Return **only** the plan, in a **single fenced code block**.
* Use **lowercase** for all action and object names.
* One action per line, showing which car moves and the positions it moves between.
* Example format:

```
(move-small red loc1_3 loc2_3 loc3_3)
(move-large car13 loc4_1 loc5_1 loc6_1 loc6_2)
...
```
