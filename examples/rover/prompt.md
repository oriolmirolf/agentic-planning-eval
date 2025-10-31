You are planning a mission for a small **planetary rover** called **rover1**. It can drive between waypoints, pick up rock samples, drop them at a depot, take photos of landmarks, and recharge its battery at a charging station.

## Mission goals

By the end of the plan, all of the following must be true:

* The rover has **stored** these samples at the sample depot: **sample1, sample2, sample3, sample4, sample5, sample6**.
* The rover has **taken photos** of these objectives: **objective1, objective2, objective3, objective4**.
* The rover finishes **at waypoint1**.

## What the rover can do

Use only these action names (one per line in your plan):

* `move rover1 <from> <to>` — drive along a **one-way road** from `<from>` to `<to>`. Uses **8** battery.
* `take-sample rover1 <sample> <waypoint>` — pick up a sample that lies at `<waypoint>`. Uses **3** battery. The rover can carry at most **2** samples at a time.
* `drop-sample rover1 <sample> <waypoint>` — drop a carried sample at the **sample depot** (this stores it). Uses **2** battery.
* `take-image rover1 <objective> <waypoint>` — take a photo of an objective that is **visible** from the current waypoint. Uses **1** battery.
* `recharge rover1 <waypoint>` — refill the battery to full at the **charging station**.

## Important facts

* **Start:** rover1 is at **waypoint6** with battery **40/100** and carrying **0/2** samples.
* **Charging station:** **waypoint1**.
* **Sample depot (drop point):** **waypoint7**.
* **Battery:** starts at **40**, max **100**.
  Driving costs **8**, picking a sample **3**, dropping a sample **2**, taking a photo **1**. Recharging fills back to **100** (only at waypoint1).
* **Carrying capacity:** at most **2** samples at once. Drop at waypoint7 to free space.

### Where things are

* **Samples on the ground:**

  * sample1 at **waypoint2**
  * sample2 at **waypoint3**
  * sample3 at **waypoint9**
  * sample4 at **waypoint8**
  * sample5 at **waypoint3**
  * sample6 at **waypoint3**
* **Where each objective can be photographed from:**

  * objective1 is visible from **waypoint2**, **waypoint3**, **waypoint4**
  * objective2 is visible from **waypoint5**, **waypoint7**
  * objective3 is visible from **waypoint8**
  * objective4 is visible from **waypoint5**, **waypoint1**

### Map (one-way roads)

You may only drive along the directed roads listed below:

```
(waypoint1 → waypoint5) (waypoint1 → waypoint9)
(waypoint2 → waypoint5)
(waypoint3 → waypoint4) (waypoint3 → waypoint6)
(waypoint4 → waypoint3) (waypoint4 → waypoint8) (waypoint4 → waypoint9)
(waypoint5 → waypoint1) (waypoint5 → waypoint2)
(waypoint6 → waypoint3) (waypoint6 → waypoint7) (waypoint6 → waypoint8)
(waypoint7 → waypoint6)
(waypoint8 → waypoint4) (waypoint8 → waypoint6)
(waypoint9 → waypoint1) (waypoint9 → waypoint4)
```

## Planning tips (not part of the output)

* Mind the **battery**: long drives add up; you can **recharge at waypoint1** whenever needed.
* Respect the **bag size (2)**: after picking up two samples, head to **waypoint7** to drop them before collecting more.
* Take photos when you pass a waypoint that has **visibility** of an objective.

## Output format

* Return **only** the plan, inside **one fenced code block**.
* Use **lowercase** names exactly as shown.
* Put **one action per line**. No explanations, comments, or costs.

Example lines (illustrative only):

```
(move rover1 waypoint6 waypoint3)
(take-sample rover1 sample2 waypoint3)
(take-image rover1 objective1 waypoint3)
(move rover1 waypoint3 waypoint4)
(take-image rover1 objective1 waypoint4)
...
```