# Design Note — LLM Agent in a Virtual World

## Why PyBullet + Franka Panda

Most agent-in-a-world submissions use a 2D grid or text environment because they are easy to define. I chose a 3D physics simulation with a real robot URDF for one reason: **the hard problems in robotics are physical, not textual**.

A grid world lets an agent "pick up" an object by setting a flag. PyBullet forces you to deal with IK failures, grasp drift, object rolling, reachability limits, and visual servo oscillation — problems that actually exist on real hardware. The agent harness I built has to reason about all of these, not just plan a sequence of steps.

The Franka Panda was chosen because its URDF ships with PyBullet, its kinematics are well-characterised (±110° working arc, 0.855m max reach), and it is representative of the class of arms used in real warehouse automation.

---

## Observation Design

The core question was: **what does the agent need to know, and what would confuse it?**

Early versions gave the agent raw PyBullet coordinates and let it reason freely. This caused two problems: the agent hallucinated objects that did not exist (it invented a "yellow block"), and it attempted picks from positions it could not physically reach.

The final observation format solves both:

```
CONVEYOR SLOTS — never pick from these:
  Slot 1: (0.63, +0.30) — occupied by red
  Slot 2: (0.63,  0.00) — empty
  Slot 3: (0.63, -0.30) — empty

OBJECTS (pickable):
  red:   (0.55, 0.00) | dist=0.55m | ✅ reachable
  blue:  (0.50, 0.30) | dist=0.58m | ✅ reachable
  green: (0.50,-0.30) | dist=0.58m | ✅ reachable

Holding: nothing
Last result: success — placed red at (0.63, 0.30)
Current subtask: pick blue
```

Three decisions here:

1. **Reachability badges** (✅/⚠️/❌) computed before the observation is sent. The agent never attempts a pick it cannot complete — the system blocks it upstream.

2. **Slot occupancy** tracked explicitly. Without this, the agent repeatedly tried to place two objects in the same slot because it could not distinguish "the slot has an object" from "the slot is a destination".

3. **Known objects list** passed to the decomposer. This prevents hallucination — Claude can only plan actions on objects that physically exist in the scene.

---

## Action Space Design

I chose six actions: `pick`, `place_slot`, `place`, `stack`, `scan`, `home`.

The key decision was **`place_slot` as a separate action from `place`**. When the agent uses `place(x, y)` for conveyor dispatch, it sometimes generates coordinates that drift into occupied slots or off the table edge. `place_slot(n)` gives the agent a high-level intent ("dispatch to slot 2") that the system resolves to exact physics-safe coordinates internally. The agent reasons about intent; the harness handles geometry.

---

## What Worked

**Reachability gate** — blocking picks before attempting them eliminated a whole class of failure where the agent would retry an impossible task three times before giving up.

**Observation as contract** — treating the observation format as a strict contract between the environment and the agent (slots are never pickable, known objects list is authoritative) made agent behaviour predictable.

**Layered retry** — each subtask gets three attempts with fresh observations between each. The agent sees its own failure reason and can adapt (e.g. "slot 2 is occupied — try slot 3").

---

## What Did Not Work (and the fixes)

**Visual servo oscillation** — the overhead camera detects object colour blobs, but for objects near the table edge (green at y=−0.30), the initial pixel offset was >50px. Applying correction naively caused the gripper to overshoot 16cm, dragging the object across the table via the grasp constraint. Fix: skip servoing if initial offset exceeds 25px; fall back to known PyBullet position. This is honest — in simulation, IK is exact and the known position needs no correction.

**Torque-sensing descent** — I attempted to detect table contact by monitoring joint torque spikes during descent. PyBullet's torque reporting is noisy and inconsistent with constrained objects; the threshold that worked for one pose failed for another. Fix: calibrated fixed descent heights (pick: z=0.20m, place: z=0.12m) derived from known object geometry. On real hardware this would be replaced by wrist force-torque sensing.

**Slot drift** — placed objects consistently landed 4–7cm from the target slot centre due to constraint release dynamics. Fix: moved slots from x=0.72m to x=0.63m, keeping placed objects within the safe reach zone even after drift, so the agent can recover them if needed.

---

## Real-World Extension

In this simulation, object positions come from PyBullet ground truth. In a real deployment, the observation layer would be replaced by an Intel RealSense D435 depth camera with point cloud segmentation, or a calibrated overhead RGB camera with homography projection. The agent interface — observation format and action space — remains identical. Only the perception pipeline changes.

---

## Algorithms Used

### TOPP-RA — Time-Optimal Path Parameterisation via Reachability Analysis

Takes a path (list of waypoints) and computes the fastest way to move along it without exceeding joint velocity or acceleration limits.

**Analogy:** Driving a fixed road — the route is already drawn, TOPP-RA decides how fast to go at each point. Slow for sharp corners, fast on straights, never exceed the speed limit.

**In this project:**
```
RRT* gives the path:   A → B → C → D
TOPP-RA answers:       speed at each point?
  A: 0.1 m/s  (accelerating)
  B: 0.8 m/s  (cruising)
  C: 0.4 m/s  (slowing for approach)
  D: 0.0 m/s  (stopped at target)
```

Without TOPP-RA the robot moves at constant speed — jerky and mechanically stressful. With it, motion is smooth and time-optimal. Implemented via the `toppra` library in `motion/trajectory.py`.

---

### RRT* — Rapidly-Exploring Random Tree (Star Variant)

Finds a collision-free path from current position to target by growing a tree of random samples through free space, then continuously rewiring to find shorter routes.

**In this project — 6 strategies tried in order:**
```
Strategy 1: Direct           straight line A → B
Strategy 2: Arc left         curve around obstacle left
Strategy 3: Arc right        curve around obstacle right
Strategy 4: High arc         rise over obstacle, cross, descend
Strategy 5: Wide arc         large detour around obstacle
Strategy 6: Elbow reconfigure  change arm pose, then retry
```

First success wins. Most picks succeed on Strategy 1. Complex manoeuvres (e.g. carrying a block past another block) use Strategy 6. Implemented in `motion/rrt_planner.py`.

---

### Inverse Kinematics (IK)

Given a target position (x, y, z), computes the joint angles needed to place the end effector there.

**Analogy:** Your arm has 7 degrees of freedom. When you reach for a cup, your brain solves IK automatically. PyBullet does the same numerically.

Called thousands of times per session — every waypoint in every trajectory requires an IK solve. Uses PyBullet's `calculateInverseKinematics` with 100 iterations and 1mm residual threshold.

---

### Visual Servoing

Uses a downward-facing camera to fine-align the gripper over the target object before descending — corrects residual IK error.

**Analogy:** Parking a car. You drive to roughly the right spot (IK), then use your eyes to nudge into perfect alignment (visual servo).

**How it works:**
```
1. Robot moves above object via IK (~5mm accurate)
2. Overhead camera captures downward RGB frame
3. Colour blob detection finds object centre in pixels
4. Pixel offset → metre correction (0.3× damping factor)
5. Gripper nudges max 8mm per step
6. Repeat until offset < 10px

Safety rule: initial offset > 25px → camera angle unreliable
             → skip servo, use known PyBullet position directly
```

Damping (0.3×) is critical — without it a 90px offset produces a 16cm correction that overshoots and drags the object across the table via the grasp constraint.

---

### Trapezoidal Velocity Profile

Shapes velocity over each movement: accelerate → cruise → decelerate. Applied after TOPP-RA for smooth transitions between waypoints.

**Analogy:** An escalator — it does not start at full speed instantly. It ramps up, runs at cruise speed, then ramps down.

```
Short moves:  triangle profile (no cruise phase)
Long moves:   full trapezoid (ramp up, cruise, ramp down)
```

---

### Gaussian Noise Model (Real-World Extension)

In simulation, object positions come from PyBullet ground truth. On a real robot, a depth camera introduces position uncertainty — typically ±3–5mm for an Intel RealSense at 1m range.

A Gaussian noise model simulates this:
```python
measured_x = true_x + random.gauss(0, 0.003)  # ±3mm std dev
```

Most readings land within 3mm of truth (68% probability). Very few land beyond 9mm (0.3% probability). This is the bell-curve distribution — most error is small, large errors are rare.

Adding this to the observation layer would make the simulation behave like a real perception system, and would make visual servoing necessary rather than optional — closing the loop between simulation and real deployment.

---

### Summary

| Algorithm | Purpose | Location |
|-----------|---------|----------|
| RRT* | Collision-free path planning | `motion/rrt_planner.py` |
| TOPP-RA | Time-optimal trajectory generation | `motion/trajectory.py` |
| Inverse Kinematics | Joint angle computation | PyBullet built-in |
| Visual Servoing | Fine gripper alignment via camera | `motion/visual_servoing.py` |
| Trapezoidal profile | Smooth velocity shaping | `motion/trajectory.py` |
| Gaussian noise | Sensor uncertainty modelling | Real-world extension |
