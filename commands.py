# =============================================================
#   FREE-FORM COMMANDS
#
#   Type anything. Nothing about the scene is written down in
#   advance: no product names, no colour table, no positions.
#
#   1. The camera takes a picture and measures whatever is on the
#      table: where each thing is, its colour as three numbers,
#      its length, width, height, how round it is.
#   2. The language model is shown that picture, with a number
#      drawn beside each thing, and the measurements. It decides
#      what each thing is ("7 is a blue cylinder") and which ones
#      the instruction is about.
#   3. It replies with steps. Code checks every step against what
#      the camera measured before the robot moves: is the gripper
#      free, is the thing small enough to grip, is there room in
#      the box. A plan that fails goes back once with the reasons,
#      then is refused.
#
#   The model chooses and orders the robot's skills. It cannot
#   add one, and it never sends a coordinate: it names a thing by
#   its number, and the position comes from the camera.
# =============================================================

import base64
import json
import math
import struct
import zlib

import numpy as np

import config as cfg

MAX_STEPS = 30


class CommandError(Exception):
    """The instruction cannot be carried out. The message says why."""


SKILLS = {
    "look": "Take a fresh picture and measure the table again. The "
            "gripper must be empty.",
    "pick": "Pick one thing up. Give \"thing\": its number. The gripper "
            "must be empty and the thing must be grippable.",
    "place": "Put down what is held. Give \"on\": the number of a "
             "container to put it inside, or \"table\" for a clear "
             "place on the table. The robot finds the exact spot.",
    "dispatch": "The conveyor takes away the container standing on it, "
                "with everything inside, and brings an empty one. The "
                "gripper must be empty.",
    "say": "Say something to the operator. Give \"text\". Use it to "
           "answer a question or to explain what cannot be done.",
}

SYSTEM_PROMPT = """You are the planner for a robot arm with a camera \
above its table. An operator types an instruction in plain words. You \
work out what they mean and turn it into steps the robot can carry out.

Nobody has told the robot what is on the table. You are given:
- a picture from the overhead camera, with a number drawn beside each \
thing the camera found;
- the camera's measurements of each numbered thing;
- what the robot is able to do.

You identify the things yourself, from the picture and the numbers. \
Reading the measurements:
- colour_rgb is the measured colour, 0-255 per channel.
- size_mm is the footprint seen from above (length, width); height_mm \
is how tall it stands.
- roundness is near 1.0 for a circle seen from above (a cylinder), \
about 0.64 for a square (a cube), lower for a long rectangle (a bar \
or block).
- containers are open boxes that things can be put into. The one with \
on_conveyor true is the one the conveyor can take away.

Reply with one JSON object and nothing else:
{"understood": "<one sentence: what you take the instruction to mean>",
 "names": {"<number>": "<what you call it, e.g. yellow cube>", ...},
 "steps": [{"step": "<skill>", ...}, ...]}

"names" must cover every numbered thing and container.

How to interpret:
- People speak loosely. "box", "tray", "crate", "bin", "dispatch box" \
mean a container; pick the one that fits by colour or position. \
"block", "brick", "bar" mean a long rectangular thing; "can", "tin", \
"puck", "round one" mean a cylinder. Match colours by eye, the way a \
person would.
- Take the most reasonable reading and say so in "understood". Do not \
refuse because the wording is loose.
- Ask (with a single "say" step) only when two readings would make \
the robot do physically different things and you cannot tell which is \
meant, for example "the red one" when two red things are there.
- "all", "everything", "the rest" mean every grippable thing that \
fits the description.

Rules the robot cannot break:
- Use only the skills listed, with the arguments they list.
- Refer to things only by their numbers.
- One gripper: pick, then place, before the next pick.
- Only pick things whose can_grip is true.
- A container takes only as many more things as its free_places.
- The robot does not stack loose things on each other. It puts them \
in a container or on the table.
- If the instruction needs something the robot cannot do, do the part \
that can be done if that is sensible, and explain the rest in a "say" \
step. If none of it can be done, reply with a single "say" step.
- If the instruction is a question, answer it with a single "say" step.
"""


# ---- the picture the model is shown ---------------------------------
_DIGITS = {
    "0": "111101101101111", "1": "010110010010111", "2": "111001111100111",
    "3": "111001111001111", "4": "101101111001001", "5": "111100111001111",
    "6": "111100111101111", "7": "111001001001001", "8": "111101111101111",
    "9": "111101111001111",
}


def _png(pixels):
    """Encode an (h, w, 3) uint8 array as PNG. No imaging library."""
    height, width, _ = pixels.shape
    raw = b"".join(b"\x00" + pixels[row].tobytes() for row in range(height))

    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height,
                                         8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def _draw_number(pixels, text, row, col, scale=4):
    """White tag with black digits, top-left corner at (row, col)."""
    height, width, _ = pixels.shape
    tag_h, tag_w = 5 * scale + 6, len(text) * 4 * scale + 4
    row = int(min(max(row, 0), height - tag_h))
    col = int(min(max(col, 0), width - tag_w))
    pixels[row:row + tag_h, col:col + tag_w] = 255
    pixels[row, col:col + tag_w] = 0
    pixels[row + tag_h - 1, col:col + tag_w] = 0
    pixels[row:row + tag_h, col] = 0
    pixels[row:row + tag_h, col + tag_w - 1] = 0
    for index, char in enumerate(text):
        bits = _DIGITS[char]
        for i in range(15):
            if bits[i] == "1":
                r = row + 3 + (i // 3) * scale
                c = col + 3 + index * 4 * scale + (i % 3) * scale
                pixels[r:r + scale, c:c + scale] = 0


def picture(cell):
    """
    The last camera image, cropped to the table, with each thing's
    number drawn beside it. Returns PNG bytes, or None before the
    first look.
    """
    oc = cell.robot.open_cell
    finder = oc.finder
    if finder is None or getattr(finder, "last_world", None) is None:
        return None
    image = (finder.last_image * 255).astype(np.uint8).copy()
    world = finder.last_world
    X, Y = world[..., 0], world[..., 1]
    keep = (X > 0.22) & (X < 0.82) & (Y > -0.52) & (Y < 0.52)
    rows, cols = np.where(keep)
    r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    for thing in oc.listing():
        if thing.held or thing.under is not None:
            continue
        gap = (X - thing.x) ** 2 + (Y - thing.y) ** 2
        r, c = np.unravel_index(np.argmin(gap), gap.shape)
        if thing.container:
            # Tag just outside one corner, clear of anything inside.
            gap = ((X - (thing.x - thing.long / 2 + 0.02)) ** 2
                   + (Y - (thing.y + thing.short / 2 + 0.005)) ** 2)
            r, c = np.unravel_index(np.argmin(gap), gap.shape)
            _draw_number(image, str(thing.id), r - 28, c - 10)
        else:
            reach = int(max(thing.long, thing.short) * 1000 / 2) + 6
            _draw_number(image, str(thing.id), r - reach - 26, c - 10)
    return _png(image[r0:r1, c0:c1])


# ---- the briefing ----------------------------------------------------
def briefing(cell):
    """What the model is told. Every figure is a camera measurement."""
    oc = cell.robot.open_cell
    things, boxes = [], []
    conveyor = oc.conveyor_container()
    for t in oc.listing():
        rgb = [int(round(255 * float(v))) for v in t.rgb]
        if t.container:
            boxes.append({
                "number": t.id, "colour_rgb": rgb,
                "outer_size_mm": [round(1000 * t.long),
                                  round(1000 * t.short)],
                "wall_height_mm": round(1000 * (t.top - t.floor)),
                "centre_mm": [round(1000 * t.x), round(1000 * t.y)],
                "holds": sorted(o.id for o in oc.listing()
                                if o.inside == t.id),
                "free_places": len(oc.spots_in(t)),
                "on_conveyor": conveyor is not None
                               and conveyor.id == t.id})
            continue
        why = None if t.held else oc.why_not_grip(t)
        things.append({
            "number": t.id, "colour_rgb": rgb,
            "size_mm": [round(1000 * t.long), round(1000 * t.short)],
            "height_mm": round(1000 * t.height),
            "roundness": round(t.roundness, 2),
            "turned_deg": round(math.degrees(t.angle) % 180),
            "centre_mm": [round(1000 * t.x), round(1000 * t.y)],
            "inside_container": t.inside,
            "in_gripper": t.held,
            "can_grip": True if why is None else False,
            **({"cannot_grip_because": why} if why else {})})
    held = oc.held()
    return {
        "cell": "A robot arm with a two-finger gripper stands at a table. "
                "A camera above the table found the numbered things "
                "below. Loose things can be picked up and put into a "
                "container or somewhere clear on the table. A conveyor "
                "runs along the far side and can carry one container "
                "away.",
        "things": things,
        "containers": boxes,
        "gripper": f"holding thing {held.id}" if held else "empty",
        "clear_place_on_table": oc.table_spot() is not None,
        "skills": SKILLS,
    }


# ---- checking a plan ---------------------------------------------------
def validate(steps, cell):
    """
    Walk the plan against what the camera measured. Returns a list of
    problems; empty means every step can be carried out in order.
    """
    oc = cell.robot.open_cell
    if not isinstance(steps, list) or not steps:
        return ["the plan has no steps"]
    if len(steps) > MAX_STEPS:
        return [f"the plan has {len(steps)} steps, more than the "
                f"{MAX_STEPS} allowed"]

    things = {t.id: t for t in oc.listing()}
    held = oc.held()
    holding = held.id if held else None
    room = {t.id: len(oc.spots_in(t)) for t in things.values()
            if t.container}
    where = {t.id: t.inside for t in things.values() if not t.container}
    capacity = {box: free + sum(1 for inside in where.values()
                                if inside == box)
                for box, free in room.items()}
    gone = set()
    conveyor = oc.conveyor_container()
    table_room = 8 if oc.table_spot() is not None else 0
    problems = []

    def number(value):
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
        return None

    for index, step in enumerate(steps, 1):
        if not isinstance(step, dict):
            problems.append(f"step {index} is not an object")
            continue
        name = step.get("step")
        if name not in SKILLS:
            problems.append(f"step {index}: '{name}' is not something this "
                            f"robot can do (it can: {', '.join(SKILLS)})")
            continue

        if name == "say":
            if not str(step.get("text", "")).strip():
                problems.append(f"step {index}: say needs \"text\"")

        elif name == "look":
            if holding is not None:
                problems.append(f"step {index}: put down thing {holding} "
                                f"before looking; the arm would be in the "
                                f"camera's way")

        elif name == "pick":
            ident = number(step.get("thing"))
            thing = things.get(ident)
            if thing is None or ident in gone:
                problems.append(f"step {index}: there is no thing "
                                f"{step.get('thing')} on the table")
                continue
            if holding is not None:
                problems.append(f"step {index}: the gripper is already "
                                f"holding thing {holding}")
                continue
            why = oc.why_not_grip(thing)
            if why:
                problems.append(f"step {index}: thing {ident} cannot be "
                                f"picked up: {why}")
                continue
            if where.get(ident) is not None:
                room[where[ident]] = room.get(where[ident], 0) + 1
            else:
                table_room += 1
            where[ident] = None
            holding = ident

        elif name == "place":
            if holding is None:
                problems.append(f"step {index}: the gripper is empty, there "
                                f"is nothing to put down")
                continue
            target = step.get("on")
            if isinstance(target, str) and target.strip().lower() in (
                    "table", "the table"):
                if table_room < 1:
                    problems.append(f"step {index}: there is no clear place "
                                    f"left on the table")
                    continue
                table_room -= 1
                where[holding] = None
                holding = None
                continue
            ident = number(target)
            box = things.get(ident)
            if box is None or ident in gone:
                problems.append(f"step {index}: there is no thing {target} "
                                f"to put it on; use a container's number "
                                f"or \"table\"")
                continue
            if not box.container:
                problems.append(f"step {index}: thing {ident} is not a "
                                f"container. The robot does not stack "
                                f"loose things; put it in a container or "
                                f"on the table")
                continue
            if room.get(ident, 0) < 1:
                problems.append(f"step {index}: container {ident} has no "
                                f"free place left")
                continue
            room[ident] -= 1
            where[holding] = ident
            holding = None

        elif name == "dispatch":
            if holding is not None:
                problems.append(f"step {index}: put down thing {holding} "
                                f"first")
                continue
            if conveyor is None:
                problems.append(f"step {index}: there is no container on "
                                f"the conveyor to send away")
                continue
            for ident, box in where.items():
                if box == conveyor.id:
                    gone.add(ident)
            room[conveyor.id] = capacity[conveyor.id]
    return problems


# ---- asking the model -------------------------------------------------
def _extract(text):
    text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in the reply")
    return json.loads(text[start:end + 1])


def plan(command, cell, client, model=None, max_attempts=2, log=None):
    """
    Show the model the picture and the measurements, get a plan, check
    it. Returns (steps, understood, names, attempts, usage).
    """
    model = model or cfg.AI["model"]
    log = log or (lambda *a: None)
    facts = json.dumps(briefing(cell), indent=1)
    content = []
    image = picture(cell)
    if image is not None:
        content.append({"type": "image", "source": {
            "type": "base64", "media_type": "image/png",
            "data": base64.b64encode(image).decode("ascii")}})
    content.append({"type": "text",
                    "text": f"Camera measurements and robot skills:\n"
                            f"{facts}\n\nOperator: {command}"})
    messages = [{"role": "user", "content": content}]
    usage = {"input": 0, "output": 0}
    problems = ["no reply"]
    for attempt in range(1, max_attempts + 1):
        reply = client.messages.create(
            model=model, max_tokens=1500, system=SYSTEM_PROMPT,
            messages=messages)
        raw = reply.content[0].text
        counted = getattr(reply, "usage", None)
        if counted is not None:
            usage["input"] += getattr(counted, "input_tokens", 0) or 0
            usage["output"] += getattr(counted, "output_tokens", 0) or 0
        try:
            data = _extract(raw)
            steps = data.get("steps")
            problems = validate(steps, cell)
        except (ValueError, AttributeError, TypeError) as exc:
            problems = [f"the reply was not a usable plan ({exc})"]
        if not problems:
            names = data.get("names")
            names = ({str(k): str(v)[:40] for k, v in names.items()}
                     if isinstance(names, dict) else {})
            return (steps, str(data.get("understood", ""))[:240], names,
                    attempt, usage)
        log("CHECK", "plan refused by the checker: " + "; ".join(problems))
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content":
                         "That plan cannot be carried out: "
                         + "; ".join(problems)
                         + ". Reply with a corrected JSON object, or a "
                           "single say step explaining the problem."})
    raise CommandError("; ".join(problems))


# ---- carrying it out ----------------------------------------------------
def execute(steps, cell, log=None):
    """
    Run a checked plan through the safety gate. Returns
    (steps done, things said, failure text or None).
    """
    log = log or (lambda *a: None)
    oc = cell.robot.open_cell
    robot = cell.supervised
    done, said = [], []

    def force_log(z, force):
        force_log.count += 1
        if force > 0.05 or force_log.count % 6 == 0:
            log("FORCE", f"gripper z={z:.4f} m   contact {force:5.1f} N")
    force_log.count = 0

    def finish(failure=None):
        if not cell.robot.grasped_object:
            log("CAMERA", "looking again to see how things stand now")
            oc.scan()
        return done, said, failure

    for index, step in enumerate(steps, 1):
        name = step["step"]
        cell.audit.log("command_step", index=index, step=step)

        if name == "say":
            said.append(str(step["text"]))
            done.append(step)

        elif name == "look":
            log("CAMERA", "arm parked clear, taking a picture")
            found = oc.scan()
            log("CAMERA", f"{sum(not t.container for t in found)} loose "
                          f"things and {sum(t.container for t in found)} "
                          f"containers measured")
            done.append(step)

        elif name == "pick":
            thing = oc.things.get(int(step["thing"]))
            if thing is None:
                return finish(f"step {index}: thing {step['thing']} is no "
                              f"longer there")
            args = oc.pick_args(thing)
            log("MOVE", f"pick thing {thing.id}: camera puts it at "
                        f"({thing.x:.3f}, {thing.y:.3f}), top face "
                        f"{1000 * (thing.top - oc.table_z):.0f} mm above "
                        f"the table, turned "
                        f"{math.degrees(thing.angle) % 180:.0f} deg")
            result = robot.pick_at(**args)
            if result.get("status") != "success":
                return finish(f"step {index}: could not pick up thing "
                              f"{thing.id}: {result.get('detail')}")
            oc.note_picked(thing)
            log("GRIP", f"holding thing {thing.id}; its foot is "
                        f"{1000 * thing.hold:.1f} mm below the gripper")
            done.append(step)

        elif name == "place":
            thing = oc.held()
            if thing is None:
                return finish(f"step {index}: nothing in the gripper")
            target = step["on"]
            long_one = thing.is_rect and thing.long > thing.short + 0.012
            if isinstance(target, str) and not target.strip().isdigit():
                spot, surface, inside = oc.table_spot(), oc.table_z, None
                label, where_word = "the table", "on"
            else:
                box = oc.things.get(int(target))
                if box is None:
                    return finish(f"step {index}: container {target} is no "
                                  f"longer there")
                spots = oc.spots_in(box)
                spot = spots[0] if spots else None
                surface, inside = box.floor, box.id
                label, where_word = f"container {box.id}", "in"
            if spot is None:
                return finish(f"step {index}: no free place {where_word} {label}")
            touch = surface + thing.hold
            log("MOVE", f"place thing {thing.id} {where_word} {label} at "
                        f"({spot[0]:.3f}, {spot[1]:.3f}); camera puts the "
                        f"surface at z={surface:.4f}, so contact is "
                        f"expected with the gripper at z={touch:.4f}")
            force_log.count = 0
            result = robot.place_by_touch(
                spot[0], spot[1], touch,
                finger_axis="x" if long_one else "y", strict=long_one,
                open_to=min(0.04, thing.grip / 2 + 0.012),
                on_step=force_log, label=label)
            if result.get("status") != "success":
                return finish(f"step {index}: could not put thing "
                              f"{thing.id} {where_word} {label}: "
                              f"{result.get('detail')}. It is still in "
                              f"the gripper")
            log("TOUCH", result.get("detail", ""))
            oc.note_placed(thing, spot[0], spot[1], surface, inside=inside)
            done.append(step)

        elif name == "dispatch":
            log("BELT", "conveyor running: container leaves, empty one "
                        "arrives")
            carried = oc.dispatch()
            log("BELT", f"{carried} thing(s) sent out with it")
            done.append(step)

    return finish()
