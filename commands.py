# =============================================================
#   FREE-FORM COMMANDS
#
#   Type anything; the language model works out what to do.
#
#   The model is given a briefing first: what this cell is, what is
#   on it right now (from the camera, not from a stored list), what
#   the robot can physically do, and the rules each step has to
#   obey. It replies with a sequence of steps.
#
#   It does not drive the robot. Every step is checked here against
#   the cell's real state before anything moves: the gripper has to
#   be empty before a pick, a nest has to be free before a place, a
#   product has to be visible to be picked. A plan that breaks a
#   rule is sent back once with the reasons, then refused.
#
#   The model cannot invent a new skill. It can only order the ones
#   below, so "stack them in a pyramid" is refused, while "put the
#   yellow one in the reject bin" is four ordinary steps.
# =============================================================

import json

import config as cfg
from cell import packing as P

MAX_STEPS = 24


class CommandError(Exception):
    """The command cannot be carried out. The message says why."""


# ---- what the robot can do ---------------------------------------
# name -> (what it needs, what it does)
SKILLS = {
    "scan": "Look at the cell with the overhead camera and update "
            "what is known to be where. Takes no arguments.",
    "pick": "Pick one item up. Give \"product\". The gripper must be "
            "empty and the camera must be able to see one.",
    "place": "Put down what is held. Give \"where\": a nest number "
             "(1 to the tray's capacity), \"reject\" for the reject "
             "bin, or \"infeed\" to put it back where stock sits.",
    "insert": "Pick an item and seat it in a tray nest in one step. "
              "Give \"product\" and \"nest\". This is the normal way "
              "to fill a tray.",
    "change_format": "Send the current tray out and bring in a tray "
                     "for another format. Give \"format\". The "
                     "gripper must be empty.",
    "dispatch": "Send the tray on the conveyor out of the cell, "
                "finished or not, and bring an empty one in. The "
                "gripper must be empty.",
    "report": "Say something back to the operator. Give \"text\". "
              "Use this to answer a question, or to explain what you "
              "could not do.",
}

SYSTEM_PROMPT = """You plan the work of a robot cell that packs items \
into trays. An operator types an instruction in plain language and you \
turn it into steps the robot can carry out.

You are given a briefing: what the cell is, what the camera can see \
right now, what is already in the tray, and what the robot can do.

Reply with one JSON object and nothing else:
{"steps": [{"step": "<skill>", ...arguments}, ...],
 "note": "<one short sentence for the operator>"}

Rules:
- Use only the skills in the briefing, with the arguments they list.
- Use only product and format names from the briefing. The operator \
may use a colour or a loose description instead; map it to the right \
product yourself using the colours in the briefing.
- The robot has one gripper and can hold one item at a time. It must \
be empty before a pick, an insert, a dispatch or a changeover.
- Only pick what the camera can see, and never more of a product \
than the count given.
- A nest can hold one item. Do not use a nest that is already full, \
and do not use a nest number beyond the tray's capacity.
- Keep the plan as short as the instruction needs. Do not add steps \
the operator did not ask for.
- If the instruction asks for something the robot cannot do, or for \
items that are not there, reply with {"steps": [{"step": "report", \
"text": "<the reason>"}]} and nothing else.
- If the instruction is a question about the cell, answer it with a \
single report step.
"""


# ---- the briefing -------------------------------------------------
def colour_word(rgb):
    """A plain colour name for a product, so a command can use one."""
    names = [("red", (0.8, 0.15, 0.15)), ("orange", (0.93, 0.5, 0.12)),
             ("yellow", (0.96, 0.8, 0.16)), ("green", (0.24, 0.7, 0.46)),
             ("blue", (0.2, 0.45, 0.85)), ("purple", (0.62, 0.26, 0.6)),
             ("pink", (0.9, 0.5, 0.6)), ("cream", (0.95, 0.88, 0.75)),
             ("brown", (0.5, 0.35, 0.2)), ("grey", (0.5, 0.5, 0.5)),
             ("white", (0.95, 0.95, 0.95)), ("black", (0.1, 0.1, 0.1))]
    best, best_distance = "", 9.0
    for name, ref in names:
        distance = sum((a - b) ** 2 for a, b in zip(rgb[:3], ref)) ** 0.5
        if distance < best_distance:
            best, best_distance = name, distance
    return best


def briefing(cell):
    """Everything the model is told about the cell, as it is now."""
    line = cell.robot.packing_line
    tray = {}
    for nest, record in sorted(line.nests.items()):
        tray[nest] = record["product"] if record.get("gripped") else None
    capacity = P.nests_per_tray(line.format)
    return {
        "cell": f"A robot arm packs {P.SCENE['item']}s into a tray on a "
                f"conveyor. An overhead camera finds the items. Items "
                f"are seated in tray nests with 2 mm clearance, using "
                f"force feedback.",
        "scene": P.SCENE["name"],
        "products": {name: {"colour": colour_word(rgba),
                            "also_called": f"the {colour_word(rgba)} "
                                           f"{P.SCENE['item']}"}
                     for name, rgba in P.FLAVOUR_COLOURS.items()},
        "formats": {name: {"item_size_mm": round(f["carton"] * 1000),
                           "tray_holds": P.nests_per_tray(name),
                           "tray_layout": f"{f['tray']['rows']} rows x "
                                          f"{f['tray']['cols']} columns"}
                    for name, f in P.FORMATS.items()},
        "current_format": line.format,
        "tray_nests": capacity,
        "tray_now": {str(n): (p or "empty") for n, p in tray.items()}
                    or {str(n): "empty" for n in range(1, capacity + 1)},
        "camera_sees_now": {fmt: line.stock_levels(fmt)
                            for fmt in P.FORMATS},
        "gripper": cell.robot.grasped_object or "empty",
        "skills": SKILLS,
    }


# ---- checking a plan ----------------------------------------------
def validate(steps, cell):
    """
    Walk the plan against the cell's state. Returns a list of
    problems; empty means every step can be carried out in order.
    """
    line = cell.robot.packing_line
    problems = []
    if not isinstance(steps, list) or not steps:
        return ["the plan has no steps"]
    if len(steps) > MAX_STEPS:
        return [f"the plan has {len(steps)} steps, more than the "
                f"{MAX_STEPS} allowed"]

    fmt = line.format
    holding = cell.robot.grasped_object
    stock = {f: dict(line.stock_levels(f)) for f in P.FORMATS}
    filled = {n for n, r in line.nests.items() if r.get("gripped")}

    for index, step in enumerate(steps, 1):
        if not isinstance(step, dict):
            problems.append(f"step {index} is not an object")
            continue
        name = step.get("step")
        if name not in SKILLS:
            problems.append(f"step {index}: '{name}' is not a skill this "
                            f"robot has (it has: {', '.join(SKILLS)})")
            continue

        if name == "report":
            if not str(step.get("text", "")).strip():
                problems.append(f"step {index}: report needs \"text\"")

        elif name == "scan":
            pass

        elif name in ("pick", "insert"):
            product = step.get("product")
            if product not in P.FLAVOUR_COLOURS:
                problems.append(f"step {index}: '{product}' is not a "
                                f"product here (they are: "
                                f"{', '.join(P.FLAVOUR_COLOURS)})")
                continue
            if holding:
                problems.append(f"step {index}: the gripper is already "
                                f"holding something")
                continue
            left = stock[fmt].get(product, 0)
            if left < 1:
                problems.append(f"step {index}: the camera cannot see "
                                f"any {product} in the {fmt} size")
                continue
            stock[fmt][product] = left - 1
            if name == "pick":
                holding = product
            else:
                nest = step.get("nest")
                if not isinstance(nest, int) or isinstance(nest, bool):
                    problems.append(f"step {index}: insert needs a nest "
                                    f"number")
                    continue
                if not 1 <= nest <= P.nests_per_tray(fmt):
                    problems.append(f"step {index}: nest {nest} does not "
                                    f"exist; this tray has "
                                    f"{P.nests_per_tray(fmt)}")
                    continue
                if nest in filled:
                    problems.append(f"step {index}: nest {nest} is "
                                    f"already full")
                    continue
                filled.add(nest)

        elif name == "place":
            if not holding:
                problems.append(f"step {index}: the gripper is empty, "
                                f"there is nothing to put down")
                continue
            where = str(step.get("where", "")).strip().lower()
            if where in ("reject", "reject bin", "bin"):
                holding = None
            elif where in ("infeed", "stock", "back"):
                stock[fmt][holding] = stock[fmt].get(holding, 0) + 1
                holding = None
            else:
                digits = "".join(c for c in where if c.isdigit())
                if not digits:
                    problems.append(f"step {index}: '{where}' is not a "
                                    f"place; use a nest number, "
                                    f"'reject' or 'infeed'")
                    continue
                nest = int(digits)
                if not 1 <= nest <= P.nests_per_tray(fmt):
                    problems.append(f"step {index}: nest {nest} does not "
                                    f"exist; this tray has "
                                    f"{P.nests_per_tray(fmt)}")
                    continue
                if nest in filled:
                    problems.append(f"step {index}: nest {nest} is "
                                    f"already full")
                    continue
                filled.add(nest)
                holding = None

        elif name in ("dispatch", "change_format"):
            if holding:
                problems.append(f"step {index}: put down what the "
                                f"gripper is holding first")
                continue
            if name == "change_format":
                new = step.get("format")
                if new not in P.FORMATS:
                    problems.append(f"step {index}: '{new}' is not a "
                                    f"format here (they are: "
                                    f"{', '.join(P.FORMATS)})")
                    continue
                fmt = new
            filled = set()
    return problems


# ---- asking the model ---------------------------------------------
def _extract(text):
    text = text.strip()
    if "```" in text:
        text = text.split("```")[1]
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in the reply")
    return json.loads(text[start:end + 1])


def plan(command, cell, client, model=None, max_attempts=2):
    """Ask the model for a plan and check it. Raises CommandError."""
    model = model or cfg.AI["model"]
    facts = json.dumps(briefing(cell), indent=1)
    messages = [{"role": "user",
                 "content": f"Cell briefing:\n{facts}\n\n"
                            f"Operator: {command}"}]
    problems = ["no reply"]
    for attempt in range(1, max_attempts + 1):
        reply = client.messages.create(
            model=model, max_tokens=900, system=SYSTEM_PROMPT,
            messages=messages)
        raw = reply.content[0].text
        try:
            data = _extract(raw)
            steps = data.get("steps")
            problems = validate(steps, cell)
        except (ValueError, AttributeError, TypeError) as exc:
            problems = [f"the reply was not a usable plan ({exc})"]
        if not problems:
            return steps, str(data.get("note", ""))[:160], attempt
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content":
                         "That plan cannot be carried out: "
                         + "; ".join(problems)
                         + ". Reply with a corrected JSON object, or a "
                           "single report step explaining the problem."})
    raise CommandError("; ".join(problems))


# ---- carrying it out ------------------------------------------------
def execute(steps, cell, say=print):
    """
    Run a checked plan. Each step goes through the same safety gate
    and audit trail as any other motion. Stops at the first step that
    fails and says which.
    """
    from packing_orders import quiet_scan
    line = cell.robot.packing_line
    robot = cell.robot
    done, said = [], []

    for index, step in enumerate(steps, 1):
        name = step["step"]
        cell.audit.log("command_step", index=index, step=step)

        if name == "report":
            said.append(step["text"])
            say(f"  robot: {step['text']}")
            done.append(step)
            continue

        if name == "scan":
            quiet_scan(cell)
            seen = ", ".join(f"{k} {v}" for k, v in
                             line.stock_levels().items() if v)
            say(f"  looked: {seen or 'nothing'}")
            done.append(step)
            continue

        if name in ("pick", "insert"):
            product = step["product"]
            detection = line.take(product)
            if detection is None:
                return done, said, (f"step {index}: no {product} left "
                                    f"to pick")
            spec = line.pick_spec(detection)
            if name == "insert":
                nest = step["nest"]
                line.nests[nest] = {"nest": nest, "product": product,
                                    "detection": detection,
                                    "gripped": None}
                result = cell.agent.run(
                    f"insert {product}",
                    subtasks=[{"action": "insert", "product": product,
                               "nest": nest, "pick": spec,
                               "pocket": P.nest_pocket(line.format, nest),
                               "label": f"{product} into nest {nest}"}])
                if not result["success"]:
                    return done, said, (f"step {index}: could not seat "
                                        f"the {product} in nest {nest}")
            else:
                picked = robot.pick_at(
                    spec["x"], spec["y"], spec["centre_z"],
                    finger_axis=spec["finger_axis"],
                    open_to=spec["open_to"],
                    grasp_depth=spec["grasp_depth"], label=product)
                if picked.get("status") != "success":
                    return done, said, (f"step {index}: "
                                        f"{picked.get('detail')}")
            done.append(step)
            continue

        if name == "place":
            where = str(step["where"]).strip().lower()
            held = robot.grasped_object
            if where in ("reject", "reject bin", "bin"):
                target = P.REJECT_BIN[len(line.rejected)
                                      % len(P.REJECT_BIN)]
                line.rejected.append(held)
                result = robot.place_object(*target)
            elif where in ("infeed", "stock", "back"):
                home = robot.stations.get(held, (0.38, 0.0))
                result = robot.place_object(*home)
            else:
                nest = int("".join(c for c in where if c.isdigit()))
                pocket = P.nest_pocket(line.format, nest)
                line.nests[nest] = {"nest": nest,
                                    "product": held.split("_")[0],
                                    "detection": None, "gripped": None}
                result = robot.insert_part(held, pocket)
                if result.get("status") == "success":
                    line.record_gripped(nest, held)
            if isinstance(result, dict) and result.get("status") != "success":
                return done, said, f"step {index}: {result.get('detail')}"
            done.append(step)
            continue

        if name in ("dispatch", "change_format"):
            line.eject_tray()
            line.load_tray(step.get("format", line.format))
            for _ in range(60):
                robot.step()
            cell.audit.log("tray_in", format=line.format)
            quiet_scan(cell)
            say(f"  tray out, empty {line.format} tray in")
            done.append(step)
            continue

    return done, said, None
