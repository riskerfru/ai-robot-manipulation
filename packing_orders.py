# =============================================================
#   PACKING ORDERS - from a typed order to a packed, checked tray
#
#   The language model has one job here: read what the operator
#   typed and return a small structured plan (which format, how
#   many of each flavour). It is given the SKU master and the
#   stock, and it never touches the robot.
#
#   Everything after that is ordinary code: the plan is validated
#   against the tray and the stock, rejected with a reason if it
#   cannot be packed, turned into pick-and-seat steps, executed
#   through the safety gate, and the finished tray is checked.
#
#   plan_offline() does the same job with a small parser, so the
#   cell can be run and tested with no language model at all.
# =============================================================

import json
import re
from dataclasses import dataclass, field

import config as cfg
from cell import packing as P

NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
                "eleven": 11, "twelve": 12, "a": 1, "an": 1}


class PlanError(Exception):
    """The order cannot be packed as asked. The message is for the operator."""


@dataclass
class Plan:
    format: str
    items: list                     # [(flavour, quantity)]
    source: str                     # "language model" or "built-in parser"
    note: str = ""
    attempts: int = 1
    raw: str = field(default="", repr=False)

    def summary(self):
        return ", ".join(f"{q} {f}" for f, q in self.items)


# ---- what the planner is told -----------------------------------

def line_facts(line):
    """The SKU master and line state, as given to the planner."""
    return {
        "scene": P.SCENE.get("name", ""),
        "current_format": line.format,
        "formats": {
            name: {
                "description": f["description"],
                "units_per_tray": P.nests_per_tray(name),
                "products_in_stock": line.stock_levels(name),
            } for name, f in P.FORMATS.items()},
    }


SYSTEM_PROMPT = """You plan orders for a robotic packing cell that packs \
products into trays.

You are given the line facts as JSON: the pack formats, how many units one \
tray holds for each format, and how many units of each product are in stock \
in each format. The operator types an order in plain language.

Reply with one JSON object and nothing else:
{"format": "<format name>",
 "items": [{"product": "<product>", "quantity": <integer>}],
 "note": "<one short sentence for the operator>"}

Rules:
- Use only format and product names that appear in the line facts.
- If the operator does not name a format, keep current_format. Words such \
as "changeover", "switch" or a format name mean a different format.
- The quantities must add up to exactly units_per_tray for the format.
- Never ask for more of a product than is in stock in that format.
- List the items in the order the operator gave them.
- If the order cannot be packed under these rules, reply with \
{"error": "<the reason, in one sentence>"} instead.
"""


# ---- validation: the code has the last word ----------------------

def validate(fmt, items, line):
    """Return (normalised items, problems). No problems means packable."""
    problems = []
    if fmt not in P.FORMATS:
        return [], [f"unknown format '{fmt}' (known: "
                    f"{', '.join(P.FORMATS)})"]
    stock = line.stock_levels(fmt)
    merged = {}
    for flavour, qty in items:
        flavour = str(flavour).strip().lower()
        if flavour not in stock:
            problems.append(f"no '{flavour}' in the {fmt} format")
            continue
        if not isinstance(qty, int) or isinstance(qty, bool) or qty < 1:
            problems.append(f"quantity for {flavour} must be a whole "
                            f"number above zero")
            continue
        merged[flavour] = merged.get(flavour, 0) + qty
    for flavour, qty in merged.items():
        if qty > stock[flavour]:
            problems.append(f"{qty} {flavour} asked for, {stock[flavour]} "
                            f"in stock")
    total = sum(merged.values())
    nests = P.nests_per_tray(fmt)
    if not problems and total != nests:
        problems.append(f"a {fmt} tray holds {nests} cartons, the order "
                        f"has {total}")
    return list(merged.items()), problems


def _extract_json(text):
    text = text.strip()
    if "```" in text:
        text = text.split("```")[1]
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in the reply")
    return json.loads(text[start:end + 1])


def _items_from(data):
    items = data.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("'items' must be a non-empty list")
    out = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("each item must be an object")
        out.append((item.get("product", item.get("flavour")),
                    item.get("quantity")))
    return out


# ---- planners ----------------------------------------------------

def plan_with_llm(order, line, client, model=None, max_attempts=2):
    """
    Ask the language model for a plan and validate it. If the plan
    is not packable the problems are sent back once for a corrected
    plan. A plan that is still wrong is refused.
    """
    model = model or cfg.AI["model"]
    facts = json.dumps(line_facts(line), indent=1)
    messages = [{"role": "user",
                 "content": f"Line facts:\n{facts}\n\nOrder: {order}"}]
    problems = ["no reply"]
    for attempt in range(1, max_attempts + 1):
        response = client.messages.create(
            model=model, max_tokens=400, system=SYSTEM_PROMPT,
            messages=messages)
        raw = response.content[0].text
        try:
            data = _extract_json(raw)
            if "error" in data:
                raise PlanError(str(data["error"]))
            fmt = str(data.get("format", "")).strip().lower()
            items, problems = validate(fmt, _items_from(data), line)
        except PlanError:
            raise
        except (ValueError, AttributeError, TypeError) as exc:
            problems = [f"reply was not a valid plan ({exc})"]
        if not problems:
            return Plan(fmt, items, "language model",
                        note=str(data.get("note", ""))[:120],
                        attempts=attempt, raw=raw)
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content":
                         "That plan cannot be packed: " + "; ".join(problems)
                         + ". Reply with a corrected JSON object, or an "
                         "error object if the order cannot be packed."})
    raise PlanError("; ".join(problems))


def plan_offline(order, line):
    """The same job with a small parser. No language model."""
    text = order.lower()
    fmt = line.format
    for name in P.FORMATS:
        if re.search(rf"\b{name}\b", text):
            fmt = name
    counts = "|".join(list(NUMBER_WORDS) + [r"\d+"])
    flavours = "|".join(P.FLAVOUR_COLOURS)
    items = []
    for qty, flavour in re.findall(rf"\b({counts})\s+({flavours})s?\b",
                                   text):
        items.append((flavour, int(qty) if qty.isdigit()
                      else NUMBER_WORDS[qty]))
    if not items:
        raise PlanError("no quantities and products found in the order "
                        "(the built-in parser needs e.g. 'two lemon')")
    items, problems = validate(fmt, items, line)
    if problems:
        raise PlanError("; ".join(problems))
    return Plan(fmt, items, "built-in parser")


def make_planner(offline=False):
    """Return (planner function, label for the operator)."""
    if offline:
        return plan_offline, "built-in parser (no language model)"
    if not cfg.ANTHROPIC_API_KEY:
        raise SystemExit("ANTHROPIC_API_KEY is not set. Put it in the .env "
                         "file, or run with --offline.")
    import anthropic
    client = anthropic.Anthropic(api_key=cfg.ANTHROPIC_API_KEY,
                                 timeout=cfg.AI.get("timeout_s", 20.0),
                                 max_retries=cfg.AI.get("api_retries", 1))
    return (lambda order, line: plan_with_llm(order, line, client),
            f"language model ({cfg.AI['model']})")


# ---- running an order --------------------------------------------

def quiet_scan(cell):
    """Scan the cell and log what the camera found."""
    import contextlib
    import io
    line = cell.robot.packing_line
    with contextlib.redirect_stdout(io.StringIO()):
        found = line.scan()
    cell.audit.log("scan", format=line.format,
                   found=[{"product": d.product,
                           "x": round(d.x, 4), "y": round(d.y, 4),
                           "pixels": d.pixels} for d in found],
                   hidden=len(line.hidden),
                   stock=line.stock_levels())
    return found


def run_order(cell, order, planner, tray_number=0):
    """
    Plan, change over if needed, pack, check. Returns a dict with the
    plan, the agent result and the tray check. A refused order returns
    {"refused": reason} and moves nothing.
    """
    line = cell.robot.packing_line
    audit = cell.audit
    audit.log("order_text", order=order)
    # The planner is told what the camera can see, so an order for
    # cartons that are not there is refused before anything moves.
    quiet_scan(cell)
    seen = line.stock_levels()
    try:
        plan = planner(order, line)
    except PlanError as exc:
        audit.log("order_refused", order=order, reason=str(exc))
        return {"refused": str(exc)}
    except Exception as exc:                       # network, key, timeout
        reason = f"planner unavailable: {type(exc).__name__}: {exc}"
        audit.log("order_refused", order=order, reason=reason[:200])
        return {"refused": reason}

    audit.log("order_plan", format=plan.format, items=plan.items,
              source=plan.source, note=plan.note, attempts=plan.attempts,
              nests=P.nests_per_tray(plan.format))

    change = plan.format != line.format
    if line.nests or change:
        if line.nests:
            audit.log("tray_out", format=line.format,
                      cartons=len(line.nests))
            line.eject_tray()
        elif line.tray_body is not None:
            line.eject_tray()
        if change:
            f = P.FORMATS[plan.format]
            audit.log("changeover", from_format=line.format,
                      to_format=plan.format,
                      carton_mm=round(f["carton"] * 1000),
                      nests=P.nests_per_tray(plan.format),
                      pattern=f"{f['tray']['rows']} x {f['tray']['cols']}",
                      clearance_mm=P.CLEARANCE * 1000)
        line.load_tray(plan.format)
        audit.log("tray_in", format=plan.format)
        for _ in range(60):
            cell.robot.step()

    # Look again before packing: the cartons the camera can no longer
    # see have gone, and the tray has moved.
    quiet_scan(cell)
    if any(qty > line.stock_levels().get(product, 0)
           for product, qty in plan.items):
        reason = ("the cartons the order needs are no longer all "
                  "visible on the infeed")
        audit.log("order_refused", order=order, reason=reason)
        return {"refused": reason}
    subtasks = line.subtasks_for(plan.items)
    before = len(audit.events("part_rejected"))
    result = cell.agent.run(order, subtasks=subtasks)
    check = line.check_tray()
    rejected = [r["data"] for r in audit.events("part_rejected")[before:]]
    audit.log("tray_check", results=check,
              seated=sum(c["seated"] for c in check), nests=len(subtasks))
    return {"plan": plan, "result": result, "check": check,
            "planned": len(subtasks), "rejected": rejected,
            "seen": seen}
