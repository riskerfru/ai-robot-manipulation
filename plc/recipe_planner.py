# =============================================================
#   RECIPE PLANNER - typed instruction -> recipe number
#
#   The language model's only output that reaches the PLC is a
#   recipe number. It never gets a channel to the controller: the
#   runner writes the number to RECIPE, and the PLC decides whether
#   that recipe may run (approved list, safety chain, fault latch).
#   A number the PLC does not approve is refused before JOB_START,
#   whatever the model said about it.
#
#   Without a client (no API key, no network) the planner falls
#   back to a keyword match on the recipe names, and says so.
# =============================================================

import json
import re

from plc.recipes import RECIPES

NO_RECIPE = 0


def _recipe_lines():
    return "\n".join(f"{n}: {r['name']}" for n, r in sorted(RECIPES.items()))


SYSTEM = (
    "You choose which pre-approved job a robot cell should run for an "
    "operator's instruction. The cell can only run these recipes, by "
    "number:\n"
    + _recipe_lines() + "\n\n"
    "Pick the single recipe that does what the operator asked. If none "
    "of them does, answer 0: the cell will then do nothing, which is "
    "correct when the instruction asks for something not on the list.\n"
    "You do not decide whether the job may run now; the cell's PLC "
    "checks that separately.\n"
    'Answer with JSON only: {"recipe": <number>, "reason": "<one short '
    'sentence>"}'
)


class PlanResult:
    def __init__(self, recipe, reason, source):
        self.recipe = recipe          # int; 0 = no recipe fits
        self.reason = reason
        self.source = source          # "model" or "keywords"

    def __repr__(self):
        return f"PlanResult({self.recipe}, {self.reason!r}, {self.source})"


def parse_answer(text):
    """Pull {"recipe": n, "reason": ...} out of a model answer."""
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object in model answer: {text!r}")
    data = json.loads(match.group(0))
    recipe = data.get("recipe")
    if isinstance(recipe, bool) or not isinstance(recipe, int):
        raise ValueError(f"recipe is not an integer: {recipe!r}")
    if not 0 <= recipe <= 0xFFFF:
        raise ValueError(f"recipe out of register range: {recipe}")
    return recipe, str(data.get("reason", "")).strip()


def keyword_recipe(instruction):
    """Offline fallback: match the instruction against recipe names."""
    text = (instruction or "").lower()
    if any(w in text for w in ("assemble", "assembly", "insert", "bearing",
                               "motor", "sensor")):
        return 3
    if "red" in text and not any(c in text for c in ("blue", "green", "all")):
        return 2
    if "dispatch" in text or "conveyor" in text or "all" in text:
        return 1
    return NO_RECIPE


def choose_recipe(instruction, client=None, model=None):
    """
    Return a PlanResult. The recipe number is NOT validated here
    against the approved list on purpose: validation is the PLC's
    job, and the runner writes whatever number comes out.
    """
    if client is None:
        n = keyword_recipe(instruction)
        return PlanResult(n, "keyword match (language model unavailable)",
                          "keywords")
    response = client.messages.create(
        model=model,
        max_tokens=1000,
        system=SYSTEM,
        messages=[{"role": "user",
                   "content": f"Operator instruction: {instruction}"}],
    )
    text = "".join(b.text for b in response.content
                   if getattr(b, "type", "") == "text")
    recipe, reason = parse_answer(text)
    return PlanResult(recipe, reason, "model")
