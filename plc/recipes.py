# =============================================================
#   RECIPES - pre-approved jobs the PLC can request by number
#
#   A recipe is a fixed, reviewed list of subtasks. It needs no
#   language model, so the cell keeps producing if the model or
#   the network is unavailable. Free-text orders from an operator
#   HMI go through the language model and are then executed by
#   exactly the same gated, verified pipeline.
# =============================================================

RECIPES = {
    1: {
        "name": "dispatch all parts to the conveyor",
        "subtasks": [
            {"action": "pick", "object": "red"},
            {"action": "place_slot", "slot": 1},
            {"action": "pick", "object": "blue"},
            {"action": "place_slot", "slot": 2},
            {"action": "pick", "object": "green"},
            {"action": "place_slot", "slot": 3},
        ],
    },
    2: {
        "name": "dispatch red only",
        "subtasks": [
            {"action": "pick", "object": "red"},
            {"action": "place_slot", "slot": 1},
        ],
    },
    3: {
        # Centre pocket first: with a taller component already seated
        # beside it, the centre insertion was unreliable in testing.
        "name": "assemble unit: bearing, motor, sensor",
        "subtasks": [
            {"action": "insert", "object": "bearing"},
            {"action": "insert", "object": "motor"},
            {"action": "insert", "object": "sensor"},
        ],
    },
}


def get_recipe(number):
    return RECIPES.get(int(number))
