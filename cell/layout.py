# =============================================================
#   CELL LAYOUT - single source of truth for cell geometry
#
#   Every module that needs a table, conveyor or slot coordinate
#   imports it from here. Coordinates are in the robot base frame
#   (metres). On real hardware these values come from the cell
#   survey / calibration, not from this file's defaults.
# =============================================================

# Table top
TABLE_Z      = 0.10          # top surface height
TABLE_CENTRE = (0.42, 0.0)
TABLE_HALF   = (0.42, 0.50, 0.05)   # half extents x, y, z

# Conveyor outfeed with three numbered slots
SLOT_X        = 0.63
SLOT_HALF_W   = 0.07         # half-width of the slot marker
CONVEYOR_SLOTS = {
    1: (SLOT_X,  0.30),
    2: (SLOT_X,  0.00),
    3: (SLOT_X, -0.30),
}

# Infeed part nests (nominal part positions on the table)
PART_Z = TABLE_Z + 0.025
PART_NESTS = {
    "red":   (0.40,  0.00),
    "blue":  (0.40,  0.30),
    "green": (0.40, -0.30),
}
PART_COLOURS = {
    "red":   [1, 0, 0, 1],
    "blue":  [0, 0, 1, 1],
    "green": [0, 1, 0, 1],
}

# Region the infeed may present parts in (used for randomised trials)
INFEED_X = (0.30, 0.50)
INFEED_Y = (-0.32, 0.32)
INFEED_MIN_SPACING = 0.12
