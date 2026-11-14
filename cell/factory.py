# =============================================================
#   CELL FACTORY - assembles a complete cell
#
#   robot backend -> SupervisedRobot (safety gate + audit)
#                 -> placement verifier
#                 -> agent (language model or recipe)
#                 -> cell controller (PLC handshake)
# =============================================================

import random
from types import SimpleNamespace

import config as cfg
from audit.trail import AuditTrail
from cell import layout as L
from cell.scene import build_cell, sample_part_positions
from place_verifier import PlacementVerifier
from plc.bus import InMemoryBus
from plc.cell_controller import CellController
from plc.interlocks import PLCInterlocks
from plc.soft_plc import SoftPLC
from safety.supervised_robot import SupervisedRobot
from safety.supervisor import SafetyLimits, SafetySupervisor, Zone


def default_limits():
    """Limits for the simulated cell. A real cell loads its own."""
    return SafetyLimits(
        workspace_x=(-0.10, 0.80),
        workspace_y=(-0.60, 0.60),
        workspace_z=(0.05, 1.00),
        reach_min=0.25, reach_max=0.75,
        max_tcp_speed=0.25, max_payload_kg=1.0,
        keep_out=(
            # Operator loading side, behind the robot base.
            Zone("operator_side", x=(-1.0, -0.05), y=(-1.0, 1.0)),
        ),
    )


def build_sim_cell(logger, gui=False, seed=None, randomize=False,
                   audit_path=None, with_plc=True, llm_enabled=False,
                   limits=None, bus=None, verifier_kwargs=None,
                   start_soft_plc=True, scene="dispatch",
                   housing_offset=(0.0, 0.0)):
    """
    Build a complete simulated cell. Returns a namespace with:
      robot (raw backend), supervised, safety, audit, agent,
      verifier, bus, soft_plc, interlocks, controller
    """
    from agent_loop import EMERGENCY_STOP, WarehouseAgent
    from hardware.sim_backend import make_sim_robot

    rng = random.Random(seed)
    robot = make_sim_robot(logger, gui=gui)
    if scene == "assembly":
        # Housing with pockets and three supply stations. The housing
        # may sit a few millimetres off nominal (housing_offset); the
        # controller is only told the nominal pockets.
        from cell import assembly as A
        positions = {}
        build_cell(robot, part_positions={}, gui=gui)
        A.add_housing((A.HOUSING_CENTRE[0] + housing_offset[0],
                       A.HOUSING_CENTRE[1] + housing_offset[1]))
        A.add_components(robot)
        robot.use_pocket_bias = True
        robot.pockets = {n: A.pocket_for(n) for n in A.COMPONENTS}
        robot.stations = {n: c["station"] for n, c in A.COMPONENTS.items()}
        for _ in range(100):
            robot.step()
    elif scene == "packing":
        # Shelf-ready tray on the conveyor, cartons in two formats on
        # the infeed. The tray is a few millimetres off its mark.
        from cell.packing import PackingLine
        positions = {}
        build_cell(robot, part_positions={}, gui=gui)
        robot.use_pocket_bias = True
        robot.packing_line = PackingLine(robot, "standard", housing_offset)
        for _ in range(100):
            robot.step()
    else:
        positions = sample_part_positions(rng) if randomize else None
        build_cell(robot, part_positions=positions, gui=gui)

    audit = AuditTrail(audit_path)

    soft_plc = interlocks = None
    if with_plc:
        bus = bus or InMemoryBus()
        soft_plc = SoftPLC(bus)
        if start_soft_plc:
            soft_plc.start()
        interlocks = PLCInterlocks(bus)
        if start_soft_plc and not interlocks.wait_until_alive():
            raise RuntimeError("soft PLC heartbeat not detected")
    safety = SafetySupervisor(limits or default_limits(),
                              interlock_source=interlocks)

    def on_trip(state, code, reason):
        audit.log("safety_trip", state=state.value, code=code.value,
                  reason=reason)
        EMERGENCY_STOP.trigger(f"{state.value}: {reason}")

    safety.on_trip(on_trip)

    supervised = SupervisedRobot(robot, safety, audit)
    vkw = dict(calibration_path=None,
               drift_warn=cfg.PLACEMENT["warn_m"],
               drift_retry=cfg.PLACEMENT["tolerance_m"],
               max_retries=cfg.PLACEMENT["max_retries"])
    vkw.update(verifier_kwargs or {})
    verifier = PlacementVerifier(supervised, **vkw)

    ai_config = {
        "api_key": cfg.ANTHROPIC_API_KEY,
        "model": cfg.AI["model"],
        "timeout_s": cfg.AI.get("timeout_s", 20.0),
    }
    agent = WarehouseAgent(supervised, ai_config, logger, audit=audit,
                           verifier=verifier, llm_enabled=llm_enabled)

    controller = None
    if with_plc:
        controller = CellController(bus, agent, safety, audit=audit,
                                    interlocks=interlocks)

    return SimpleNamespace(
        robot=robot, supervised=supervised, safety=safety, audit=audit,
        agent=agent, verifier=verifier, bus=bus, soft_plc=soft_plc,
        interlocks=interlocks, controller=controller,
        part_positions=positions or dict(L.PART_NESTS),
    )
