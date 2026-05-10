# =============================================================
#   WEB SERVER - Flask API for robot control frontend
#   Serves scene state to webpage and accepts commands.
#
#   Start alongside main.py:
#     python web/server.py
#   Then open: http://localhost:5000
#
#   Endpoints:
#     GET  /scene          current scene state (objects, robot, e-stop)
#     POST /command        send auto order to agent
#     POST /emergency_stop trigger emergency stop
#     POST /resume         clear emergency stop
#     GET  /reachability   reachability check for x,y
# =============================================================

import sys
import os
import threading
import json

# Add parent directory to path so we can import robot modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__, static_folder="static", template_folder="templates")

# Shared state — set by main.py after robot initialises
robot       = None
agent       = None
viz         = None
e_stop      = None
agent_log   = []          # live log of agent actions
agent_running = False


def init(robot_ref, agent_ref, viz_ref, estop_ref):
    """Call from main.py to connect robot to web server."""
    global robot, agent, viz, e_stop
    robot  = robot_ref
    agent  = agent_ref
    viz    = viz_ref
    e_stop = estop_ref
    print("  [WEB] Server connected to robot")


# ----------------------------------------------------------
# API endpoints
# ----------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory("templates", "index.html")


@app.route("/scene")
def scene():
    """Return current scene state as JSON."""
    if robot is None:
        return jsonify({"error": "Robot not connected"}), 503

    objects = {}
    for name in robot.objects:
        pos   = robot.get_object_position(name)
        reach = robot.check_reachability(pos[0], pos[1]) if pos else None
        objects[name] = {
            "position":    list(pos) if pos else None,
            "reachability": reach
        }

    ee  = robot.get_end_effector_position()
    return jsonify({
        "objects":    objects,
        "robot": {
            "position": list(ee),
            "holding":  robot.grasped_object or None
        },
        "obstacles":   list(robot.obstacles.keys()),
        "estop":       e_stop.is_active() if e_stop else False,
        "agent_running": agent_running,
        "log":         agent_log[-20:]   # last 20 log entries
    })


@app.route("/command", methods=["POST"])
def command():
    """Run autonomous agent with given order."""
    global agent_running, agent_log

    if agent is None:
        return jsonify({"error": "Agent not connected"}), 503
    if agent_running:
        return jsonify({"error": "Agent already running"}), 409

    data  = request.get_json()
    order = data.get("order", "").strip()
    if not order:
        return jsonify({"error": "No order provided"}), 400

    agent_log = []

    def run():
        global agent_running
        agent_running = True
        try:
            result = agent.run(order)
            agent_log.append({
                "type":    "result",
                "success": result["success"],
                "summary": result["summary"]
            })
            if viz:
                viz.update_objects()
        except Exception as e:
            agent_log.append({"type": "error", "message": str(e)})
        finally:
            agent_running = False

    t = threading.Thread(target=run, daemon=True)
    t.start()

    return jsonify({"status": "started", "order": order})


@app.route("/emergency_stop", methods=["POST"])
def emergency_stop():
    if e_stop:
        e_stop.trigger("web interface")
    return jsonify({"stopped": True})


@app.route("/resume", methods=["POST"])
def resume():
    if e_stop:
        e_stop.reset()
    return jsonify({"resumed": True})


@app.route("/reachability")
def reachability():
    """Check reachability for given x, y coordinates."""
    if robot is None:
        return jsonify({"error": "Robot not connected"}), 503
    try:
        x = float(request.args.get("x", 0))
        y = float(request.args.get("y", 0))
        result = robot.check_reachability(x, y)
        return jsonify(result)
    except ValueError:
        return jsonify({"error": "Invalid coordinates"}), 400


@app.route("/log", methods=["POST"])
def add_log():
    """Agent loop posts log entries here during execution."""
    data = request.get_json()
    if data:
        agent_log.append(data)
    return jsonify({"ok": True})


def start(port=5000):
    """Start Flask server in background thread."""
    t = threading.Thread(
        target=lambda: app.run(port=port, debug=False, use_reloader=False),
        daemon=True
    )
    t.start()
    print(f"  [WEB] Server running at http://localhost:{port}")
