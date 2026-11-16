#!/usr/bin/env python3
# =============================================================
#   CHECK SETUP - run this first on a new machine
#
#   Says what is installed, what is missing, and the exact
#   command to fix each thing. Changes nothing.
#
#       python check_setup.py
# =============================================================

import importlib
import os
import shutil
import sys

# package to import, what it is for, and whether it is required
PACKAGES = [
    ("numpy", "maths", True),
    ("pybullet", "the physics simulator and the window", True),
    ("anthropic", "talking to the language model", True),
    ("dotenv", "reading your key out of the .env file", True),
    ("pytest", "running the tests", False),
    ("toppra", "smoother trajectories (there is a fallback)", False),
    ("flask", "the optional web view", False),
    ("PIL", "the optional web view", False),
]

PIP_NAME = {"dotenv": "python-dotenv", "PIL": "pillow"}


def line(ok, text, fix=None):
    print(f"  [{'ok ' if ok else 'XX '}] {text}")
    if not ok and fix:
        print(f"         fix: {fix}")


def main():
    print("\n=== CHECKING THIS MACHINE " + "=" * 36)
    problems = 0

    # ---- Python itself ----
    print("\nPython")
    version = sys.version_info
    good = version >= (3, 11)
    line(good, f"Python {version.major}.{version.minor}."
               f"{version.micro}",
         "install Python 3.12 from python.org and tick "
         "\"Add Python to PATH\"")
    if not good:
        problems += 1
    if version >= (3, 14):
        print("         note: very new Python sometimes has no ready-made "
              "pybullet")
        print("         build. If pybullet fails to install, use "
              "Python 3.12.")

    # ---- packages ----
    print("\nPackages")
    missing_required, missing_optional = [], []
    for module, purpose, required in PACKAGES:
        try:
            importlib.import_module(module)
            found = True
        except Exception:
            found = False
        name = PIP_NAME.get(module, module)
        if found:
            line(True, f"{name:<14} {purpose}")
        elif required:
            line(False, f"{name:<14} {purpose}",
                 f"pip install {name}")
            missing_required.append(name)
            problems += 1
        else:
            print(f"  [-- ] {name:<14} {purpose} - not installed, "
                  f"that is allowed")
            missing_optional.append(name)

    # ---- the key ----
    print("\nYour API key")
    here = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(here, ".env")
    if not os.path.exists(env_path):
        line(False, ".env file not found in this folder",
             "create a file called .env containing one line:\n"
             "              ANTHROPIC_API_KEY=your-key-here")
        problems += 1
    else:
        key = ""
        with open(env_path, encoding="utf-8", errors="replace") as handle:
            for row in handle:
                if row.strip().startswith("ANTHROPIC_API_KEY"):
                    key = row.split("=", 1)[-1].strip().strip('"').strip("'")
        if not key or key == "PUT_YOUR_KEY_HERE":
            line(False, ".env found but the key is not filled in",
                 "open .env and replace PUT_YOUR_KEY_HERE with your key")
            problems += 1
        elif len(key) < 20:
            line(False, f".env has a key but it is only {len(key)} "
                        f"characters, which looks wrong",
                 "check you pasted the whole key")
            problems += 1
        else:
            # never print the key
            line(True, f".env has a key ({len(key)} characters, "
                       f"ends ...{key[-4:]})")

    # ---- ffmpeg, only for recording ----
    print("\nRecording video (optional)")
    if shutil.which("ffmpeg"):
        line(True, "ffmpeg found - record_demo.py will work")
    else:
        print("  [-- ] ffmpeg not found - you can still run everything")
        print("         with a window. You only need ffmpeg for")
        print("         record_demo.py, which writes an MP4.")
        print("         fix: Windows:  winget install ffmpeg")
        print("              Mac:      brew install ffmpeg")
        print("              Linux:    sudo apt install ffmpeg")

    # ---- the project files ----
    print("\nProject files")
    for name in ("run_robot.py", "config.py", "cell/packing.py",
                 "scenes/tea.json"):
        there = os.path.exists(os.path.join(here, name))
        line(there, name,
             "you are running this from the wrong folder, or the "
             "unzip was incomplete")
        if not there:
            problems += 1

    # ---- verdict ----
    print("\n" + "=" * 62)
    if missing_required:
        print("\nInstall what is missing with one command:\n")
        print("    pip install -r requirements.txt\n")
    if problems == 0:
        print("Everything needed is here. Start the robot with:\n")
        print("    python run_robot.py --gui\n")
        print("Then type  start  and tell it what to do.")
        return 0
    print(f"{problems} thing(s) need fixing. The fixes are above.")
    print("Run this check again when you have done them.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
