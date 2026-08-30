# =============================================================
#   WRIST CAMERA LIVE VIEWER
#   Run alongside main.py to see gripper camera feed
#   
#   Usage: python wrist_viewer.py
#   (run in separate terminal while main.py is running)
# =============================================================

import os
import time
import glob


def view_wrist_frames():
    """
    Watch wrist_frames folder and display latest frame.
    Uses matplotlib for display - no OpenCV needed.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        import numpy as np

        print("Wrist Camera Viewer")
        print("Watching wrist_frames/ folder...")
        print("Press Ctrl+C to stop\n")

        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        ax.set_title("Wrist Camera View", color='white')
        ax.axis('off')
        plt.tight_layout()
        plt.ion()
        plt.show()

        last_file = None

        while True:
            # Find latest frame
            frames = sorted(glob.glob("wrist_frames/frame_*.png"))
            if frames:
                latest = frames[-1]
                if latest != last_file:
                    try:
                        img = mpimg.imread(latest)
                        ax.clear()
                        ax.imshow(img)
                        ax.set_title(
                            f"Wrist Camera — {os.path.basename(latest)}",
                            color='white', fontsize=10
                        )
                        ax.axis('off')
                        fig.canvas.draw()
                        fig.canvas.flush_events()
                        last_file = latest
                        print(f"  Showing: {os.path.basename(latest)}")
                    except Exception as e:
                        pass
            else:
                print("  Waiting for frames... (run main.py and pick something)")

            time.sleep(0.2)

    except ImportError:
        print("matplotlib not found. Install: pip install matplotlib")
    except KeyboardInterrupt:
        print("\nViewer stopped")


if __name__ == "__main__":
    view_wrist_frames()
