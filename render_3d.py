"""
Launcher for 3D Brain Tumor Renderer.
Delegates directly to my_try_init/render_3d.py.
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_SCRIPT = os.path.join(BASE_DIR, "my_try_init", "render_3d.py")

if __name__ == "__main__":
    import subprocess
    cmd = [sys.executable, TARGET_SCRIPT] + sys.argv[1:]
    sys.exit(subprocess.call(cmd))
