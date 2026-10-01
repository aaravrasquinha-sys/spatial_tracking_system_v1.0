"""
python -m pyslam.tools.bundle <run_directory>

Zips a run_xxx/ directory (config, telemetry, trajectory, logs) into one
file so a failure can be reported by sending a single zip instead of a
back-and-forth about versions/settings/errors.
"""
from __future__ import annotations
import sys
import shutil
import os


def main():
    if len(sys.argv) < 2:
        print("usage: python -m pyslam.tools.bundle <run_directory>")
        sys.exit(1)
    run_dir = sys.argv[1].rstrip("/")
    if not os.path.isdir(run_dir):
        print(f"error: {run_dir} is not a directory")
        sys.exit(1)
    out_path = shutil.make_archive(run_dir, "zip", root_dir=os.path.dirname(run_dir) or ".",
                                    base_dir=os.path.basename(run_dir))
    print(f"Bundled -> {out_path}")


if __name__ == "__main__":
    main()
