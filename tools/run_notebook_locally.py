"""Execute a pipeline notebook's code cells locally against a simulated Google Drive.

The notebook targets Google Colab, so this runner swaps the Colab-only pieces:
the Drive mount becomes a plain directory, `/content/data` moves under
`--runtime`, and DataLoader workers default to 0 because macOS spawns workers
and cannot pickle functions defined in an exec'd namespace.

Usage:
    python tools/run_notebook_locally.py notebooks/sam_reproduction_vi.ipynb \
        --drive /tmp/fake-drive --runtime /tmp/runtime [--plan smoke] [--seed 0]

The same runner executes notebooks/sam_stage2_reproduction_vi.ipynb (plans smoke/rho).

`--drive` must contain `MyDrive/CS2229-SAM/data/cifar-10-python.tar.gz`.
The script prints NOTEBOOK_OK when every code cell has executed.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")


class SimulatedDrive:
    """Stand-in for google.colab.drive that only creates the MyDrive directory."""

    @staticmethod
    def mount(mount_point):
        (Path(mount_point) / "MyDrive").mkdir(parents=True, exist_ok=True)


def substitute_once(source, old, new):
    if source.count(old) != 1:
        raise SystemExit(f"Notebook layout changed; cannot substitute {old!r}")
    return source.replace(old, new)


def localize_cell(source, args, drive, runtime):
    if source.startswith("# =========================\n# CẤU HÌNH THEO BRANCH"):
        source = substitute_once(source, '\nRUN_PLAN = "', f'\nRUN_PLAN = "{args.plan}"  # was "')
        source = substitute_once(source, "\nSEED = ", f"\nSEED = {args.seed}  # was ")
        source = substitute_once(source, "NUM_WORKERS = 2", f"NUM_WORKERS = {args.workers}")
        source = substitute_once(source, 'Path("/content/data")', f'Path("{runtime}/data")')
        source = substitute_once(source, 'Path("/content/drive")', f'Path("{drive}")')
    if "RESULTS_ROOT = setup_google_drive_results(" in source:
        source = substitute_once(
            source,
            "    results_relative_dir=GOOGLE_DRIVE_RESULTS_DIR,\n)",
            "    results_relative_dir=GOOGLE_DRIVE_RESULTS_DIR,\n"
            "    drive_module=SIMULATED_DRIVE,\n"
            "    mount_checker=lambda path: True,\n)",
        )
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("notebook")
    parser.add_argument("--drive", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--plan", default="smoke", choices=("smoke", "core", "rho"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    drive = Path(args.drive).resolve()
    runtime = Path(args.runtime).resolve()
    notebook = json.loads(Path(args.notebook).read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__", "SIMULATED_DRIVE": SimulatedDrive}

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = localize_cell("".join(cell["source"]), args, drive, runtime)
        print(f"--- cell {index} ---", flush=True)
        exec(compile(source, f"<cell {index}>", "exec"), namespace)

    print("NOTEBOOK_OK", flush=True)


if __name__ == "__main__":
    main()
