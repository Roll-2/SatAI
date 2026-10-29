"""Chain the post-acquisition phases after streaming finishes.

Waits for ``Src.acquisition.streaming`` to exit, then — all automatic:

1. ``Tools/training/salvage_crops.py`` — optional QC salvage: only when the
   old ``Data/crops/staging_quarantine/`` folder exists;
2. clean ``Data/crops/{train,val,test}`` (derived artifacts, regenerated);
3. ``Src.data.build_dataset``         — stratified 70/15/15 split;
4. ``Tools/training/audit_crops.py``  — QC report / green-light gate;
5. ``Src.training.train``             — full retraining on the new data;
6. write a completion marker with all exit codes.

The chain is intentionally explicit: it never deletes source FITS, and it
only starts after the streaming process is gone. Use ``--dry-run`` to print
the exact commands without waiting or changing anything.

Launch detached via ``Tools/detached_runner.py`` (or ``nohup``)::

    .venv/bin/python Tools/training/phase_chain.py --n-per-class 5000
    .venv/bin/python Tools/training/phase_chain.py --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

STREAMING = "Src.acquisition.streaming"
PROJECT = Path(__file__).resolve().parents[2]
PHASE2_LOG = "/tmp/satai_phase2.log"
MARKER = "/tmp/satai_phases_done"

SALVAGE = PROJECT / "Tools" / "training" / "salvage_crops.py"
AUDIT = PROJECT / "Tools" / "training" / "audit_crops.py"
QUARANTINE = PROJECT / "Data" / "crops" / "staging_quarantine"
SPLIT_DIRS = tuple(PROJECT / "Data" / "crops" / s
                   for s in ("train", "val", "test"))


def _streaming_running() -> bool:
    """True while the Phase 1 streaming process is alive."""
    out = subprocess.run(["pgrep", "-f", STREAMING],
                         capture_output=True, text=True)
    return out.returncode == 0


def _weakest_first() -> bool:
    """True when the streaming run uses the --weakest-first order."""
    out = subprocess.run(["pgrep", "-f", f"{STREAMING}.*weakest-first"],
                         capture_output=True, text=True)
    return out.returncode == 0


def _run(label: str, cmd: list[str], dry_run: bool = False) -> int:
    """Run one phase step, streaming output into the phase-2 log."""
    printable = " ".join(str(part) for part in cmd)
    print(f"[chain] {label}: {printable}", flush=True)
    if dry_run:
        return 0
    with open(PHASE2_LOG, "ab") as log:
        log.write(f"\n===== {label}: {printable} =====\n".encode())
        log.flush()
        rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT,
                             cwd=str(PROJECT))
    print(f"[chain] {label} -> exit {rc}", flush=True)
    return rc


def main() -> None:
    """Wait for streaming, then run the post-acquisition chain."""
    parser = argparse.ArgumentParser(
        description="Post-acquisition chain: salvage, split, audit, train")
    parser.add_argument("--n-per-class", type=int, default=5000,
                        help="target crops per class for the split (default 5000)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the commands and exit without waiting "
                             "or touching any file")
    args = parser.parse_args()

    # Capture the streaming order before waiting: once the process exits,
    # pgrep can no longer tell whether it used --weakest-first.
    weakest_first = _weakest_first()

    if not args.dry_run:
        print(f"[chain] waiting for {STREAMING} to finish...", flush=True)
        while _streaming_running():
            time.sleep(60)
        print("[chain] Phase 1 (streaming) exited — starting post-phases",
              flush=True)

    py = sys.executable

    # 1. Salvage is a one-time migration from the old quarantine layout.
    #    New streaming writes directly to Data/crops/staging, so it is
    #    skipped unless that legacy folder is actually present.
    rc_salvage = 0
    if QUARANTINE.exists():
        salvage_cmd = [py, str(SALVAGE)]
        if weakest_first:
            salvage_cmd.append("--weakest-first")
        rc_salvage = _run("salvage", salvage_cmd, dry_run=args.dry_run)
    else:
        print("[chain] no Data/crops/staging_quarantine — salvage skipped.",
              flush=True)

    # 2. The split directories are derived artifacts: rebuild them from
    #    scratch so stale copies never mix with the new staging.
    if not args.dry_run:
        for split_dir in SPLIT_DIRS:
            shutil.rmtree(split_dir, ignore_errors=True)

    rc_build = _run(
        "build_dataset",
        [py, "-m", "Src.data.build_dataset",
         "--n-per-class", str(args.n_per_class)],
        dry_run=args.dry_run,
    )
    rc_audit = _run("audit", [py, str(AUDIT)], dry_run=args.dry_run)
    rc_train = 0
    if rc_audit == 0:                     # train only on green-light data
        rc_train = _run("train", [py, "-m", "Src.training.train"],
                        dry_run=args.dry_run)

    if not args.dry_run:
        with open(MARKER, "w") as fh:
            fh.write(f"salvage_exit={rc_salvage}\n"
                     f"build_exit={rc_build}\n"
                     f"audit_exit={rc_audit}\n"
                     f"train_exit={rc_train}\n")
    sys.exit(rc_build or rc_train)


if __name__ == "__main__":
    main()
