"""
Process Claude Code agent trajectory logs from a SWE-bench-like log directory.

Directory layout expected:
  <log_dir>/
    <instance_name>/          # one dir per swebench instance
      -<trajectory_dir>/      # dir whose name starts with "-"
        <uuid>.jsonl          # raw trajectory log
      trajectory.json         # written by this script (output)
      ...
"""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Placeholder - implement this to turn the raw JSONL into the desired shape
# ---------------------------------------------------------------------------

def process_trajectory(jsonl_path: Path) -> list:
    """Convert a raw trajectory .jsonl file into a structured trajectory object.

    Reads each line of the JSONL file and includes it in the result unless its
    ``type`` field equals ``"queue-operation"``.

    Args:
        jsonl_path: Path to the .jsonl file inside the trajectory directory.

    Returns:
        A list of trajectory entries (dicts), one per qualifying JSONL line.
    """
    trajectory = []
    with jsonl_path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping invalid JSON on line %d of %s: %s", lineno, jsonl_path.name, exc)
                continue
            if entry.get("type") != "queue-operation":
                trajectory.append(entry)
    return trajectory


# ---------------------------------------------------------------------------
# File-system helpers
# ---------------------------------------------------------------------------

def find_trajectory_jsonl(instance_dir: Path) -> Path | None:
    """Return the path to the trajectory .jsonl file for one instance.

    Looks for a sub-directory whose name starts with '-', then finds the first
    .jsonl file inside it.

    Args:
        instance_dir: Path to one instance's log directory.

    Returns:
        Path to the .jsonl file, or ``None`` if none could be found.
    """
    dash_dirs = [d for d in instance_dir.iterdir() if d.is_dir() and d.name.startswith("-")]

    if not dash_dirs:
        logger.warning("No '-' directory found in %s - skipping", instance_dir)
        return None

    if len(dash_dirs) > 1:
        logger.warning(
            "Multiple '-' directories found in %s; using the first one: %s",
            instance_dir,
            dash_dirs[0].name,
        )

    trajectory_dir = dash_dirs[0]
    jsonl_files = list(trajectory_dir.glob("*.jsonl"))

    if not jsonl_files:
        logger.warning("No .jsonl file found in %s - skipping", trajectory_dir)
        return None

    if len(jsonl_files) > 1:
        logger.warning(
            "Multiple .jsonl files found in %s; using the first one: %s",
            trajectory_dir,
            jsonl_files[0].name,
        )

    return jsonl_files[0]


def process_instance(instance_dir: Path, overwrite: bool = False) -> bool:
    """Process a single instance directory and write ``trajectory.json``.

    Args:
        instance_dir: Path to one instance's log directory.
        overwrite: If ``False``, skip instances that already have a
            ``trajectory.json``.

    Returns:
        ``True`` if processing succeeded, ``False`` otherwise.
    """
    output_path = instance_dir / "trajectory.json"

    if output_path.exists() and not overwrite:
        logger.debug("trajectory.json already exists for %s - skipping", instance_dir.name)
        return True

    jsonl_path = find_trajectory_jsonl(instance_dir)
    if jsonl_path is None:
        return False

    logger.info("Processing %s …", instance_dir.name)
    try:
        trajectory = process_trajectory(jsonl_path)
    except NotImplementedError:
        raise
    except Exception as exc:
        logger.error("Failed to process %s: %s", instance_dir.name, exc)
        return False

    output_path.write_text(json.dumps(trajectory, indent=2, ensure_ascii=False))
    logger.info("  → wrote %s", output_path)
    return True


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process Claude Code agent trajectory logs for each SWE-bench instance."
    )
    parser.add_argument(
        "log_dir",
        type=Path,
        help="Root log directory containing one sub-directory per instance.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Re-process instances that already have a trajectory.json.",
    )
    args = parser.parse_args()

    log_dir: Path = args.log_dir.resolve()
    if not log_dir.is_dir():
        parser.error(f"log_dir does not exist or is not a directory: {log_dir}")

    instance_dirs = sorted(d for d in log_dir.iterdir() if d.is_dir())
    if not instance_dirs:
        logger.warning("No sub-directories found in %s", log_dir)
        return

    succeeded = 0
    failed = 0
    for instance_dir in instance_dirs:
        ok = process_instance(instance_dir, overwrite=args.overwrite)
        if ok:
            succeeded += 1
        else:
            failed += 1

    logger.info("Done. %d succeeded, %d failed (out of %d).", succeeded, failed, len(instance_dirs))


if __name__ == "__main__":
    main()
