"""
Triage script for reviewing and categorizing episodes in a dataset.

Displays the last frame's top camera image for each episode, collects a human
rating, and organizes files into subfolders accordingly.

Example usage:
    python examples/gim/triage_dataset.py --data-dir logs/pi0_gim_tshirt_dagger
    python examples/gim/triage_dataset.py --data-dir logs/pi0_gim_tshirt_dagger --resume
"""

import argparse
import json
import pickle
import shutil
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np


def decompress_jpg_to_image(jpg_bytes: bytes) -> np.ndarray | None:
    """Decompress JPG bytes back to numpy image array."""
    if jpg_bytes is None:
        return None
    try:
        nparr = np.frombuffer(jpg_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    except Exception as e:
        print(f"Warning: Failed to decompress JPG bytes: {e}")
        return None


SUBFOLDER_MAP = {
    "1_good": "good/",
    "1_dagger": "dagger/",
    "2": "positioning/",
    "3": "failure/",
    "4": "discarded/",
}

MENU = """
  1. Good demo
  2. Tshirt loaded successfully but positioning is not ideal
  3. Complete failure
  4. Discard episode
"""


def collect_files(data_dir: Path) -> list[tuple[Path, str]]:
    """Collect all rollout pkl files from root and fails/ subdir.

    Returns list of (filepath, source_label) tuples.
    """
    files = []
    for f in sorted(data_dir.glob("rollout_*.pkl")):
        files.append((f, "root"))
    fails_dir = data_dir / "fails"
    if fails_dir.exists():
        for f in sorted(fails_dir.glob("rollout_*.pkl")):
            files.append((f, "fails"))
    return files


def load_triage_log(log_path: Path) -> dict[str, dict]:
    """Load existing triage log, returning a dict keyed by filename."""
    if not log_path.exists():
        return {}
    with open(log_path) as f:
        entries = json.load(f)
    return {e["filename"]: e for e in entries}


def save_triage_log(log_path: Path, log: dict[str, dict]):
    """Save triage log as a JSON list."""
    with open(log_path, "w") as f:
        json.dump(list(log.values()), f, indent=2)


def count_interventions(episode: list[dict]) -> int:
    """Count steps where grip_active is True on either arm."""
    count = 0
    for step in episode:
        grip = step.get("grip_active", {})
        if grip.get("left_arm", False) or grip.get("right_arm", False):
            count += 1
    return count


def process_episode(filepath: Path, source: str, data_dir: Path) -> dict | None:
    """Process a single episode: display image, get rating, move file.

    Returns a triage log entry dict, or None if user quits.
    """
    with open(filepath, "rb") as f:
        episode = pickle.load(f)

    num_steps = len(episode)
    duration_s = round(num_steps / 50.0, 1)

    if duration_s < 10.0:
        print(f"  WARNING: Episode only {duration_s}s ({num_steps} steps)")

    # Decode and display last frame's top camera
    last_step = episode[-1]
    jpg_bytes = last_step["image"]["top"]["color"]
    img = decompress_jpg_to_image(jpg_bytes)
    if img is not None:
        # Convert BGR (OpenCV) to RGB for matplotlib
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure("Triage - Last Frame (Top Camera)")
        plt.clf()
        plt.imshow(img_rgb)
        plt.axis("off")
        plt.title(filepath.name)
        plt.draw()
        plt.pause(0.1)
    else:
        print("  WARNING: Could not decode top camera image")

    # Print info
    source_label = "[from fails/]" if source == "fails" else "[from root]"
    print(f"\n  File: {filepath.name}  {source_label}")
    print(f"  Duration: {duration_s}s  |  Steps: {num_steps}")
    print(MENU)

    # Get rating
    while True:
        choice = input("  Enter choice (1-4, or q to quit): ").strip()
        if choice == "q":
            return None
        if choice in ("1", "2", "3", "4"):
            break
        print("  Invalid choice. Please enter 1-4 or q.")

    choice = int(choice)

    # Determine destination
    intervention_steps = 0
    has_intervention = False

    if choice == 1:
        intervention_steps = count_interventions(episode)
        has_intervention = intervention_steps > 0
        print(f"  Intervention detected: {intervention_steps}/{num_steps} steps had grip_active")
        if has_intervention:
            dest_subfolder = "dagger/"
        else:
            dest_subfolder = "good/"
    elif choice == 2:
        dest_subfolder = "positioning/"
    elif choice == 3:
        dest_subfolder = "failure/"
    else:
        dest_subfolder = "discarded/"

    # Move file
    dest_dir = data_dir / dest_subfolder
    dest_path = dest_dir / filepath.name

    if filepath == dest_path:
        print(f"  Already in {dest_subfolder} — skipping move.")
    else:
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(filepath), str(dest_path))
        print(f"  Moved to {dest_subfolder}")

    return {
        "filename": filepath.name,
        "label": choice,
        "duration_s": duration_s,
        "steps": num_steps,
        "has_intervention": has_intervention,
        "intervention_steps": intervention_steps,
        "moved_to": dest_subfolder,
    }


def main():
    parser = argparse.ArgumentParser(description="Triage dataset episodes")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("logs/pi0_gim_tshirt_dagger"),
        help="Root directory containing rollout_*.pkl files",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already-triaged files (from triage_log.json)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    log_path = data_dir / "triage_log.json"

    # Collect all files upfront
    files = collect_files(data_dir)
    print(f"Found {len(files)} episode files")

    # Load existing log for resume
    triage_log = load_triage_log(log_path) if args.resume else {}
    if triage_log:
        print(f"Resuming: {len(triage_log)} episodes already triaged")

    triaged_count = 0
    skipped_count = 0

    for filepath, source in files:
        if filepath.name in triage_log:
            skipped_count += 1
            continue

        print(f"\n{'='*60}")
        print(f"  Episode {triaged_count + skipped_count + 1}/{len(files)}")

        entry = process_episode(filepath, source, data_dir)
        if entry is None:
            print("\nQuitting...")
            break

        triage_log[entry["filename"]] = entry
        save_triage_log(log_path, triage_log)
        triaged_count += 1

    plt.close("all")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    counts: dict[str, int] = {}
    for entry in triage_log.values():
        dest = entry["moved_to"]
        counts[dest] = counts.get(dest, 0) + 1
    for dest in sorted(counts):
        print(f"  {dest:<15s} {counts[dest]}")
    print(f"  {'TOTAL':<15s} {sum(counts.values())}")
    print(f"\nTriaged this session: {triaged_count}")
    if skipped_count:
        print(f"Skipped (already triaged): {skipped_count}")


if __name__ == "__main__":
    main()
