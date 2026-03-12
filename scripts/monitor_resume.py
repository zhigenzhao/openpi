#!/usr/bin/env python3
"""Continuously monitor a PACE run and auto-resume when needed.

This script shells out to:
  - `pace run status`
  - `pace run resume`
and applies a simple policy:
  - If training is COMPLETE: exit (by default).
  - If SLURM state is active (RUNNING/PENDING/...): keep waiting.
  - Otherwise: attempt `pace run resume`.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from dataclasses import dataclass


ACTIVE_STATES = {
    "PENDING",
    "RUNNING",
    "CONFIGURING",
    "COMPLETING",
    "STAGE_OUT",
    "SUSPENDED",
    "RESIZING",
}

TERMINAL_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "TIMEOUT",
}


@dataclass
class ParsedStatus:
    complete: bool
    training_line: str | None
    slurm_state_raw: str | None
    slurm_state_norm: str | None
    status_output: str


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True)


def _normalize_slurm_state(raw: str | None) -> str | None:
    if raw is None:
        return None
    token = raw.strip()
    if not token:
        return None
    # Sacct/squeue output may include extra suffixes like FAILED+, COMPLETED|...
    token = token.split()[0].split("|")[0].split("+")[0].upper()
    return token


def _parse_status_output(stdout: str) -> ParsedStatus:
    training_match = re.search(r"^\s*Training:\s*(.+?)\s*$", stdout, re.MULTILINE)
    slurm_match = re.search(r"^\s*SLURM State:\s*(.+?)\s*$", stdout, re.MULTILINE)

    training_value = training_match.group(1).strip() if training_match else None
    slurm_value = slurm_match.group(1).strip() if slurm_match else None
    normalized = _normalize_slurm_state(slurm_value)

    return ParsedStatus(
        complete=(training_value == "COMPLETE"),
        training_line=training_value,
        slurm_state_raw=slurm_value,
        slurm_state_norm=normalized,
        status_output=stdout,
    )


def _status_cmd(args: argparse.Namespace) -> list[str]:
    cmd = ["pace", "run", "status", args.run_name]
    if args.config:
        cmd.extend(["-c", args.config])
    if args.remote:
        cmd.append("--remote")
    if args.no_remote:
        cmd.append("--no-remote")
    return cmd


def _resume_cmd(args: argparse.Namespace) -> list[str]:
    cmd = ["pace", "run", "resume", args.run_name]
    if args.config:
        cmd.extend(["-c", args.config])
    if args.remote:
        cmd.append("--remote")
    if args.no_remote:
        cmd.append("--no-remote")
    if args.checkpoint:
        cmd.extend(["--checkpoint", args.checkpoint])
    return cmd


def _print_header(args: argparse.Namespace) -> None:
    print(
        (
            "monitor_resume: run=%s poll=%ss cooldown=%ss max_resumes=%s "
            "remote=%s no_remote=%s"
        )
        % (
            args.run_name,
            args.poll_seconds,
            args.resume_cooldown_seconds,
            args.max_resumes if args.max_resumes >= 0 else "unlimited",
            args.remote,
            args.no_remote,
        ),
        flush=True,
    )


def _should_resume(parsed: ParsedStatus, resume_on_unknown: bool) -> bool:
    if parsed.complete:
        return False

    state = parsed.slurm_state_norm
    if state in ACTIVE_STATES:
        return False
    if state in TERMINAL_STATES:
        return True

    # Missing/unknown state: optionally allow resume.
    return resume_on_unknown


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Continuously monitor `pace run status` and auto-run `pace run resume`."
    )
    parser.add_argument("run_name", help="PACE run name")
    parser.add_argument("-c", "--config", default=None, help="Path to pace yaml config")
    parser.add_argument("--remote", action="store_true", help="Pass --remote to pace commands")
    parser.add_argument("--no-remote", action="store_true", help="Pass --no-remote to pace commands")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path for resume")
    parser.add_argument("--poll-seconds", type=int, default=60, help="Polling interval in seconds")
    parser.add_argument(
        "--resume-cooldown-seconds",
        type=int,
        default=180,
        help="Minimum seconds between resume attempts",
    )
    parser.add_argument(
        "--max-resumes",
        type=int,
        default=-1,
        help="Max resume attempts before exit (-1 = unlimited)",
    )
    parser.add_argument(
        "--resume-on-unknown-state",
        action="store_true",
        help="Resume when SLURM state is missing/unknown",
    )
    parser.add_argument(
        "--exit-on-complete",
        action="store_true",
        default=True,
        help="Exit when status reports Training: COMPLETE (default: true)",
    )
    args = parser.parse_args()

    if args.remote and args.no_remote:
        print("error: cannot set both --remote and --no-remote", file=sys.stderr)
        return 2

    _print_header(args)

    resumes_done = 0
    last_resume_at = 0.0

    status_cmd = _status_cmd(args)
    resume_cmd = _resume_cmd(args)

    while True:
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        status_proc = _run(status_cmd)

        if status_proc.returncode != 0:
            print(f"[{now}] status failed (exit={status_proc.returncode})", flush=True)
            if status_proc.stderr:
                print(status_proc.stderr.strip(), flush=True)
            time.sleep(args.poll_seconds)
            continue

        parsed = _parse_status_output(status_proc.stdout)

        print(
            f"[{now}] training={parsed.training_line or 'UNKNOWN'} "
            f"slurm_state={parsed.slurm_state_raw or 'UNKNOWN'}",
            flush=True,
        )

        if parsed.complete and args.exit_on_complete:
            print(f"[{now}] training complete; exiting", flush=True)
            return 0

        if not _should_resume(parsed, args.resume_on_unknown_state):
            time.sleep(args.poll_seconds)
            continue

        if args.max_resumes >= 0 and resumes_done >= args.max_resumes:
            print(f"[{now}] reached max resume attempts ({args.max_resumes}); exiting", flush=True)
            return 1

        if (time.time() - last_resume_at) < args.resume_cooldown_seconds:
            wait_for = args.resume_cooldown_seconds - int(time.time() - last_resume_at)
            print(f"[{now}] resume cooldown active ({wait_for}s left)", flush=True)
            time.sleep(args.poll_seconds)
            continue

        print(f"[{now}] attempting resume: {' '.join(resume_cmd)}", flush=True)
        resume_proc = _run(resume_cmd)

        if resume_proc.stdout:
            print(resume_proc.stdout.strip(), flush=True)
        if resume_proc.returncode != 0:
            print(f"[{now}] resume failed (exit={resume_proc.returncode})", flush=True)
            if resume_proc.stderr:
                print(resume_proc.stderr.strip(), flush=True)
        else:
            resumes_done += 1
            last_resume_at = time.time()
            print(f"[{now}] resume submitted successfully (count={resumes_done})", flush=True)

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
