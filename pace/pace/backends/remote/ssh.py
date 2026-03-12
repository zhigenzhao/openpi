"""SSH transport for remote orchestration."""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from pace.config.models import RemoteConfig


class SSHRemoteError(RuntimeError):
    """Raised when SSH/rsync operations fail."""


@dataclass
class SSHRemoteExecutor:
    """Execute commands on a remote host over SSH and sync workspace via rsync."""

    host: str
    user: str | None = None
    ssh_options: list[str] = field(default_factory=list)
    rsync_options: list[str] = field(default_factory=list)

    @classmethod
    def from_config(cls, remote: RemoteConfig) -> SSHRemoteExecutor:
        """Build an executor from RemoteConfig."""
        return cls(
            host=remote.host,
            user=remote.user,
            ssh_options=remote.ssh_options,
            rsync_options=remote.rsync_options,
        )

    @property
    def target(self) -> str:
        """Return SSH target in user@host form when user is provided."""
        if self.user:
            return f"{self.user}@{self.host}"
        return self.host

    def _ssh_cmd(self, remote_command: str) -> list[str]:
        return ["ssh", *self.ssh_options, self.target, remote_command]

    def run(
        self,
        command: list[str],
        remote_cwd: str,
        stream_output: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        """Run a command remotely inside `remote_cwd`."""
        command_str = shlex.join(command)
        remote_command = f"cd {shlex.quote(remote_cwd)} && {command_str}"
        ssh_cmd = self._ssh_cmd(remote_command)

        try:
            if stream_output:
                completed = subprocess.run(ssh_cmd, text=True)
            else:
                completed = subprocess.run(
                    ssh_cmd,
                    text=True,
                    capture_output=True,
                )
            return completed
        except FileNotFoundError as exc:
            raise SSHRemoteError("ssh command not found in PATH") from exc

    def ssh_run(self, command: list[str], remote_cwd: str) -> int:
        """Run remote command with streamed output; return exit code."""
        completed = self.run(command, remote_cwd=remote_cwd, stream_output=True)
        return completed.returncode

    def ssh_capture(
        self, command: list[str], remote_cwd: str
    ) -> subprocess.CompletedProcess[str]:
        """Run remote command and capture stdout/stderr."""
        return self.run(command, remote_cwd=remote_cwd, stream_output=False)

    def ensure_remote_dir(self, remote_dir: str) -> None:
        """Create remote directory if missing."""
        completed = self.run(["mkdir", "-p", remote_dir], remote_cwd="/", stream_output=False)
        if completed.returncode != 0:
            raise SSHRemoteError(
                f"Failed to create remote directory '{remote_dir}': {completed.stderr}"
            )

    def sync_project(
        self,
        local_dir: str | Path,
        remote_dir: str,
        excludes: list[str] | None = None,
    ) -> None:
        """Sync local project directory to remote directory with rsync."""
        self.ensure_remote_dir(remote_dir)

        local_dir = str(Path(local_dir).resolve())
        excludes = excludes or []

        cmd = ["rsync", "-a", "--delete", *self.rsync_options]
        for pattern in excludes:
            cmd.extend(["--exclude", pattern])

        cmd.extend([f"{local_dir}/", f"{self.target}:{remote_dir}/"])

        try:
            result = subprocess.run(cmd, text=True, capture_output=True)
        except FileNotFoundError as exc:
            raise SSHRemoteError("rsync command not found in PATH") from exc

        if result.returncode != 0:
            raise SSHRemoteError(
                "rsync failed while syncing local workspace to remote. "
                f"stderr: {result.stderr.strip()}"
            )

    def sync_from_remote(
        self,
        remote_dir: str,
        local_dir: str | Path,
        excludes: list[str] | None = None,
    ) -> None:
        """Sync a remote directory to local path via rsync."""
        local_dir = Path(local_dir).resolve()
        local_dir.mkdir(parents=True, exist_ok=True)
        excludes = excludes or []

        cmd = ["rsync", "-a", *self.rsync_options]
        for pattern in excludes:
            cmd.extend(["--exclude", pattern])
        cmd.extend([f"{self.target}:{remote_dir}/", f"{local_dir}/"])

        try:
            result = subprocess.run(cmd, text=True, capture_output=True)
        except FileNotFoundError as exc:
            raise SSHRemoteError("rsync command not found in PATH") from exc

        if result.returncode != 0:
            raise SSHRemoteError(
                "rsync failed while syncing remote workspace to local. "
                f"stderr: {result.stderr.strip()}"
            )

    def rsync_to_remote(
        self,
        local_dir: str | Path,
        remote_dir: str,
        excludes: list[str] | None = None,
    ) -> None:
        """Copy local directory to remote directory via rsync."""
        self.sync_project(local_dir=local_dir, remote_dir=remote_dir, excludes=excludes)
