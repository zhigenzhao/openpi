"""Remote orchestration backends for PACE."""

from pace.backends.remote.ssh import SSHRemoteError, SSHRemoteExecutor

__all__ = ["SSHRemoteError", "SSHRemoteExecutor"]
