"""Run an evaluation on a remote Linux machine: sync the code, run in tmux, pull the rows back.

Training-time scaffolding, never imported by the agent.

    python -m training.ops.remote fetch REMOTE_FILE LOCAL_FILE     # copy a checkpoint that is being rewritten
    python -m training.ops.remote sync                             # push the working tree to the remote directory
    python -m training.ops.remote status                           # tmux sessions and load of the machine

The machine is named by two settings, given as flags or environment variables:

    DOOM_BOX       user@host:port                    (--box)
    DOOM_BOX_DIR   directory of this checkout there  (--box-dir), for example /workspace/eval/doom_ai
    DOOM_BOX_PY    python interpreter there          (--box-python, default /opt/venv/bin/python)

Rules this module keeps, because the machine is shared with training runs:

* it only ever writes inside DOOM_BOX_DIR and only syncs with deletion into a directory that is
  empty or carries the marker file `.promote_eval_dir` (so a mistyped directory cannot wipe a
  training run);
* every job runs in its own tmux session whose name starts with `eval-`, and only sessions and
  process ids started here are ever stopped: nothing is matched by name;
* the code is sent with rsync without `.git`, `.venv`, `runs`, `data`, caches and any `.env`
  file, so no credentials and no history leave the development machine.

A checkpoint that a training run rewrites every few minutes can be read half-written. `fetch`
therefore copies it on the remote side into DOOM_BOX_DIR's parent first, loads the copy there to
prove it is whole, retries when it is torn, and only then downloads the copy.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from training import paths

MARKER = ".promote_eval_dir"
SESSION_PREFIX = "eval-"
EXCLUDES = (".git", ".venv", "runs", "data", "__pycache__", ".pytest_cache", "_vizdoom.ini", ".env", ".env.*", "*.env")
MAX_REMOTE_WORKERS = 64      # worker processes one evaluation may start on the shared machine
SSH_OPTIONS = ("-o", "ConnectTimeout=20", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30")


class RemoteError(RuntimeError):
    """A command on the remote machine failed, or the machine could not be reached."""


def session_name(tag: str) -> str:
    """tmux session of the job for `tag`: the fixed prefix plus the tag with odd characters replaced."""
    return SESSION_PREFIX + re.sub(r"[^A-Za-z0-9_-]+", "_", tag)


@dataclass
class Box:
    """One remote machine and the directory this checkout lives in there."""

    target: str                      # user@host
    port: int = 22
    directory: str = ""
    python: str = "/opt/venv/bin/python"

    @classmethod
    def from_settings(cls, box: str | None = None, directory: str | None = None, python: str | None = None) -> "Box":
        box = box or os.environ.get("DOOM_BOX", "")
        directory = directory or os.environ.get("DOOM_BOX_DIR", "")
        if not box or not directory:
            raise RemoteError("no remote machine configured: give --box user@host:port and --box-dir DIRECTORY, "
                              "or set DOOM_BOX and DOOM_BOX_DIR")
        target, _, port = box.partition(":")
        directory = directory.rstrip("/")
        if not directory.startswith("/") or directory.count("/") < 2:
            raise RemoteError(f"the remote directory must be an absolute path below the top level, not {directory!r}")
        return cls(target, int(port or 22), directory, python or os.environ.get("DOOM_BOX_PY", "/opt/venv/bin/python"))

    # ---- plumbing -------------------------------------------------------------------------------
    def _ssh(self) -> list[str]:
        return ["ssh", "-p", str(self.port), *SSH_OPTIONS, self.target]

    def run(self, command: str, check: bool = True, timeout: float = 120.0) -> subprocess.CompletedProcess:
        """Run a shell command there. Returns the completed process (stdout and stderr as text)."""
        try:
            done = subprocess.run([*self._ssh(), command], capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise RemoteError(f"no answer from {self.target} within {timeout:.0f} s: {command[:80]}") from exc
        if check and done.returncode != 0:
            raise RemoteError(f"remote command failed with status {done.returncode}: {command[:120]}\n{done.stderr.strip()[-600:]}")
        return done

    def _rsync(self, args: list[str], timeout: float = 900.0) -> None:
        cmd = ["rsync", "-az", "-e", "ssh -p %d %s" % (self.port, " ".join(SSH_OPTIONS)), *args]
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if done.returncode != 0:
            raise RemoteError(f"rsync failed with status {done.returncode}: {done.stderr.strip()[-600:]}")

    # ---- code and files -------------------------------------------------------------------------
    def sync_code(self, root: Path = paths.ROOT) -> None:
        """Mirror the working tree into the remote directory (see EXCLUDES; `runs/` and `data/` there are kept)."""
        d = shlex.quote(self.directory)
        state = self.run(f"if [ -e {d}/{MARKER} ]; then echo ours; elif [ ! -e {d} ] || [ -z \"$(ls -A {d})\" ]; then echo empty; "
                         "else echo foreign; fi").stdout.strip().splitlines()[-1:]
        if state == ["foreign"]:
            raise RemoteError(f"{self.directory} on {self.target} holds files and no {MARKER} marker: refusing to sync into a "
                              "directory this tool did not create")
        if state not in (["ours"], ["empty"]):
            raise RemoteError(f"could not inspect {self.directory} on {self.target}: {state}")
        self.run(f"mkdir -p {d} && touch {d}/{MARKER}")
        excludes = [arg for name in EXCLUDES + (MARKER,) for arg in ("--exclude", name)]
        self._rsync(["--delete", *excludes, f"{root}/", f"{self.target}:{self.directory}/"])

    def push(self, relative: str, root: Path = paths.ROOT) -> None:
        """Send one directory of the working tree (for example a candidate under runs/) to the same place there."""
        relative = relative.strip("/")
        self.run(f"mkdir -p {shlex.quote(self.directory + '/' + relative)}")
        self._rsync(["--delete", "--exclude", "__pycache__", f"{root / relative}/", f"{self.target}:{self.directory}/{relative}/"])

    def pull(self, relative: str, local: Path) -> None:
        """Fetch one file from the remote directory."""
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        self._rsync([f"{self.target}:{self.directory}/{relative.strip('/')}", str(local)])

    # ---- jobs -----------------------------------------------------------------------------------
    def sessions(self) -> list[str]:
        out = self.run("tmux ls -F '#{session_name}' 2>/dev/null || true").stdout
        return [line.strip() for line in out.splitlines() if line.strip() and " " not in line.strip()]

    def run_job(self, session: str, command: str, log: str, status: str, poll_s: float = 8.0,
                timeout_s: float = 3 * 3600.0, echo=print) -> int:
        """Run `command` in the remote directory inside a new tmux session and wait for it.

        `log` and `status` are paths relative to the remote directory: the job's output and its exit
        status. New lines of the log are echoed while waiting. A lost connection does not stop the
        job; calling `wait_job` with the same arguments picks it up again. Returns the exit status.
        """
        if not session.startswith(SESSION_PREFIX):
            raise RemoteError(f"session names start with {SESSION_PREFIX!r}, got {session!r}")
        if session in self.sessions():
            raise RemoteError(f"tmux session {session} already exists on {self.target}: a job with this tag is still running "
                              f"(attach with `tmux attach -t {session}`, or wait for it)")
        d = shlex.quote(self.directory)
        inner = (f"cd {d} && export PYTHONPATH={d}/env && ( {command} ) > {shlex.quote(log)} 2>&1; "
                 f"echo $? > {shlex.quote(status)}")
        self.run(f"cd {d} && mkdir -p {shlex.quote(str(Path(log).parent))} && rm -f {shlex.quote(status)} && "
                 f"tmux new-session -d -s {shlex.quote(session)} {shlex.quote(inner)}")
        return self.wait_job(log, status, poll_s, timeout_s, echo)

    def wait_job(self, log: str, status: str, poll_s: float = 8.0, timeout_s: float = 3 * 3600.0, echo=print) -> int:
        d = shlex.quote(self.directory)
        started, shown, misses = time.monotonic(), 0, 0
        while True:
            try:
                out = self.run(f"cd {d} && (cat {shlex.quote(status)} 2>/dev/null || echo running) && "
                               f"tail -n +{shown + 1} {shlex.quote(log)} 2>/dev/null", timeout=60.0).stdout.splitlines()
                misses = 0
            except RemoteError:
                misses += 1                                   # a dropped connection: the job runs on in tmux
                if misses > 20:
                    raise
                time.sleep(poll_s)
                continue
            state, lines = (out[0].strip() if out else "running"), out[1:]
            for line in lines:
                echo(f"  [remote] {line}")
            shown += len(lines)
            if state != "running":
                return int(state) if state.lstrip("-").isdigit() else 1
            if time.monotonic() - started > timeout_s:
                raise RemoteError(f"the remote job did not finish within {timeout_s:.0f} s (log: {self.directory}/{log})")
            time.sleep(poll_s)

    # ---- checkpoints ----------------------------------------------------------------------------
    def fetch_checkpoint(self, remote_file: str, local_file, tries: int = 6, wait_s: float = 7.0) -> dict:
        """Download a checkpoint that may be rewritten at any moment. Returns {"path", "iteration", "env_steps", "sha256"}.

        The file is first copied next to the remote directory (the source is only read), the copy is
        loaded there to prove it is complete, and a torn copy is taken again.
        """
        stage = shlex.quote(str(Path(self.directory).parent / "ckpt"))
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", remote_file.strip("/"))[-120:]
        probe = ("import sys, torch\n"
                 "p = torch.load(sys.argv[1], map_location='cpu', weights_only=False)\n"
                 "print('WHOLE', int(p.get('iteration', -1)), int(p.get('env_steps', -1)))\n")
        script = (f"set -e; mkdir -p {stage}; cd {stage}; n=0; while [ $n -lt {int(tries)} ]; do n=$((n+1)); "
                  f"cp {shlex.quote(remote_file)} {name}.tmp; "
                  f"if {shlex.quote(self.python)} -c {shlex.quote(probe)} {name}.tmp; then mv {name}.tmp {name}; "
                  f"sha256sum {name}; exit 0; fi; echo torn copy, attempt $n >&2; sleep {wait_s:g}; done; exit 3")
        done = self.run(script, check=False, timeout=tries * (wait_s + 60.0))
        whole = [line.split() for line in done.stdout.splitlines() if line.startswith("WHOLE ")]
        if done.returncode != 0 or not whole:
            raise RemoteError(f"could not take a complete copy of {remote_file} in {tries} attempts:\n{done.stderr.strip()[-600:]}")
        digest = done.stdout.split()[-2]
        local_file = Path(local_file)
        local_file.parent.mkdir(parents=True, exist_ok=True)
        self._rsync([f"{self.target}:{Path(self.directory).parent}/ckpt/{name}", str(local_file)])
        return {"path": str(local_file), "iteration": int(whole[-1][1]), "env_steps": int(whole[-1][2]), "sha256": digest}


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--box", help="user@host:port of the remote machine (default: $DOOM_BOX)")
    ap.add_argument("--box-dir", help="directory of this checkout on it (default: $DOOM_BOX_DIR)")
    ap.add_argument("--box-python", help="python interpreter on it (default: $DOOM_BOX_PY or /opt/venv/bin/python)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m training.ops.remote", description=__doc__.split("\n")[0])
    add_arguments(ap)
    sub = ap.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch", help="copy a checkpoint that is being rewritten, retrying a torn copy")
    fetch.add_argument("remote_file")
    fetch.add_argument("local_file")
    sub.add_parser("sync", help="push the working tree to the remote directory")
    sub.add_parser("status", help="tmux sessions and load of the remote machine")
    args = ap.parse_args(argv)
    try:
        box = Box.from_settings(args.box, args.box_dir, args.box_python)
        if args.command == "fetch":
            got = box.fetch_checkpoint(args.remote_file, args.local_file)
            print(f"{got['path']}: iteration {got['iteration']}, {got['env_steps']} env steps, sha256 {got['sha256'][:12]}")
        elif args.command == "sync":
            box.sync_code()
            print(f"synced {paths.ROOT} -> {box.target}:{box.directory}")
        else:
            print("\n".join(box.sessions()) or "no tmux sessions")
            print(box.run("uptime").stdout.strip().splitlines()[-1])
    except RemoteError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
