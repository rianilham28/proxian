#!/usr/bin/env python3
"""update.py — refresh the proxy pool inside a PandaStack microVM sandbox.

End-to-end "update the pool" driver for the two sibling tools that live next to
this file:

    ../proxplore   harvests keyless public proxies          -> proxies.txt
    ../proxalyze   validates a list over raw connections     -> out/{proxies.txt,proxies.jsonl}

It does NOT run them on this machine. It spins up a clean PandaStack microVM
(the `base` template), ships both source trees into it, builds the release binaries there, runs
the harvest, feeds the result straight into the validator, and downloads the
validated output back into the CURRENT working directory (``./out/`` by
default). Finally it commits and pushes that result, so the tracked proxy pool
stays fresh.

Usage
-----
    export PANDASTACK_API_KEY=<paste-your-key>   # required for a live run
    python update.py                        # full pipeline + commit + push
    python update.py --dry-run              # print every command, touch nothing
    python update.py --providers geonode,e89ip --jobs 2048 --keep

Design notes (verified against the pandastack 0.9.0 SDK)
--------------------------------------------------------
* ``Sandbox.exec`` never forwards a per-call timeout — it inherits
  ``Client.timeout`` (30s default). A single command that blocks past that dies
  client-side on a requests read timeout, which rules out ``cargo build --release``
  (proxplore ships ``lto = "fat", codegen-units = 1``), a full harvest, and a
  validator pass over a large list. Two mitigations, both applied:
    1. raise the default client timeout (floor for short-but-slow ops: apt, tar), and
    2. run every *unbounded* stage detached — ``setsid … ; echo $? > rc`` — then
       poll the rc marker with sub-second execs. No single call can trip the cap.
* CPU/memory are baked into the template (``create`` warns and ignores them), so
  ``-j`` / ``--concurrency`` are tuned to the template, not requested here.

Only the standard library is required to *import* this file; ``pandastack`` is
imported lazily so ``--dry-run`` works on a box without the SDK installed.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Everything heavy or irrelevant to a from-scratch build stays out of the upload:
# build artifacts, git history, this tool's own prior output, the multi-hundred-
# MB GeoLite2 databases, and the large checked-in sample/harvest lists.
_EXCLUDE_PARTS = {
    ".git",
    "target",
    "node_modules",
    "__pycache__",
    ".cargo",
    ".github",
    ".venv",
    "venv",
    "dist",
    ".pytest_cache",
    ".ruff_cache",
}
_EXCLUDE_NAMES = {
    "GeoLite2-City.mmdb",
    "GeoLite2-ASN.mmdb",
    "validated.jsonl",
    "candidates.jsonl",
    "proxies.txt",
    "validated-pool.txt",
}
_EXCLUDE_DIR_SUFFIX = {"out", "out-big", "geo", "lists", "examples"}


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- #
# source upload: build a small tarball of both repos, applying the excludes
# --------------------------------------------------------------------------- #
def _include(arcname_parts: list[str]) -> bool:
    for part in arcname_parts:
        if part in _EXCLUDE_PARTS:
            return False
    if arcname_parts and (arcname_parts[-1] in _EXCLUDE_NAMES):
        return False
    if any(seg in _EXCLUDE_DIR_SUFFIX for seg in arcname_parts[:-1]):
        return False
    return True


def build_source_tarball(repos: dict[str, Path], dest: Path) -> tuple[Path, int, int]:
    """Pack ``repos`` (name -> directory) into ``dest`` as ``<name>/...``.

    Returns ``(path, uncompressed_bytes, file_count)``. Pure local work — safe to
    run without any sandbox or API key (``--dry-run`` uses it to prove the upload
    stays small).
    """
    nfiles = 0
    nbytes = 0
    with tarfile.open(dest, "w:gz") as tf:
        for name, root in repos.items():
            root = root.resolve()
            for path in sorted(root.rglob("*")):
                if not path.is_file():
                    continue
                rel_parts = list(path.relative_to(root).parts)
                if not _include([name, *rel_parts]):
                    continue
                st = path.stat()
                tf.add(path, arcname=f"{name}/{'/'.join(rel_parts)}", recursive=False)
                nfiles += 1
                nbytes += st.st_size
    return dest, nbytes, nfiles


# --------------------------------------------------------------------------- #
# remote exec helpers (advisory-safe)
# --------------------------------------------------------------------------- #
_CARGO_ENV = 'export PATH="$HOME/.cargo/bin:$PATH"; . "$HOME/.cargo/env" 2>/dev/null || true'


@dataclass
class Remote:
    """A live sandbox with detached-exec + poll support."""

    sb: object
    client_timeout: float

    def quick(self, cmd: str, *, timeout: float = 120.0) -> tuple[int, str, str]:
        r = self.sb.exec(cmd, timeout_seconds=int(timeout), check=False)  # type: ignore[attr-defined]
        return r.exit_code, r.stdout, r.stderr

    def detached(self, stage: str, cmd: str, *, poll: float = 15.0, budget: float = 5400.0) -> int:
        """Run ``cmd`` detached; return its exit code.

        Launches ``setsid bash -c 'cmd; echo $? > rc'`` so it survives the exec
        session and the client read timeout, then polls the ``rc`` marker with
        short execs. Streams the growing log back so a multi-minute build/harvest
        doesn't look like a hang.
        """
        log = f"/tmp/{stage}.log"
        rc = f"/tmp/{stage}.rc"
        self.quick(f"rm -f {shlex.quote(rc)} {shlex.quote(log)}")
        wrapped = f"{cmd}; echo $? > {shlex.quote(rc)}"
        launch = (
            f"setsid bash -c {shlex.quote(wrapped)} "
            f"> {shlex.quote(log)} 2>&1 < /dev/null & disown; echo launched"
        )
        _, out, err = self.quick(launch)
        if "launched" not in out:
            raise RuntimeError(f"stage {stage!r} failed to launch: {out}{err}")
        t0 = time.monotonic()
        seen = 0
        while True:
            code, so, _ = self.quick(f"test -f {shlex.quote(rc)} && cat {shlex.quote(rc)}")
            if code == 0 and so.strip():
                return int(so.strip().splitlines()[-1])
            code2, tail, _ = self.quick(
                f"wc -c < {shlex.quote(log)} 2>/dev/null; tail -n 40 {shlex.quote(log)} 2>/dev/null"
            )
            if code2 == 0 and tail:
                lines = tail.splitlines()
                if lines and lines[0].strip().isdigit():
                    size = int(lines[0].strip())
                    if size > seen:
                        _log(f"  [{stage}] …{size}B")
                        seen = size
                for ln in lines[1:]:
                    _log(f"  [{stage}] {ln}")
            if time.monotonic() - t0 > budget:
                raise TimeoutError(f"stage {stage!r} exceeded {budget:.0f}s budget")
            time.sleep(poll)


# --------------------------------------------------------------------------- #
# pipeline
# --------------------------------------------------------------------------- #
class Pipeline:
    def __init__(self, a: argparse.Namespace, remote: Remote | None) -> None:
        self.a = a
        self.remote = remote
        self.dry = a.dry_run
        self.r = "/root/proxian"

    def _stages(self) -> list[tuple[str, str, bool]]:
        """Canonical ordered pipeline: ``(label, guest_command, detached)``.

        SINGLE source of truth consumed by both :meth:`run` and :meth:`_dry_run`
        so the executed stages and the printed plan can never drift (grouping,
        detached flags). ``detached=True`` runs via ``setsid`` + rc-poll, which
        is what keeps any command longer than the 30s exec read-timeout alive.
        """
        a = self.a
        prov = f" --providers {','.join(a.providers)}" if a.providers else ""
        conc = f" --concurrency {a.fetch_concurrency}" if a.fetch_concurrency else ""
        jobs = f" -j {a.jobs}" if a.jobs else ""
        out = f"{self.r}/out"
        work = f"{self.r}/work"
        toolchain = " && ".join(
            [
                "DEBIAN_FRONTEND=noninteractive apt-get update -y",
                "DEBIAN_FRONTEND=noninteractive apt-get install -y curl ca-certificates build-essential",
                'curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs '
                "| sh -s -- -y --profile minimal --default-toolchain stable",
            ]
        )
        return [
            # build-essential gives cargo a linker; toolchain is detached because
            # apt + rustup can each exceed a short exec even with the raised floor.
            ("unpack", f"mkdir -p {self.r} && tar -xzf /root/src.tgz -C {self.r}", False),
            ("toolchain", toolchain, True),
            # release-build both crates (proxplore uses lto=fat, codegen-units=1 — slow).
            ("build-proxplore", f"{_CARGO_ENV}; cd {self.r}/proxplore && cargo build --release --bin proxplore", True),
            ("build-proxalyze", f"{_CARGO_ENV}; cd {self.r}/proxalyze && cargo build --release --bin proxalyze", True),
            ("harvest", f"mkdir -p {work}; cd {self.r}/proxplore && ./target/release/proxplore --output {work}/proxies.txt{prov}{conc}", True),
            ("validate", f"mkdir -p {out}; cd {self.r}/proxalyze && ./target/release/proxalyze -i {work}/proxies.txt --out {out}{jobs}", True),
        ]

    def run(self) -> int:
        a = self.a
        repos = {"proxplore": a.proxplore, "proxalyze": a.proxalyze}
        work = a.dest / "out"
        work.mkdir(parents=True, exist_ok=True)

        if self.dry:
            return self._dry_run(repos, work)

        from pandastack import Client, Sandbox, set_default_client  # lazy import

        key = os.environ.get("PANDASTACK_API_KEY", "").strip()
        if not key:
            _log("error: PANDASTACK_API_KEY is unset — a live run needs the real key.")
            return 2
        # raise the read-timeout floor; long stages are detached regardless.
        client = Client(api_key=key, timeout=a.client_timeout)
        set_default_client(client)
        # Never assume the template exists: resolve it and fail with the real
        # names, rather than letting create() throw an opaque API error.
        try:
            names = sorted(t.name for t in client.templates.list())
        except Exception as exc:  # noqa: BLE001 — template list is best-effort
            _log(f"warn: could not list templates ({exc}); trying {a.template!r} anyway")
        else:
            if a.template not in names:
                _log(f"error: template {a.template!r} is not offered by this account.")
                _log(f"       available: {', '.join(names) or '(none)'}")
                return 2
            _log(f"template {a.template!r} confirmed (of {len(names)} available)")
        _log(f"creating sandbox from template {a.template!r} …")
        sb = Sandbox.create(template=a.template, ttl_seconds=a.ttl)
        _log(f"sandbox {sb.id}")
        rm = Remote(sb, a.client_timeout)
        try:
            tg = a.dest / ".proxian-src.tgz"
            _, nbytes, nfiles = build_source_tarball(repos, tg)
            _log(f"uploading {nfiles} files ({nbytes/1e6:.2f} MB unpacked) …")
            sb.filesystem.upload(str(tg), "/root/src.tgz")  # type: ignore[attr-defined]
            tg.unlink(missing_ok=True)

            for name, cmd, detached in self._stages():
                _log(f"== {name} ==")
                if detached:
                    rc = rm.detached(name, cmd, poll=a.poll, budget=a.stage_timeout)
                else:
                    rc, _, err = rm.quick(cmd, timeout=a.client_timeout)
                    if rc != 0:
                        _log(f"  stderr: {err.strip()[:500]}")
                if rc != 0:
                    _log(f"!! stage {name!r} exited {rc} — aborting (keep={a.keep})")
                    if not a.keep:
                        return 1
                    raise RuntimeError(f"stage {name!r} failed rc={rc}")

            # pull validated result into the current directory
            _log("== download ==")
            for fn in ("proxies.txt", "proxies.jsonl"):
                remote = f"{self.r}/out/{fn}"
                code, _, _ = rm.quick(f"test -f {shlex.quote(remote)}")
                if code != 0:
                    _log(f"  missing {remote} on guest — skipping")
                    continue
                local = work / fn
                sb.filesystem.download(remote, str(local))  # type: ignore[attr-defined]
                _log(f"  -> {local}  ({local.stat().st_size} bytes)")

            rc_file = work / "proxplore-raw.txt"
            code, _, _ = rm.quick(f"test -f {self.r}/work/proxies.txt")
            if code == 0:
                sb.filesystem.download(f"{self.r}/work/proxies.txt", str(rc_file))  # type: ignore[attr-defined]

            jp = work / "proxies.jsonl"
            if jp.exists():
                import json as _json
                live = auth = 0
                for _ln in jp.read_text().splitlines():
                    if _ln.strip():
                        _r = _json.loads(_ln)
                        if "auth-required" in (_r.get("tags") or []):
                            auth += 1
                        elif _r.get("exit_ip"):
                            live += 1
                _log(f"usable pool: {live} live · {auth} auth-required "
                     f"(of {live + auth} survivors written)")
            if a.commit:
                grc = git_publish(work, message=a.message, push=a.push, dry=False)
                if grc != 0:
                    _log(f"error: git publish failed (rc={grc}) — saved locally, not pushed")
                    return grc
            return 0
        finally:
            if a.keep:
                _log(f"kept sandbox {sb.id} alive (ttl {a.ttl}s)")
            else:
                sb.kill()  # type: ignore[attr-defined]
                _log("sandbox killed")

    def _dry_run(self, repos: dict[str, Path], work: Path) -> int:
        a = self.a
        tg = Path("/tmp/.proxian-dryrun.tgz")
        _, nbytes, nfiles = build_source_tarball(repos, tg)
        tg.unlink(missing_ok=True)
        print("# update.py — dry run (nothing executed against PandaStack or git)")
        print(f"template={a.template!r}  ttl={a.ttl}s  client_timeout={a.client_timeout}s  keep={a.keep}")
        print(f"proxplore={repos['proxplore']}")
        print(f"proxalyze={repos['proxalyze']}")
        print(f"\n# local upload artifact: {nfiles} files, {nbytes/1e6:.2f} MB unpacked (heavy data excluded)")
        print(f"\n# sandbox filesystem.write  /root/src.tgz  <- tarball")
        print(f"# sandbox filesystem.download {self.r}/out/proxies.txt   -> {work}/proxies.txt")
        print(f"# sandbox filesystem.download {self.r}/out/proxies.jsonl -> {work}/proxies.jsonl")
        print("# sandbox filesystem.download (raw harvest)               -> out/proxplore-raw.txt")
        print("\n# ordered guest commands ([detached] = setsid + rc-poll, survives the 30s exec read-timeout):")
        for label, cmd, d in self._stages():
            print(f"\n# [{label}]{' [detached]' if d else ''}\n{cmd}")
        print("\n# git (local, in the repo containing the result dir):")
        for line in _git_plan(work, message=a.message, push=a.push):
            print(f"  {line}")
        return 0


# --------------------------------------------------------------------------- #
# local git publish
# --------------------------------------------------------------------------- #
def _repo_root(path: Path) -> Path | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return Path(out) if out else None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _git_plan(dest: Path, *, message: str, push: bool) -> list[str]:
    root = _repo_root(dest)
    files = [str(dest / "proxies.txt"), str(dest / "proxies.jsonl")]
    if root is None:
        return [f"(not a git repository at/above {dest} — commit skipped)"]
    rel = [str(Path(f).relative_to(root)) for f in files]
    plan = ["git add " + " ".join(shlex.quote(r) for r in rel), f"git commit -m {shlex.quote(message)}"]
    if push:
        plan.append("git push origin HEAD")
    return plan


def git_publish(dest: Path, *, message: str, push: bool, dry: bool) -> int:
    root = _repo_root(dest)
    if root is None:
        _log(f"git: {dest} is not inside a repo — result saved, commit skipped")
        return 0
    rel = []
    for fn in ("proxies.txt", "proxies.jsonl"):
        f = dest / fn
        if f.exists():
            rel.append(str(f.relative_to(root)))
    if not rel:
        _log("git: no result files to commit")
        return 0
    steps = [["git", "add", *rel], ["git", "commit", "-m", message]]
    if push:
        steps.append(["git", "push", "origin", "HEAD"])
    for s in steps:
        _log("git: " + " ".join(shlex.quote(x) for x in s))
        if dry:
            continue
        r = subprocess.run(s, cwd=root, text=True, capture_output=True)
        if r.returncode != 0:
            _log(f"  git step failed ({s[1]}): {(r.stderr or r.stdout).strip()[:400]}")
            return r.returncode
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _csv(v: str) -> list[str]:
    return [x.strip() for x in v.split(",") if x.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--proxplore", type=Path, default=HERE.parent / "proxplore")
    ap.add_argument("--proxalyze", type=Path, default=HERE.parent / "proxalyze")
    ap.add_argument("--dest", type=Path, default=HERE,
                    help="where the validated result lands (default: this file's directory)")
    ap.add_argument("--template", default="base")
    ap.add_argument("--ttl", type=int, default=3600, help="sandbox TTL seconds")
    ap.add_argument("--client-timeout", type=float, default=900.0,
                    help="pandastack client read-timeout floor (seconds)")
    ap.add_argument("--stage-timeout", type=float, default=5400.0,
                    help="per detached-stage budget before giving up")
    ap.add_argument("--poll", type=float, default=15.0, help="detached poll interval seconds")
    ap.add_argument("--providers", type=_csv, default=None,
                    help="restrict proxplore to these provider ids")
    ap.add_argument("--fetch-concurrency", type=int, default=None, dest="fetch_concurrency")
    ap.add_argument("--jobs", type=int, default=None, help="proxalyze -j concurrency")
    ap.add_argument("--keep", action="store_true", help="leave the sandbox running after the run")
    ap.add_argument("--commit", action=argparse.BooleanOptionalAction, default=True,
                    help="commit the downloaded result (default: on)")
    ap.add_argument("--push", action=argparse.BooleanOptionalAction, default=True,
                    help="git push after commit (default: on)")
    ap.add_argument("--message", default="chore(pool): refresh validated proxies from proxplore+proxalyze")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and build the local tarball; run nothing")
    a = ap.parse_args(argv)
    for d in (a.proxplore, a.proxalyze):
        if not (d / "Cargo.toml").exists():
            ap.error(f"{d} has no Cargo.toml — wrong path?")
    return a


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    return Pipeline(a, remote=None).run()


if __name__ == "__main__":
    sys.exit(main())
