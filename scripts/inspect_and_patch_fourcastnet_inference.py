#!/usr/bin/env python3
"""Inspect (and optionally CPU-patch) a FourCastNet SageMaker model.tar.gz.

Why this exists
---------------
The FourCastNet serving handler (``code/inference.py`` with ``backbone.ckpt``)
does NOT live in this repository. It only exists inside the model artifact in
S3:

    s3://.../sagemaker/fourcastnet/fcn-v1/model/model.tar.gz

The registered model ``sbnai-fourcastnet-fcn-v1`` uses a GPU image, but the
account only has CPU Batch Transform quota. Before we can register a CPU model
that reuses the same artifact, we MUST confirm the handler does not hardcode
CUDA. This tool answers that question at the code level once the artifact has
been downloaded locally (no AWS calls are made here).

Usage
-----
Inspect only (safe, read-only)::

    python scripts/inspect_and_patch_fourcastnet_inference.py \
        --model-tar artifacts/fcn-v1-model.tar.gz

Emit a CPU-patched copy (original is never modified)::

    python scripts/inspect_and_patch_fourcastnet_inference.py \
        --model-tar artifacts/fcn-v1-model.tar.gz \
        --emit-patched artifacts/fcn-v1-cpu-model.tar.gz

Exit codes
----------
0  handler is already CPU-safe (or a patched copy was written successfully)
2  handler hardcodes CUDA and no --emit-patched was requested (action needed)
3  no inference.py found inside the archive
1  usage / IO error
"""
from __future__ import annotations

import argparse
import io
import re
import sys
import tarfile
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# --- Pattern catalogue -------------------------------------------------------
# Hard blockers: these force CUDA and will crash on a CPU-only instance.
HARD_CUDA_PATTERNS = [
    (re.compile(r"""torch\.device\(\s*['"]cuda[^'"]*['"]\s*\)"""), "hardcoded torch.device('cuda')"),
    (re.compile(r"\.cuda\s*\("), "explicit .cuda() call"),
    (re.compile(r"""\.to\(\s*['"]cuda[^'"]*['"]"""), "tensor/module .to('cuda')"),
    (re.compile(r"set_device\s*\("), "torch.cuda.set_device(...)"),
]

# Soft signals: these are CPU-safe fallbacks. Their presence is reassuring.
SOFT_SAFE_PATTERNS = [
    (re.compile(r"torch\.cuda\.is_available\s*\("), "cuda.is_available() fallback (CPU-safe)"),
    (re.compile(r"""map_location\s*=\s*['"]cpu['"]"""), "map_location='cpu' (CPU-safe)"),
    (re.compile(r"map_location\s*=\s*torch\.device\(\s*['\"]cpu['\"]"), "map_location=torch.device('cpu')"),
]

# torch.load without an explicit map_location defaults to the saved device and
# can fail when a CUDA checkpoint is loaded on CPU. We flag these for review.
TORCH_LOAD = re.compile(r"torch\.load\s*\(")
HAS_MAP_LOCATION = re.compile(r"map_location")


@dataclass
class Finding:
    lineno: int
    line: str
    label: str
    severity: str  # "BLOCKER" | "WARN" | "SAFE"


@dataclass
class ScanResult:
    findings: List[Finding] = field(default_factory=list)

    @property
    def blockers(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == "BLOCKER"]

    @property
    def warnings(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == "WARN"]

    @property
    def safe_signals(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == "SAFE"]

    @property
    def cpu_safe(self) -> bool:
        """CPU-safe means: no hard CUDA blockers. Warnings are non-fatal."""
        return len(self.blockers) == 0


def scan_source(text: str) -> ScanResult:
    """Statically scan handler source for CUDA hardcoding. Pure / testable."""
    result = ScanResult()
    for i, raw in enumerate(text.splitlines(), start=1):
        # Ignore comment-only lines to reduce false positives.
        code_part = raw.split("#", 1)[0]
        if not code_part.strip():
            continue
        for pat, label in HARD_CUDA_PATTERNS:
            if pat.search(code_part):
                result.findings.append(Finding(i, raw.strip(), label, "BLOCKER"))
        for pat, label in SOFT_SAFE_PATTERNS:
            if pat.search(code_part):
                result.findings.append(Finding(i, raw.strip(), label, "SAFE"))
        if TORCH_LOAD.search(code_part) and not HAS_MAP_LOCATION.search(code_part):
            result.findings.append(
                Finding(i, raw.strip(), "torch.load(...) without map_location='cpu'", "WARN")
            )
    return result


def patch_source(text: str) -> Tuple[str, List[str]]:
    """Apply conservative CPU-safe rewrites. Returns (new_text, change_log).

    Transformations are intentionally narrow and reported line-by-line so a
    human can review the diff before re-uploading the artifact:
      - torch.device('cuda...') -> torch.device('cpu')
      - .to('cuda...')          -> .to('cpu')
      - .cuda()                 -> .cpu()
      - torch.load(...) lacking map_location -> inject map_location='cpu'
    """
    changes: List[str] = []
    out_lines: List[str] = []
    for i, raw in enumerate(text.splitlines(), start=1):
        original = raw
        line = raw

        line, n = re.subn(r"""torch\.device\(\s*['"]cuda[^'"]*['"]\s*\)""", 'torch.device("cpu")', line)
        if n:
            changes.append(f"L{i}: torch.device('cuda') -> torch.device('cpu')")

        line, n = re.subn(r"""\.to\(\s*['"]cuda[^'"]*['"]""", '.to("cpu"', line)
        if n:
            changes.append(f"L{i}: .to('cuda') -> .to('cpu')")

        line, n = re.subn(r"\.cuda\s*\(\s*\)", ".cpu()", line)
        if n:
            changes.append(f"L{i}: .cuda() -> .cpu()")

        # Inject map_location='cpu' into torch.load(...) calls that lack it.
        code_part = line.split("#", 1)[0]
        if TORCH_LOAD.search(code_part) and not HAS_MAP_LOCATION.search(code_part):
            line, n = re.subn(r"torch\.load\s*\(", 'torch.load(', line)  # no-op normalize
            # Insert map_location as first kwarg after the opening paren.
            line, n = re.subn(
                r"(torch\.load\s*\(\s*)",
                r'\1',  # placeholder; real insertion below
                line,
            )
            # Robust insertion: add ', map_location="cpu"' before the matching
            # close paren of torch.load. We use a simple heuristic that works
            # for single-call lines (the common case in inference handlers).
            idx = line.find("torch.load(")
            if idx != -1:
                close = line.rfind(")")
                if close > idx:
                    inner = line[idx + len("torch.load(") : close].rstrip()
                    sep = "" if inner.endswith(",") or inner == "" else ", "
                    line = line[: idx + len("torch.load(")] + inner + sep + 'map_location="cpu"' + line[close:]
                    changes.append(f"L{i}: torch.load(...) injected map_location='cpu'")

        out_lines.append(line)
        _ = original  # kept for readability/debugging

    new_text = "\n".join(out_lines)
    if text.endswith("\n"):
        new_text += "\n"
    return new_text, changes


def _find_inference_member(tar: tarfile.TarFile) -> Optional[tarfile.TarInfo]:
    candidates = [m for m in tar.getmembers() if m.isfile() and m.name.rstrip("/").endswith("inference.py")]
    if not candidates:
        return None
    # Prefer code/inference.py (SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/model/code).
    for m in candidates:
        if "code/" in m.name:
            return m
    return candidates[0]


def emit_patched_tar(src_tar: str, dst_tar: str, member_name: str, patched_bytes: bytes) -> None:
    """Copy src_tar to dst_tar, replacing one member's bytes. Original untouched."""
    with tarfile.open(src_tar, "r:gz") as src, tarfile.open(dst_tar, "w:gz") as dst:
        for m in src.getmembers():
            if m.name == member_name:
                info = tarfile.TarInfo(name=m.name)
                info.size = len(patched_bytes)
                info.mode = m.mode
                info.mtime = m.mtime
                info.uid, info.gid = m.uid, m.gid
                info.uname, info.gname = m.uname, m.gname
                dst.addfile(info, io.BytesIO(patched_bytes))
            else:
                extracted = src.extractfile(m) if m.isfile() else None
                dst.addfile(m, extracted)


def _print_report(member_name: str, scan: ScanResult) -> None:
    print(f"=== inference handler: {member_name} ===")
    if scan.blockers:
        print(f"[BLOCKER] {len(scan.blockers)} hardcoded-CUDA finding(s):")
        for f in scan.blockers:
            print(f"  L{f.lineno}: {f.label}")
            print(f"       {f.line}")
    if scan.warnings:
        print(f"[WARN] {len(scan.warnings)} item(s) to verify:")
        for f in scan.warnings:
            print(f"  L{f.lineno}: {f.label}")
            print(f"       {f.line}")
    if scan.safe_signals:
        print(f"[SAFE] {len(scan.safe_signals)} CPU-safe signal(s):")
        for f in scan.safe_signals:
            print(f"  L{f.lineno}: {f.label}")
    print("---------------------------------------------")
    print(f"CPU-SAFE (no hard CUDA blockers): {scan.cpu_safe}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-tar", required=True, help="Local path to downloaded model.tar.gz")
    parser.add_argument("--emit-patched", help="Write a CPU-patched copy to this path (original untouched)")
    args = parser.parse_args()

    try:
        with tarfile.open(args.model_tar, "r:gz") as tar:
            print("Archive entries:")
            for name in tar.getnames():
                print(f"  {name}")
            member = _find_inference_member(tar)
            if member is None:
                print("ERROR: no inference.py found inside the archive.", file=sys.stderr)
                return 3
            fh = tar.extractfile(member)
            if fh is None:
                print(f"ERROR: could not read {member.name}.", file=sys.stderr)
                return 1
            source = fh.read().decode("utf-8", errors="replace")
    except (tarfile.TarError, OSError) as exc:
        print(f"ERROR: failed to open {args.model_tar}: {exc}", file=sys.stderr)
        return 1

    scan = scan_source(source)
    _print_report(member.name, scan)

    if not args.emit_patched:
        if not scan.cpu_safe:
            print(
                "\nACTION: handler hardcodes CUDA. Re-run with --emit-patched <out.tar.gz> "
                "to generate a CPU-safe copy, then review the change log before re-uploading."
            )
            return 2
        print("\nOK: handler is CPU-safe. You may register a CPU model on the SAME model.tar.gz.")
        return 0

    patched, changes = patch_source(source)
    emit_patched_tar(args.model_tar, args.emit_patched, member.name, patched.encode("utf-8"))
    print(f"\nWrote CPU-patched artifact: {args.emit_patched}")
    if changes:
        print("Change log (REVIEW BEFORE UPLOAD):")
        for c in changes:
            print(f"  - {c}")
    else:
        print("No textual changes were necessary (handler was already CPU-safe).")
    # Re-scan the patched source to confirm blockers are gone.
    rescan = scan_source(patched)
    print(f"Post-patch CPU-SAFE: {rescan.cpu_safe}")
    return 0 if rescan.cpu_safe else 2


if __name__ == "__main__":
    raise SystemExit(main())
