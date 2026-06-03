from __future__ import annotations

import re
from pathlib import Path


REALTIME_ENDPOINT_COST_RISK = "REALTIME_ENDPOINT_COST_RISK"

DANGEROUS_PATTERNS = [
    re.compile(r"\.deploy\("),
    re.compile(r"\bcreate_endpoint\("),
    re.compile(r"\bcreate_endpoint_config\("),
    re.compile(r"\bCreateEndpoint\b"),
    re.compile(r"\bCreateEndpointConfig\b"),
    re.compile(r"\bcreate-endpoint\b"),
    re.compile(r"\bcreate-endpoint-config\b"),
    re.compile(r"\binvoke_endpoint\("),
    re.compile(r"\binvoke_endpoint_async\("),
    re.compile(r"\binvoke-endpoint\b"),
    re.compile(r"\binvoke-endpoint-async\b"),
    re.compile(r"\bcreate_notebook_instance\("),
    re.compile(r"\bCreateNotebookInstance\b"),
    re.compile(r"\bcreate-notebook-instance\b"),
    re.compile(r"\bcreate_app\("),
    re.compile(r"\bCreateApp\b"),
    re.compile(r"\bcreate-app\b"),
]

MUST_FAIL_DIRS = (
    "src",
    "scripts",
    "configs",
    "notebooks",
    "examples",
)

ALLOW_DIRS = (
    "docs",
    "experimental/realtime_endpoint_dangerous",
    "tests/fixtures",
)

TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".ipynb",
    ".sh",
    ".ps1",
    ".ini",
    ".cfg",
    ".toml",
}


def _matches_any_pattern(text: str) -> bool:
    return any(pattern.search(text) for pattern in DANGEROUS_PATTERNS)


def _starts_with(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith(prefix + "/")


def _is_in_any(path: str, prefixes: tuple[str, ...]) -> bool:
    return any(_starts_with(path, prefix) for prefix in prefixes)


def test_no_realtime_endpoint_defaults() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    violations: list[str] = []

    scan_roots = list(MUST_FAIL_DIRS) + list(ALLOW_DIRS)
    for root in scan_roots:
        root_path = repo_root / root
        if not root_path.exists():
            continue
        for path in root_path.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in TEXT_SUFFIXES:
                continue

            rel = path.relative_to(repo_root).as_posix()
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = path.read_text(encoding="utf-8", errors="ignore")

            if not _matches_any_pattern(text):
                continue

            if _is_in_any(rel, MUST_FAIL_DIRS):
                violations.append(
                    f"{rel}: dangerous endpoint pattern found in default path; move to docs/ or experimental/realtime_endpoint_dangerous/"
                )
                continue

            if _is_in_any(rel, ALLOW_DIRS):
                if REALTIME_ENDPOINT_COST_RISK not in text:
                    violations.append(
                        f"{rel}: dangerous endpoint pattern allowed here only with marker {REALTIME_ENDPOINT_COST_RISK}"
                    )
                continue

    assert not violations, "Realtime endpoint safety violations:\n" + "\n".join(violations)
