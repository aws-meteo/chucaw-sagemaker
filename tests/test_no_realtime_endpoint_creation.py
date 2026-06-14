from __future__ import annotations

import re
from pathlib import Path


FORBIDDEN_PATTERNS = [
    re.compile(r"\bcreate_endpoint\("),
    re.compile(r"\bcreate_endpoint_config\("),
    re.compile(r"\.deploy\("),
    re.compile(r"\bPredictor\("),
    re.compile(r"\bEndpointConfig\b"),
]


def _is_exempt(path: Path) -> bool:
    lowered = str(path).lower()
    return "experimental" in lowered


def test_no_realtime_endpoint_creation() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    targets = list((repo_root / "scripts").rglob("*.py")) + list(
        (repo_root / "src").rglob("*.py")
    )
    violations: list[str] = []

    for path in targets:
        if _is_exempt(path):
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in FORBIDDEN_PATTERNS:
            if pattern.search(text):
                violations.append(f"{path.relative_to(repo_root)} -> {pattern.pattern}")
                break

    assert not violations, "Realtime endpoint creation detected:\n" + "\n".join(violations)
