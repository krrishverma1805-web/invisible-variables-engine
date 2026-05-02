"""CODEOWNERS consistency test (plan §187).

Asserts every directory listed in plan §121 + §154 appears in CODEOWNERS.
Catches accidental rollback of governance.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
CODEOWNERS_PATH = REPO_ROOT / "CODEOWNERS"

REQUIRED_PATHS = [
    "src/ive/llm/",
    "src/ive/models/",
    "src/ive/detection/",
    "src/ive/construction/",
    "alembic/",
    "src/ive/db/",
    "src/ive/api/",
    "streamlit_app/",
    "docs/RESPONSE_CONTRACT.md",
    "src/ive/observability/",
    "ops/",
    "docs/runbooks/",
    ".github/workflows/",
    "Dockerfile",
    "docker-compose.yml",
    "pyproject.toml",
    "src/ive/auth/",
    "CODEOWNERS",
    "docs/branch_protection.md",
]


def test_codeowners_file_exists():
    assert CODEOWNERS_PATH.is_file(), (
        "CODEOWNERS file missing — required by docs/branch_protection.md"
    )


@pytest.mark.parametrize("required_path", REQUIRED_PATHS)
def test_codeowners_lists_required_path(required_path):
    text = CODEOWNERS_PATH.read_text()
    assert required_path in text, (
        f"CODEOWNERS missing required path: {required_path}\n"
        "Update CODEOWNERS per docs/branch_protection.md + plan §121."
    )


def test_codeowners_uses_team_handles_only():
    """Owner tokens should be GitHub handles (start with @), not free-text."""
    bad_lines: list[str] = []
    for raw in CODEOWNERS_PATH.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            bad_lines.append(raw)
            continue
        for owner in parts[1:]:
            if not owner.startswith("@"):
                bad_lines.append(f"non-handle owner {owner!r} in: {raw}")
    assert not bad_lines, (
        "CODEOWNERS owners must be GitHub handles (@org/team or @user):\n"
        + "\n".join(bad_lines)
    )
