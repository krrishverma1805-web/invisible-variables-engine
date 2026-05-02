"""Output validators for LLM-generated explanations.

Defense-in-depth against three failure modes:

1. **Hallucinated numbers** — ``ground_check`` ensures every numeric value
   in the output is either an input fact or a legitimate pairwise derivation
   (ratio, product, sum, difference, complement) within ±2% tolerance.
2. **Banned phrases** — causal verbs and overconfident hedging trigger
   either hard-block or warn-only failure depending on severity.
3. **Prompt-injection echo** — outputs containing ``<|``, ``|>``, or
   role-prompt markers are rejected.

Sanitization on the input side strips injection markers from user-derived
free-text fields before they enter the prompt.

Plan reference: §A1 (validators table), §32 (numeric grounding), §104
(derived numbers), §105 (causal verbs), §138 (generative allowance).
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any

from ive.llm.types import ValidationReport

# ── Banned-phrase registry ──────────────────────────────────────────────────
# Severity ``hard_block`` always fails the validator; ``warn_only`` only fails
# when not adjacent to a hedge marker (per §139).
BANNED_PHRASES: list[tuple[str, str]] = [
    # Hard-block: definitive / overconfident claims
    ("definitely", "hard_block"),
    ("guaranteed", "hard_block"),
    ("guarantees", "hard_block"),
    ("proves", "hard_block"),
    ("100% accurate", "hard_block"),
    ("never fails", "hard_block"),
    ("always", "hard_block"),
    # Causal verbs (when allow_causal=False)
    ("causes", "causal"),
    ("caused", "causal"),
    ("causing", "causal"),
    ("leads to", "causal"),
    ("drives", "causal"),
    ("driven by", "causal"),
    ("makes", "causal"),
    ("produces", "causal"),
    ("generates", "causal"),
    ("creates", "causal"),
    ("results in", "causal"),
    ("is responsible for", "causal"),
    ("accounts for", "causal"),
    ("explains", "causal"),
    ("is the reason", "causal"),
    ("triggers", "causal"),
    ("triggered by", "causal"),
    ("induces", "causal"),
    ("brings about", "causal"),
    ("gives rise to", "causal"),
    # Warn-only: hedged correlation connectives that need a nearby hedge marker
    ("due to", "warn_only"),
    ("owing to", "warn_only"),
]

HEDGE_MARKERS: tuple[str, ...] = (
    "possibly",
    "may",
    "perhaps",
    "appears to",
    "appears",
    "seems",
    "is associated with",
    "associated with",
    "correlated with",
    "tends to",
    "suggests",
    "suggested",
    "likely",
    "probably",
)

INJECTION_MARKERS: tuple[str, ...] = (
    "<|",
    "|>",
    "system:",
    "assistant:",
    "user:",
    "ignore previous",
    "ignore above",
    "ignore the above",
    "disregard",
)

NUMBER_PATTERN = re.compile(
    r"(?<![\w.])"          # not preceded by word char or dot
    r"-?\d+(?:,\d{3})*"    # integer part with optional thousands separators
    r"(?:\.\d+)?"          # optional decimal
    r"(?:[eE][-+]?\d+)?"   # optional scientific
    # Trailing-context: either match "%" followed by not-word (period OK
    # because the percent disambiguates), or match no % but require
    # not-word-or-dot so we don't half-eat decimals.
    r"(?:%(?![\w])|(?![\w.]))"
)

# Numbers we always accept without grounding (cardinal small ints, common
# years, percentages used as bucket labels).
TRIVIAL_NUMBERS: set[Decimal] = {Decimal(n) for n in range(0, 11)}


_BIDI_AND_INVISIBLE_RE = re.compile(
    "["
    "​-‏"  # zero-width chars + bidi marks (LRM/RLM/LRE/RLE/PDF)
    "‪-‮"  # bidi embedding/override controls (incl. RLO/LRO)
    "⁠-⁤"  # word joiner + invisible separators
    "⁦-⁩"  # bidi isolates
    "﻿"          # BOM
    "]"
)
_HTML_LIKE_TAG_RE = re.compile(r"</?[a-zA-Z][a-zA-Z0-9_:-]*\s*/?>")


def sanitize_user_input(text: str, *, max_length: int = 200) -> str:
    """Strip injection markers and clamp length on user-derived free text.

    Applied to candidate names, dataset names, segment descriptions, etc.
    before they are interpolated into a prompt. Defenses (in order):

        * Strip role-prompt markers (``<|``, ``|>``, ``system:``, ...).
        * Strip HTML-like tags (``</user>``, ``<system>``) — neutralizes
          tag-confusion attacks that don't carry literal injection markers.
        * Strip bidi / zero-width / invisible Unicode controls so RTL
          override and ZWSP attacks can't smuggle hidden content.
        * Strip C0/C1 control bytes and code-fence backticks.
        * Collapse whitespace and clamp to ``max_length``.
    """
    if not isinstance(text, str):
        return ""
    cleaned = text
    for marker in INJECTION_MARKERS:
        cleaned = re.sub(re.escape(marker), "", cleaned, flags=re.IGNORECASE)
    cleaned = _HTML_LIKE_TAG_RE.sub("", cleaned)
    cleaned = _BIDI_AND_INVISIBLE_RE.sub("", cleaned)
    # Strip control chars and code-fence markers
    cleaned = cleaned.replace("`", "").replace("\r", " ").replace("\n", " ")
    cleaned = re.sub(r"[\x00-\x1f\x7f]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:max_length]


def _canonicalize_number(token: str) -> Decimal | None:
    """Convert a matched number token to a canonical ``Decimal``.

    Handles thousands separators, percent signs (kept as-is so ``42%`` and
    ``0.42`` are distinguishable until the grounding check normalizes), and
    scientific notation.
    """
    raw = token.replace(",", "").strip()
    is_percent = raw.endswith("%")
    if is_percent:
        raw = raw[:-1]
    try:
        value = Decimal(raw)
    except (InvalidOperation, ValueError):
        return None
    if is_percent:
        # Convert "42%" -> Decimal("0.42")
        value = value / Decimal(100)
    return value


def extract_numbers(text: str) -> set[Decimal]:
    """Extract every numeric token from ``text`` as canonical ``Decimal`` values."""
    out: set[Decimal] = set()
    for match in NUMBER_PATTERN.finditer(text):
        value = _canonicalize_number(match.group(0))
        if value is not None:
            out.add(value)
    return out


def _within_tolerance(value: Decimal, target: Decimal, *, pct: float = 2.0) -> bool:
    """Return True when ``value`` is within ±pct% of ``target`` (or equal when target=0)."""
    if target == 0:
        return value == 0
    delta = abs(value - target)
    tolerance = abs(target) * Decimal(pct) / Decimal(100)
    return delta <= tolerance


def _build_allowed_set(facts: dict[str, Any]) -> set[Decimal]:
    """Compute the set of fact-derived numbers an output may reference.

    Includes the raw fact values plus every pairwise derivation (sum,
    difference, product, ratio, complement) per §138.  Tolerance applies
    only to *this* set; the trivial-number set (small ints used as counts)
    is matched exactly to avoid spurious overlap with hallucinated values.
    """
    raw: list[Decimal] = []
    for value in facts.values():
        if isinstance(value, (int, float)):
            try:
                raw.append(Decimal(str(value)))
            except (InvalidOperation, ValueError):
                continue
        elif isinstance(value, Decimal):
            raw.append(value)
    allowed: set[Decimal] = set(raw)
    for i, a in enumerate(raw):
        # Self-complement
        allowed.add(Decimal(1) - a)
        # Distinct pairs only — skip self-pair derivations like a/a=1, a*a, a-a=0
        # which produce trivial values and create false-positive matches.
        for b in raw[i + 1 :]:
            allowed.add(a + b)
            allowed.add(a - b)
            allowed.add(b - a)
            allowed.add(a * b)
            if b != 0:
                allowed.add(a / b)
            if a != 0:
                allowed.add(b / a)
            allowed.add(Decimal(1) - b)
    return allowed


def ground_check(output: str, facts: dict[str, Any]) -> ValidationReport:
    """Reject outputs that introduce numbers not derivable from input facts.

    Tolerance band: ±2% on fact-derived values; trivial small integers
    (0–10) match exactly per §138.
    """
    allowed = _build_allowed_set(facts)
    output_numbers = extract_numbers(output)
    for value in output_numbers:
        # Trivial small ints match exactly (no tolerance), so e.g. "0.99"
        # cannot satisfy a "1" allowance.
        if value in TRIVIAL_NUMBERS:
            continue
        if any(_within_tolerance(value, target) for target in allowed):
            continue
        return ValidationReport(
            passed=False,
            failures=[f"output number {value} is not derivable from input facts"],
            rule="ground_check",
        )
    return ValidationReport(passed=True)


def banned_phrase_filter(output: str, *, allow_causal: bool = False) -> ValidationReport:
    """Reject outputs containing banned phrases per §105 / §139."""
    lower = output.lower()
    for phrase, severity in BANNED_PHRASES:
        if phrase not in lower:
            continue
        if severity == "hard_block":
            return ValidationReport(
                passed=False,
                failures=[f"banned phrase: {phrase!r}"],
                rule="banned_phrase_hard",
            )
        if severity == "causal" and not allow_causal:
            return ValidationReport(
                passed=False,
                failures=[f"causal verb: {phrase!r}"],
                rule="causal_verb_filter",
            )
        if severity == "warn_only":
            # Only fail if there's no hedge marker within ~10 tokens.
            idx = lower.find(phrase)
            window_start = max(0, idx - 80)
            window_end = min(len(lower), idx + len(phrase) + 80)
            window = lower[window_start:window_end]
            if not any(marker in window for marker in HEDGE_MARKERS):
                return ValidationReport(
                    passed=False,
                    failures=[f"unhedged correlation phrase: {phrase!r}"],
                    rule="unhedged_correlation",
                )
    return ValidationReport(passed=True)


def injection_echo_filter(output: str) -> ValidationReport:
    """Reject outputs echoing prompt-injection markers."""
    lower = output.lower()
    for marker in INJECTION_MARKERS:
        if marker in lower:
            return ValidationReport(
                passed=False,
                failures=[f"prompt-injection marker echoed: {marker!r}"],
                rule="injection_echo",
            )
    return ValidationReport(passed=True)


def length_sanity(output: str, *, min_chars: int = 20, max_chars: int = 1200) -> ValidationReport:
    """Reject outputs outside reasonable length bounds."""
    n = len(output)
    if n < min_chars:
        return ValidationReport(
            passed=False,
            failures=[f"output too short: {n} chars < {min_chars}"],
            rule="length_sanity",
        )
    if n > max_chars:
        return ValidationReport(
            passed=False,
            failures=[f"output too long: {n} chars > {max_chars}"],
            rule="length_sanity",
        )
    return ValidationReport(passed=True)


def composite_validate(
    output: str,
    facts: dict[str, Any],
    *,
    allow_causal: bool = False,
    min_chars: int = 20,
    max_chars: int = 1200,
) -> ValidationReport:
    """Run the full validator chain in order.

    Returns the first failing report; passes only when all rules pass.
    """
    for report in (
        length_sanity(output, min_chars=min_chars, max_chars=max_chars),
        injection_echo_filter(output),
        banned_phrase_filter(output, allow_causal=allow_causal),
        ground_check(output, facts),
    ):
        if not report.passed:
            return report
    return ValidationReport(passed=True)
