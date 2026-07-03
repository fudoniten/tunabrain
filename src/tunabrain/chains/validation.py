"""Helpers for validating LLM-selected values against a provided option set.

LLMs occasionally return choices that were not among the candidates they were
offered (typos like ``spectum`` for ``spectrum``, or entirely invented values
like ``thriller``).  They also frequently return free-form strings in the wrong
format — for example echoing raw Jellyfin genre strings like ``"Action &
Adventure"`` or ``"Documentary"`` instead of the kebab-case ``"action-and-adventure"``
and ``"documentary"`` the rest of the pipeline expects.  These helpers let
chains detect such drift, build feedback the model can act on, and guarantee
that invalid values never leak downstream.
"""

from __future__ import annotations

import re
from collections.abc import Iterable


# Lowercase words joined by single hyphens.  No leading/trailing hyphens, no
# consecutive hyphens, no spaces, capitals, or other special characters.
_KEBAB_CASE_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


def partition_values(
    returned: Iterable[str],
    allowed: Iterable[str],
) -> tuple[list[str], list[str]]:
    """Split ``returned`` into (valid, invalid) relative to ``allowed``.

    Order is preserved and duplicates are removed.  Matching is case-sensitive
    and exact, mirroring how the values are later persisted.
    """
    allowed_set = set(allowed)
    valid: list[str] = []
    invalid: list[str] = []
    seen: set[str] = set()
    for value in returned:
        if value in seen:
            continue
        seen.add(value)
        if value in allowed_set:
            valid.append(value)
        else:
            invalid.append(value)
    return valid, invalid


def format_invalid_feedback(invalid: list[str], allowed: Iterable[str]) -> str:
    """Build a corrective message describing the invalid choices for the LLM."""
    allowed_list = ", ".join(str(value) for value in allowed)
    invalid_list = ", ".join(str(value) for value in invalid)
    return (
        f"The following selected value(s) are not in the allowed set and were "
        f"rejected: {invalid_list}. You MUST choose only from these exact "
        f"options: {allowed_list}. Re-select using only valid options and return "
        f"the JSON dictated by the format instructions."
    )


def is_kebab_case(value: str) -> bool:
    """True if ``value`` is a valid kebab-case string.

    Kebab-case is lowercase words joined by single hyphens: ``"action-and-adventure"``,
    ``"sci-fi"``, ``"documentary"``.  Leading/trailing hyphens, consecutive
    hyphens, spaces, capitals, and other special characters are rejected.
    """
    return bool(_KEBAB_CASE_RE.match(value))


def partition_kebab_case(values: Iterable[str]) -> tuple[list[str], list[str]]:
    """Split ``values`` into (valid, invalid) based on kebab-case format.

    Order is preserved and duplicates are removed.  Use this as a safety net on
    any free-form LLM output that the rest of the pipeline expects to be
    kebab-case (free-form tags, prefixed category values, etc.).
    """
    valid: list[str] = []
    invalid: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        if is_kebab_case(value):
            valid.append(value)
        else:
            invalid.append(value)
    return valid, invalid


def format_kebab_feedback(invalid: list[str]) -> str:
    """Build a corrective message describing the non-kebab-case values."""
    invalid_list = ", ".join(f"'{value}'" for value in invalid)
    return (
        f"The following tag(s) are not in kebab-case format and were rejected: "
        f"{invalid_list}. Tags MUST be lowercase words joined by single hyphens "
        f"(e.g. 'action-and-adventure', 'sci-fi', 'documentary'). Do not use "
        f"spaces, ampersands, capitals, or other special characters. Re-format "
        f"each tag in kebab-case and return the JSON dictated by the format "
        f"instructions."
    )
