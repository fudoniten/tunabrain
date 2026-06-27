"""Helpers for validating LLM-selected values against a provided option set.

LLMs occasionally return choices that were not among the candidates they were
offered (typos like ``spectum`` for ``spectrum``, or entirely invented values
like ``thriller``).  These helpers let chains detect such drift, build feedback
the model can act on, and guarantee that invalid values never leak downstream.
"""

from __future__ import annotations

from collections.abc import Iterable


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
