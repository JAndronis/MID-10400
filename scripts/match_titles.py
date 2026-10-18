#!/usr/bin/env python
"""Which DAMNIT variables a ``damnit reprocess --match`` selects (not unit-tested).

``--match`` is a case-insensitive substring of the variable *title*. This reads
the titles off ``src/amore/context.py`` by AST (nothing is executed, no DAMNIT)
and prints, for each string given, the titles it selects.

    python scripts/match_titles.py 'XPCS hits' 'frames used'
    python scripts/match_titles.py --self-test

``--self-test`` checks the XPCS reprocess routes against the context file: the
``-d xpcs`` titles of ``reprocess_safe.py`` select exactly the cluster variables
of the hit-train pass and nothing of extra-speckle's pipeline, and the
``XPCS_OLD_MATCH`` array of ``reprocess_all.sh`` selects every other title that
contains "XPCS", the old ones, and no ``XPCS hits`` one.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONTEXT = HERE.parent / "src" / "amore" / "context.py"
SHELL = HERE / "reprocess_all.sh"


def titles(path: Path = CONTEXT) -> list[str]:
    """Every ``@Variable`` title: its first positional argument or ``title=``."""
    found = []
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.FunctionDef):
            continue
        for deco in node.decorator_list:
            if not (
                isinstance(deco, ast.Call)
                and getattr(deco.func, "id", "") == "Variable"
            ):
                continue
            args = [a for a in deco.args[:1]] + [
                k.value for k in deco.keywords if k.arg == "title"
            ]
            found.append(ast.literal_eval(args[0]))
    return found


def selected(matches: list[str], all_titles: list[str]) -> list[str]:
    return [t for t in all_titles if any(m.lower() in t.lower() for m in matches)]


def old_match() -> list[str]:
    body = re.search(r"XPCS_OLD_MATCH=\(([^)]*)\)", SHELL.read_text()).group(1)
    return re.findall(r"'([^']+)'", body)


def self_test() -> int:
    sys.path.insert(0, str(HERE))
    from reprocess_safe import DETECTORS

    every = titles()
    new = selected(list(DETECTORS["xpcs"].titles), every)
    old = selected(old_match(), every)
    hits = [t for t in every if t.lower().startswith("xpcs hits")]
    others = [t for t in every if "xpcs" in t.lower() and t not in hits]
    refresh = selected(["frames used", "XPCS score"], every)
    checks = {
        "xpcs titles select only XPCS hits variables": set(new) <= set(hits),
        "xpcs titles select every hits cluster variable": set(new)
        == set(hits) - {"XPCS hits frames used"},
        "old match selects no XPCS hits variable": not set(old) & set(hits),
        "old match selects every other XPCS title": set(old) == set(others),
        "refresh covers XPCS hits frames used": "XPCS hits frames used" in refresh,
    }
    print(f"{len(every)} titles in {CONTEXT}")
    for name, titles_ in (
        ("reprocess_safe -d xpcs", new),
        ("reprocess_all xpcs (old)", old),
        ("reprocess_all refresh", refresh),
    ):
        print(f"\n{name}:\n  " + "\n  ".join(titles_))
    print()
    for name, ok in checks.items():
        print(("ok   " if ok else "FAIL ") + name)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        sys.exit(self_test())
    for match in sys.argv[1:]:
        print(f"--match {match!r}:\n  " + "\n  ".join(selected([match], titles())))
