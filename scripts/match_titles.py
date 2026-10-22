#!/usr/bin/env python
"""Which DAMNIT variables a ``damnit reprocess --match`` selects.

``--match`` is a case-insensitive substring of the variable *title*. This reads
the titles off ``src/amore/context.py`` by AST (nothing is executed, no DAMNIT)
and prints, for each string given, the titles it selects.

    python scripts/match_titles.py 'XPCS g2' 'frames used'
    python scripts/match_titles.py --self-test

``--self-test`` checks every ``reprocess_safe.py -d`` route against the context file:
the titles of each detector select what it should, the AGIPD, JUNGFRAU and XPCS sets
share no variable, and the ``-d xpcs`` titles never select a cell that reads a file the
cluster job is still writing ("frames used", "XPCS score"), which ``reprocess_all.sh
refresh`` does last.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONTEXT = HERE.parent / "src" / "amore" / "context.py"
SHELL = HERE / "reprocess_all.sh"


def titles(path: Path = CONTEXT) -> list[str]:
    """Every variable's DAMNIT title: ``@Variable``'s first argument or ``title=``.

    A variable inside a ``Group`` class is titled ``<group title><sep><title>``,
    where the instance (``agipd = Agipd(title="AGIPD", sep=" ")``) gives both.
    """
    tree = ast.parse(path.read_text())
    groups = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
        ):
            kwargs = {k.arg: ast.literal_eval(k.value) for k in node.value.keywords}
            groups[node.value.func.id] = (
                kwargs.get("title", ""),
                kwargs.get("sep", "/"),
            )

    def declared(function):
        for deco in function.decorator_list:
            if (
                isinstance(deco, ast.Call)
                and getattr(deco.func, "id", "") == "Variable"
            ):
                args = [a for a in deco.args[:1]] + [
                    k.value for k in deco.keywords if k.arg == "title"
                ]
                return ast.literal_eval(args[0])
        return None

    found = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and (title := declared(node)):
            found.append(title)
        elif isinstance(node, ast.ClassDef) and node.name in groups:
            prefix, sep = groups[node.name]
            for member in node.body:
                if isinstance(member, ast.FunctionDef) and (title := declared(member)):
                    found.append(f"{prefix}{sep}{title}")
    return found


def selected(matches: list[str], all_titles: list[str]) -> list[str]:
    return [t for t in all_titles if any(m.lower() in t.lower() for m in matches)]


def self_test() -> int:
    sys.path.insert(0, str(HERE))
    from reprocess_safe import DETECTORS

    every = titles()
    sets = {d: set(selected(list(DETECTORS[d].titles), every)) for d in DETECTORS}
    refresh = selected(["frames used", "XPCS score"], every)
    reads_file = {t for t in every if t.lower().endswith(("frames used", "xpcs score"))}
    xpcs_cells = {t for t in every if t.lower().startswith(("xpcs ", "hits "))}
    checks = {
        "xpcs titles select no file-reading cell": not sets["xpcs"] & reads_file,
        "xpcs titles select every other XPCS and Hits cell": sets["xpcs"]
        == xpcs_cells - reads_file,
        "agipd and xpcs sets share no variable": not sets["agipd"] & sets["xpcs"],
        "agipd titles select its preprocessing, I(q) and SVD": {
            "AGIPD preprocessing",
            "AGIPD I(q)",
            "AGIPD I(q) overview",
            "AGIPD SVD and chi2",
        }
        == sets["agipd"],
        "refresh covers every frames-used cell and the score": {
            t for t in every if t.lower().endswith("frames used")
        }
        | {"XPCS score"}
        == set(refresh),
        "every title of the context file is unique": len(every) == len(set(every)),
    }
    print(f"{len(every)} titles in {CONTEXT}")
    for name in DETECTORS:
        print(f"\nreprocess_safe -d {name}:\n  " + "\n  ".join(sorted(sets[name])))
    print("\nreprocess_all refresh:\n  " + "\n  ".join(refresh) + "\n")
    for name, ok in checks.items():
        print(("ok   " if ok else "FAIL ") + name)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        sys.exit(self_test())
    for match in sys.argv[1:]:
        print(f"--match {match!r}:\n  " + "\n  ".join(selected([match], titles())))
