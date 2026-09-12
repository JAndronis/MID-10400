"""The DAMNIT context file loads, and the SAXS variables are wired (P5).

DAMNIT ``exec``s ``src/amore/context.py`` on the cluster. A syntax error, a
missing import or a ``var#`` naming a variable that no longer exists only
surfaces there, one Slurm job at a time, so the load is exercised here instead.

Nothing in this module runs a variable — that needs real data — it only checks
that DAMNIT could find and order them.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
pytest.importorskip("extra_data")
pytest.importorskip("extra_speckle")
damnit_pkg = pytest.importorskip("damnit")

CONTEXT_DIR = pathlib.Path(__file__).resolve().parents[1] / "src" / "amore"


@pytest.fixture(scope="module")
def variables() -> dict:
    """Every ``Variable`` in the context file, loaded the way DAMNIT loads it."""
    support = pathlib.Path(damnit_pkg.__file__).parent / "ctxsupport"
    # ctxsupport modules import each other flatly, so the directory itself has
    # to be importable, not just the damnit package.
    added = [str(support), str(CONTEXT_DIR)]
    sys.path[:0] = added
    try:
        from damnit_ctx import Variable

        source = (CONTEXT_DIR / "context.py").read_text()
        namespace: dict = {"__name__": "context"}
        # dont_inherit, because compile() otherwise adds this module's
        # `from __future__ import annotations` to the context file. Under PEP
        # 563 an annotation becomes its own source text, so DAMNIT's
        # "var#agipd_saxs" arrives as "'var#agipd_saxs'" — quotes
        # included — and no dependency resolves. DAMNIT runs the context file
        # in its own process, so a fresh compile is the faithful one.
        exec(compile(source, "context.py", "exec", dont_inherit=True), namespace)
        return {
            name: value
            for name, value in namespace.items()
            if isinstance(value, Variable)
        }
    finally:
        for entry in added:
            sys.path.remove(entry)


def test_the_context_file_loads(variables):
    assert len(variables) > 1


def test_the_agipd_saxs_variable_is_wired(variables):
    variable = variables["agipd_saxs"]
    assert variable.title == "AGIPD I(q)"
    # cluster=True: one Slurm job per run on a whole node, which is what the
    # 36-worker pool needs.
    assert variable.cluster is True


def test_the_overview_depends_on_the_agipd_saxs(variables):
    assert variables["agipd_iq_overview"].arg_dependencies() == {"grid": "agipd_saxs"}


def test_every_variable_dependency_exists(variables):
    """A `var#` naming a retired variable is a cluster-time failure."""
    for name, variable in variables.items():
        for argument, dependency in variable.arg_dependencies().items():
            assert dependency in variables, (
                f"{name}({argument}=) depends on {dependency!r}, which no "
                "context variable defines"
            )


def test_the_saxs_variables_keep_their_names(variables):
    """The names and columns are deliberately kept; the implementation changed.

    `agipd_saxs` is superseded in place rather than renamed, so the DAMNIT
    column and anything downstream keep working. What the column *holds* moved
    from Å⁻¹ I0-divided to nm⁻¹ undivided, which is why the old pipeline must
    not still be reachable from this file.
    """
    assert {"agipd_saxs", "agipd_iq_overview"} <= set(variables)

    # Comments are stripped, not searched: the block above these variables
    # names the old pipeline in order to explain what replaced it.
    source = (CONTEXT_DIR / "context.py").read_text()
    code = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    )
    assert "integrate_run" not in code
    assert "geometry_from_encoders" not in code


def test_the_context_file_holds_no_heavy_saxs_logic():
    """DAMNIT execs this file, so nothing defined here can reach a worker.

    The integration functions must stay in ``analysis.saxs``; a helper defined
    in the context file would fail to pickle only once the pool spawned.
    """
    source = (CONTEXT_DIR / "context.py").read_text()
    imports = [
        line for line in source.splitlines() if line.startswith(("import ", "from "))
    ]
    assert not any("integrate_run" in line for line in imports)

    # Each body is an import plus a call; anything longer has logic in it that
    # belongs in the package instead.
    body = source[source.index("def agipd_saxs") :]
    indented = [line for line in body.splitlines() if line.startswith("    ")]
    assert len(indented) < 20, "the SAXS variables have grown bodies"
