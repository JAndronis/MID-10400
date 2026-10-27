"""The DAMNIT context file loads, and its groups are wired.

DAMNIT ``exec``s ``src/amore/context.py`` on the cluster. A syntax error, a
missing import or a ``var#`` naming a variable that no longer exists only
surfaces there, one Slurm job at a time, so the load is exercised here instead.
The analysis columns are ``Group`` classes, which DAMNIT expands to
``<group>.<var>`` names; this module loads the file through DAMNIT's own
``ctxrunner`` so it sees the expanded names.

Nothing in this module runs a variable, which would need real data. It only
checks that DAMNIT could find and order them.
"""

from __future__ import annotations

import ast
import pathlib
import sys

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
pytest.importorskip("extra_data")
damnit_pkg = pytest.importorskip("damnit")

import analysis  # noqa: E402

#: The context beside the ``analysis`` package on the import path, so a run
#: against a copy of ``src/`` checks that copy's context.
CONTEXT_DIR = pathlib.Path(analysis.__file__).resolve().parents[1] / "amore"


@pytest.fixture(scope="module")
def variables() -> dict:
    """Every ``Variable`` in the context file, groups expanded as DAMNIT does."""
    support = pathlib.Path(damnit_pkg.__file__).parent / "ctxsupport"
    # ctxsupport modules import each other flatly, so the directory itself has
    # to be importable, not just the damnit package. The context directory
    # holds analysis_helpers, which the file imports from there.
    added = [str(support), str(CONTEXT_DIR)]
    sys.path[:0] = added
    try:
        import ctxrunner

        # The context is compiled without this module's `from __future__ import
        # annotations`: under PEP 563 an annotation becomes its own source
        # text and "var#agipd.saxs" would arrive with quotes, so no dependency
        # would resolve. ctxrunner compiles with dont_inherit semantics of its
        # own, which is the path DAMNIT takes.
        source = (CONTEXT_DIR / "context.py").read_text()
        return dict(ctxrunner.build_context_from_code(source, "context.py").vars)
    finally:
        for entry in added:
            sys.path.remove(entry)
        sys.modules.pop("ctxrunner", None)


def _tree() -> ast.Module:
    return ast.parse((CONTEXT_DIR / "context.py").read_text())


def _group_functions() -> dict[str, ast.FunctionDef]:
    """``"<group>.<var>"`` -> the method that implements it, by the AST.

    The instance name a module-level ``agipd = Agipd(...)`` binds is the group
    name DAMNIT assigns, and the method name is the variable name.
    """
    tree = _tree()
    instances = {
        node.value.func.id: target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    out = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in instances:
            for member in node.body:
                if isinstance(member, ast.FunctionDef):
                    out[f"{instances[node.name]}.{member.name}"] = member
    return out


def _body(function: ast.FunctionDef) -> list[ast.stmt]:
    body = function.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
    ):
        return body[1:]
    return body


def _call_of(function: ast.FunctionDef) -> ast.Call | None:
    last = function.body[-1]
    if isinstance(last, ast.Return) and isinstance(last.value, ast.Call):
        return last.value
    return None


def test_every_variable_dependency_exists(variables):
    """A `var#` naming a retired variable is a cluster-time failure."""
    for name, variable in variables.items():
        for argument, dependency in variable.arg_dependencies().items():
            assert dependency in variables, (
                f"{name}({argument}=) depends on {dependency!r}, which no "
                "context variable defines"
            )


AGIPD = (
    "agipd.preprocessing",
    "agipd.saxs",
    "agipd.iq_overview",
    "agipd.frames_used",
    "agipd.diagnostics",
)
JUNGFRAU = (
    "jungfrau.preprocessing",
    "jungfrau.jf1",
    "jungfrau.jf2",
    "jungfrau.iq_overview",
    "jungfrau.combined",
    "jungfrau.jf1_frames_used",
    "jungfrau.jf2_frames_used",
)
XPCS_LEGACY = (
    "xpcs.q_rings",
    "xpcs.beamcenter",
    "xpcs.outliers",
    "xpcs.intensity_outliers",
    "xpcs.kbar",
    "xpcs.correction",
    "xpcs.ttcfs",
    "xpcs.figure_g2",
    "xpcs.g2_fit",
    "xpcs.saxs_figure",
    "xpcs.mean_dataset",
    "xpcs.ttcf_per_train",
    "xpcs.saxs_curves",
)
XPCS = (
    "xpcs.preprocessing",
    "xpcs.g2",
    "xpcs.ttcf",
    "xpcs.overview",
    "xpcs.frames_used",
    "xpcs.photon_statistics",
    "xpcs.photon_statistics_overview",
    "xpcs.score",
) + XPCS_LEGACY
HITS = ("hits.summary", "hits.geometry")
CAMERA = ("camera.video", "camera.first_image")
GROUPED = AGIPD + JUNGFRAU + XPCS + HITS + CAMERA


def test_the_context_file_loads_with_every_analysis_column_in_a_group(variables):
    assert set(GROUPED) <= set(variables)
    ungrouped = {name for name in variables if "." not in name}
    assert not {n for n in ungrouped if n.startswith(("agipd", "jungfrau", "xpcs"))}


def test_the_titles_still_match_the_reprocess_patterns(variables):
    """``damnit reprocess --match`` selects variables by a substring of the title."""
    titles = {name: variables[name].title.lower() for name in GROUPED}
    frames = [n for n, t in titles.items() if "frames used" in t]
    assert set(frames) == {
        "agipd.frames_used",
        "jungfrau.jf1_frames_used",
        "jungfrau.jf2_frames_used",
        "xpcs.frames_used",
    }
    assert variables["xpcs.score"].title == "XPCS score"


#: The only variables that read detector proc; every other column reads a file
#: one of them wrote. Each entry is the dependencies the variable must have. The
#: detector passes need none: they read proc alone. The hit-train XPCS pass needs
#: the whole run's ``agipd.saxs`` (the hit rule is a running percentile over the
#: run, and a mesh block spans runs) plus the flux, so its read of proc is a
#: second one, unavoidably, and comes after the first.
PREPROCESSING = {
    "agipd.preprocessing": {},
    "jungfrau.preprocessing": {},
    "xpcs.preprocessing": {
        "saxs": "agipd.saxs",
        "xgm": "xgm_intensity",
        "transmission": "total_transmission",
        "hits": "hits.summary",
    },
}


def test_one_variable_per_detector_kind_reads_proc(variables):
    """The preprocessing variables are cluster jobs; the readers wait on them."""
    for name, dependencies in PREPROCESSING.items():
        assert variables[name].cluster is True
        assert variables[name].arg_dependencies() == dependencies, name
    assert variables["agipd.saxs"].arg_dependencies() == {"prep": "agipd.preprocessing"}
    for detector in ("jf1", "jf2"):
        deps = variables[f"jungfrau.{detector}"].arg_dependencies()
        assert deps == {"prep": "jungfrau.preprocessing"}
    stats = variables["xpcs.photon_statistics"].arg_dependencies()
    assert stats["prep"] == "agipd.preprocessing" and stats["saxs"] == "agipd.saxs"


def test_only_the_preprocessing_variables_call_a_pass():
    """Only the preprocessing variables call ``preprocess``.

    No other body calls a pass's entry point (``agipd_saxs``, ``jungfrau_waxs``)
    directly, which would read proc a second time.
    """
    for name, function in _group_functions().items():
        call = _call_of(function)
        called = (
            call.func.attr
            if call is not None and isinstance(call.func, ast.Attribute)
            else None
        )
        if name in PREPROCESSING:
            assert called == "preprocess", name
        else:
            assert called not in ("preprocess", "agipd_saxs", "jungfrau_waxs"), (
                f"{name} calls {called}"
            )


FRAMES_USED = (
    "agipd.frames_used",
    "jungfrau.jf1_frames_used",
    "jungfrau.jf2_frames_used",
    "xpcs.frames_used",
)


def test_frames_used_never_waits_on_the_pass(variables):
    """No var# on the pass, so asking for the cell cannot relaunch the pass.

    A relaunch while the pass is running would put two writers on one file. The
    cell is not a cluster job either: it reads one status column.
    """
    for name in FRAMES_USED:
        assert variables[name].arg_dependencies() == {}, name
        assert variables[name].cluster is False, name


#: Cluster variables outside the groups that still carry their logic inline.
#: ``droplet_fit`` builds on ``analysis_helpers`` in the context's own directory,
#: which is not part of the package.
INLINE_CLUSTER_VARIABLES = frozenset({"droplet_fit"})


def _module_cluster_functions() -> dict[str, ast.FunctionDef]:
    """Module-level functions decorated as ``Variable(..., cluster=True)``."""
    out = {}
    for node in _tree().body:
        if not isinstance(node, ast.FunctionDef):
            continue
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call) and any(
                k.arg == "cluster"
                and isinstance(k.value, ast.Constant)
                and k.value.value is True
                for k in decorator.keywords
            ):
                out[node.name] = node
    return out


def test_the_context_file_holds_no_heavy_integration_logic():
    """DAMNIT execs this file, so nothing defined here can reach a worker.

    The integration functions must stay under ``analysis``; a helper defined in
    the context file would fail to pickle only once the pool spawned. Each
    wrapper is a docstring, an import and a call; anything with statements
    beyond that has logic in it that belongs in the package instead. That holds
    for every grouped variable and every cluster job outside the groups.
    """
    functions = _group_functions()
    assert set(GROUPED) <= set(functions)
    cluster = _module_cluster_functions()
    assert INLINE_CLUSTER_VARIABLES <= set(cluster)
    for name, function in functions.items() | {
        (n, f) for n, f in cluster.items() if n not in INLINE_CLUSTER_VARIABLES
    }:
        kinds = [type(node).__name__ for node in _body(function)]
        assert kinds == ["ImportFrom", "Return"], (
            f"{name} does something other than import and call: {kinds}"
        )


def test_no_call_in_the_context_file_passes_overwrite():
    """DAMNIT reruns a pass whenever a dependant is requested.

    With ``overwrite=True`` every such rerun would truncate the file. A reprocess
    moves the output aside instead (``scripts/reprocess_safe.py``).
    """
    offenders = [
        f"line {node.lineno}"
        for node in ast.walk(_tree())
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg == "overwrite"
        or (
            keyword.arg is None
            and isinstance(keyword.value, ast.Dict)
            and any(
                isinstance(key, ast.Constant) and key.value == "overwrite"
                for key in keyword.value.keys
            )
        )
    ]
    assert not offenders, f"overwrite passed at {offenders}"


#: The variables that call a pass or its reader with DAMNIT's config.
INTEGRATION_WRAPPERS = (
    "agipd.preprocessing",
    "jungfrau.preprocessing",
    "agipd.saxs",
    "jungfrau.jf1",
    "jungfrau.jf2",
    "xpcs.preprocessing",
    "xpcs.g2",
    "xpcs.ttcf",
)


def test_the_integration_wrappers_pass_no_config():
    """The config DAMNIT runs with lives in ``analysis.*.damnit``, in one place.

    A keyword here, such as ``npt=200``, is a config that scripts judging
    DAMNIT's files (``reprocess_safe``, ``is_complete``) cannot see.
    """
    functions = _group_functions()
    for name in INTEGRATION_WRAPPERS:
        call = _call_of(functions[name])
        assert call is not None, name
        assert not call.keywords, f"{name} passes {[k.arg for k in call.keywords]}"


def test_the_hit_rule_comes_before_what_works_on_the_hits(variables):
    """A clean run goes preprocessing, SAXS/WAXS columns, the hit rule, then the
    photon statistics and the hit-train pass on the hits.

    The dependency on ``hits.summary`` is what makes DAMNIT run them in that order.
    """
    assert variables["hits.summary"].arg_dependencies()["saxs"] == "agipd.saxs"
    assert variables["hits.geometry"].arg_dependencies()["summary"] == "hits.summary"
    for name in ("xpcs.preprocessing", "xpcs.photon_statistics"):
        assert variables[name].arg_dependencies()["hits"] == "hits.summary", name
