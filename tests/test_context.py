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

    source = (CONTEXT_DIR / "context.py").read_text()
    assert "integrate_run" not in source
    assert "geometry_from_encoders" not in source


INTEGRATION_VARIABLES = (
    "agipd_preprocessing",
    "agipd_saxs",
    "agipd_iq_overview",
    "jungfrau_preprocessing",
    "jungfrau_waxs_jf1",
    "jungfrau_waxs_jf2",
    "jungfrau_waxs_overview",
    "jungfrau_waxs_combined",
)

DIAGNOSTIC_VARIABLES = (
    "photon_statistics",
    "photon_statistics_overview",
    "diagnostics_overview",
    "xpcs_score",
)

XPCS_HITS_VARIABLES = (
    "xpcs_hits_preprocessing",
    "xpcs_hits_g2",
    "xpcs_hits_ttcf",
    "xpcs_hits_overview",
)

#: The only variables that read detector proc; every other column reads a file
#: one of them wrote (user, 2026-10-14: one read, the table says who computes).
#: Each entry is the call its body makes and the ``var#`` dependencies it must
#: have. The detector passes need none: they read proc alone. The hit-train XPCS
#: pass needs the whole run's ``agipd_saxs`` (the hit rule is a running
#: percentile over the run, and a mesh block spans runs) plus the flux, so its
#: read of proc is a second one, unavoidably, and comes after the first.
PREPROCESSING = {
    "agipd_preprocessing": ("preprocess", {}),
    "jungfrau_preprocessing": ("preprocess", {}),
    "xpcs_hits_preprocessing": (
        "preprocess",
        {
            "saxs": "agipd_saxs",
            "xgm": "xgm_intensity",
            "transmission": "total_transmission",
        },
    ),
}


def test_one_variable_per_detector_kind_reads_proc(variables):
    """The preprocessing variables are cluster jobs; the readers wait on them."""
    for name, (_, dependencies) in PREPROCESSING.items():
        assert variables[name].cluster is True
        assert variables[name].arg_dependencies() == dependencies, name
    assert variables["agipd_saxs"].arg_dependencies() == {"prep": "agipd_preprocessing"}
    for detector in ("jf1", "jf2"):
        deps = variables[f"jungfrau_waxs_{detector}"].arg_dependencies()
        assert deps == {"prep": "jungfrau_preprocessing"}
    stats = variables["photon_statistics"].arg_dependencies()
    assert stats["prep"] == "agipd_preprocessing" and stats["saxs"] == "agipd_saxs"


def test_only_the_preprocessing_variables_call_a_pass():
    """``preprocess`` appears in their bodies and nowhere else, and no body calls
    the computing entry points of the old layout."""
    import ast

    for node in _context_tree().body:
        if not isinstance(node, ast.FunctionDef):
            continue
        called = node.body[-1].value.func.attr if isinstance(
            node.body[-1], ast.Return
        ) and isinstance(node.body[-1].value, ast.Call) and isinstance(
            node.body[-1].value.func, ast.Attribute
        ) else None  # fmt: skip
        if node.name in PREPROCESSING:
            assert called == PREPROCESSING[node.name][0], node.name
        else:
            assert called not in ("preprocess", "agipd_saxs", "jungfrau_waxs"), (
                f"{node.name} calls {called}"
            )


FRAMES_USED_VARIABLES = (
    "agipd_frames_used",
    "jf1_frames_used",
    "jf2_frames_used",
    "xpcs_hits_frames_used",
)


@pytest.mark.parametrize("name", FRAMES_USED_VARIABLES)
def test_frames_used_never_waits_on_the_pass(variables, name):
    """No var# on the pass, so asking for the cell cannot relaunch it (pitfall 26).

    Not a cluster job either: it reads one status column. The titles share
    "frames used", which is what ``damnit reprocess --match`` compares against.
    """
    variable = variables[name]
    assert variable.arg_dependencies() == {}
    assert variable.cluster is False
    assert "frames used" in variable.title.lower()


def test_the_droplet_video_is_a_cluster_job_and_the_image_is_not(variables):
    """The video reads every image of the run; the first image reads one."""
    video, image = variables["droplet_video"], variables["droplet_first_image"]
    assert video.cluster is True and image.cluster is False
    assert video.data.value == image.data.value == "raw"  # the camera is raw


CAMERA_VARIABLES = ("droplet_video", "droplet_first_image")


@pytest.mark.parametrize(
    "name",
    INTEGRATION_VARIABLES
    + FRAMES_USED_VARIABLES
    + CAMERA_VARIABLES
    + DIAGNOSTIC_VARIABLES
    + XPCS_HITS_VARIABLES,
)
def test_the_context_file_holds_no_heavy_integration_logic(name):
    """DAMNIT execs this file, so nothing defined here can reach a worker.

    The integration functions must stay in ``analysis.saxs`` and
    ``analysis.waxs``; a helper defined in the context file would fail to
    pickle only once the pool spawned. Each wrapper is a docstring, an import
    and a call — anything with statements beyond that has logic in it that
    belongs in the package instead.
    """
    import ast

    source = (CONTEXT_DIR / "context.py").read_text()
    imports = [
        line for line in source.splitlines() if line.startswith(("import ", "from "))
    ]
    assert not any("integrate_run" in line for line in imports)

    tree = ast.parse(source)
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    assert name in functions, f"{name} is not defined in the context file"
    body = functions[name].body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
    ):
        body = body[1:]  # the docstring
    kinds = [type(node).__name__ for node in body]
    assert kinds == ["ImportFrom", "Return"], (
        f"{name} does something other than import and call: {kinds}"
    )


def _context_tree():
    import ast

    return ast.parse((CONTEXT_DIR / "context.py").read_text())


def test_no_call_in_the_context_file_passes_overwrite():
    """CLAUDE.md pitfalls 26-27: DAMNIT reruns a pass whenever a dependant is
    requested, and with ``overwrite=True`` every rerun truncates the file.
    Reprocessing is done by moving the output aside
    (``scripts/reprocess_safe.py``), never from the context file."""
    import ast

    offenders = [
        f"line {node.lineno}"
        for node in ast.walk(_context_tree())
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


@pytest.mark.parametrize(
    "name",
    [
        "agipd_preprocessing",
        "jungfrau_preprocessing",
        "agipd_saxs",
        "jungfrau_waxs_jf1",
        "jungfrau_waxs_jf2",
        "xpcs_hits_preprocessing",
        "xpcs_hits_g2",
        "xpcs_hits_ttcf",
    ],
)
def test_the_integration_wrappers_pass_no_config(name):
    """The config DAMNIT runs with lives in ``analysis.*.damnit``, in one place.

    A keyword here — ``npt=200`` was one — is a config that scripts judging
    DAMNIT's files (``reprocess_safe``, ``is_complete``) cannot see.
    """
    import ast

    function = next(
        node
        for node in _context_tree().body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    call = function.body[-1].value
    assert isinstance(call, ast.Call)
    assert not call.keywords, f"{name} passes {[k.arg for k in call.keywords]}"


def test_the_xpcs_hits_readers_read_the_file_not_proc(variables):
    """Only the preprocessing variable reads proc; the others wait on it."""
    prep = {"prep": "xpcs_hits_preprocessing"}
    assert variables["xpcs_hits_g2"].arg_dependencies() == prep | {"rep": "rep_rate"}
    assert variables["xpcs_hits_ttcf"].arg_dependencies() == prep
    assert variables["xpcs_hits_overview"].arg_dependencies() == {
        "g2": "xpcs_hits_g2",
        "ttcf": "xpcs_hits_ttcf",
    }


def test_the_old_xpcs_variables_are_untouched(variables):
    """The hit-train pass is added beside ``xpcs_pipeline``, not in its place."""
    assert variables["xpcs_ttcf"].arg_dependencies() == {"ds": "xpcs_pipeline"}
    assert variables["xpcs_pipeline"].cluster is True


def test_the_xpcs_hits_titles_do_not_collide_with_the_old_xpcs_ones(variables):
    """``damnit reprocess --match`` compares title substrings."""
    new = {variables[n].title for n in (*XPCS_HITS_VARIABLES, "xpcs_hits_frames_used")}
    old = {
        v.title
        for n, v in variables.items()
        if n.startswith("xpcs_") and not n.startswith("xpcs_hits")
    }
    assert not new & old
    assert all(title.startswith("XPCS hits") for title in new)
