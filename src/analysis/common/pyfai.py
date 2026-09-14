"""Guards for talking to pyFAI safely.

pyFAI resolves a method request into whatever engine it can actually build, and
says so only in the result. A method given as a string, or one whose engine is
unavailable, therefore substitutes a different integrator silently, which is why
both passes request a tuple and check what came back.
"""

from __future__ import annotations

from typing import Any

__all__ = ["check_resolved_method", "resolved_method"]


def resolved_method(result: Any) -> tuple[str, str, str]:
    """The method pyFAI actually ran, as a ``(split, algo, impl)`` tuple."""
    method = result.method
    return (method.split_lower, method.algo_lower, method.impl_lower)


def check_resolved_method(result: Any, requested: tuple[str, str, str]) -> None:
    """Fail when pyFAI ran a different integrator than the one asked for.

    :param result: an integration result, probed with the requested method.
    :param requested: the method the config asked for.
    :raises RuntimeError: the two differ, so every q bin would come from an
        integrator nobody chose.
    """
    resolved = resolved_method(result)
    if resolved != tuple(requested):
        raise RuntimeError(
            f"pyFAI resolved method {resolved}, requested {tuple(requested)}; "
            "a method string or an unavailable engine has silently substituted "
            "another integrator"
        )
