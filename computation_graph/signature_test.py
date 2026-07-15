import functools
import inspect

import pytest

from computation_graph import base_types, signature


def _reference_node_signature(func) -> base_types.NodeSignature:
    """Independent, inspect-based derivation of the NodeSignature that
    `from_callable`'s code-object fast path must reproduce."""
    parameters = tuple(inspect.signature(func).parameters.values())
    named = tuple(
        p for p in parameters if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    )
    return base_types.NodeSignature(
        is_args=any(p.kind is p.VAR_POSITIONAL for p in parameters),
        is_kwargs=any(p.kind is p.VAR_KEYWORD for p in parameters),
        kwargs=tuple(p.name for p in named),
        optional_kwargs=tuple(p.name for p in named if p.default is not p.empty),
    )


def _no_args():
    ...


def _positional(alpha, beta):
    ...


def _with_defaults(alpha, beta):
    ...


# Defaults are set out-of-band so the test corpus needn't use default-arg syntax.
_with_defaults.__defaults__ = (1,)


def _var_args(*args):
    ...


def _var_kwargs(**kwargs):
    ...


def _var_args_kwargs(*args, **kwargs):
    ...


def _keyword_only(*, alpha, beta):
    ...


def _keyword_only_default(*, alpha, beta):
    ...


_keyword_only_default.__kwdefaults__ = {"beta": 2}


def _all_kinds(alpha, beta, *args, gamma, delta, **kwargs):
    ...


_all_kinds.__defaults__ = (1,)
_all_kinds.__kwdefaults__ = {"delta": 2}


def _positional_then_keyword_only(alpha, *, beta):
    ...


async def _async(alpha, beta):
    ...


@functools.wraps(_positional)
def _functools_wraps(*args, **kwargs):
    return _positional(*args, **kwargs)


class _Klass:
    def method(self, alpha, beta):
        ...


def _signature_override(alpha):
    ...


# A __signature__ that deliberately disagrees with the code object, so the case
# fails loudly if the fast path ever reads the code object instead of deferring.
setattr(
    _signature_override,
    "__signature__",
    inspect.Signature(
        [
            inspect.Parameter("x", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter("y", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=3),
        ]
    ),
)


# Cases up to `functools_wraps` take the code-object fast path; the rest
# (partials, bound method, __signature__ override) must defer to inspect.
_CASES = [
    ("no_args", _no_args),
    ("positional", _positional),
    ("with_defaults", _with_defaults),
    ("var_args", _var_args),
    ("var_kwargs", _var_kwargs),
    ("var_args_kwargs", _var_args_kwargs),
    ("keyword_only", _keyword_only),
    ("keyword_only_default", _keyword_only_default),
    ("all_kinds", _all_kinds),
    ("positional_then_keyword_only", _positional_then_keyword_only),
    ("async", _async),
    ("lambda", (lambda alpha, beta: None)),
    ("functools_wraps", _functools_wraps),
    ("partial_positional", functools.partial(_positional, 1)),
    ("partial_keyword", functools.partial(_positional, beta=5)),
    ("bound_method", _Klass().method),
    ("signature_override", _signature_override),
]


@pytest.mark.parametrize(
    "func", [func for _, func in _CASES], ids=[name for name, _ in _CASES]
)
def test_from_callable_matches_inspect(func):
    assert signature.from_callable(func) == _reference_node_signature(func)
