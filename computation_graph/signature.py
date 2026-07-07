import functools
import inspect
import types
from types import MappingProxyType
from typing import Callable, FrozenSet

import gamla

from computation_graph import base_types


def is_supported(signature: base_types.NodeSignature) -> bool:
    return not signature.optional_kwargs and not (
        signature.kwargs and signature.is_args
    )


def name(func: Callable) -> str:
    if isinstance(func, functools.partial):
        return func.func.__name__
    return func.__name__


def parameter_is_star(parameter) -> bool:
    return parameter.kind == inspect.Parameter.VAR_POSITIONAL


def parameter_is_double_star(parameter) -> bool:
    return parameter.kind == inspect.Parameter.VAR_KEYWORD


def _is_default(parameter):
    return parameter.default != parameter.empty


_parameter_name = gamla.attrgetter("name")

_func_parameters = gamla.compose_left(
    inspect.signature, gamla.attrgetter("parameters"), MappingProxyType.values, tuple
)


def _from_callable_via_inspect(func: Callable) -> base_types.NodeSignature:
    function_parameters = _func_parameters(func)
    return base_types.NodeSignature(
        is_args=gamla.anymap(parameter_is_star)(function_parameters),
        is_kwargs=gamla.anymap(parameter_is_double_star)(function_parameters),
        kwargs=gamla.pipe(
            function_parameters,
            gamla.remove(gamla.anyjuxt(parameter_is_star, parameter_is_double_star)),
            gamla.map(_parameter_name),
            tuple,
        ),
        optional_kwargs=gamla.pipe(
            function_parameters,
            gamla.remove(gamla.anyjuxt(parameter_is_star, parameter_is_double_star)),
            gamla.filter(_is_default),
            gamla.map(_parameter_name),
            tuple,
        ),
    )


def _as_plain_function(func: Callable):
    """The `inspect.signature`-equivalent target when it is a plain function whose
    signature is fully described by its code object, else None (wrappers with
    `__signature__`, partials, methods, builtins and classes need `inspect`)."""
    unwrapped = inspect.unwrap(func, stop=lambda f: hasattr(f, "__signature__"))
    if not isinstance(unwrapped, types.FunctionType) or hasattr(
        unwrapped, "__signature__"
    ):
        return None
    return unwrapped


def from_callable(func: Callable) -> base_types.NodeSignature:
    plain_function = _as_plain_function(func)
    if plain_function is None:
        return _from_callable_via_inspect(func)
    code = plain_function.__code__
    positional = code.co_varnames[: code.co_argcount]
    keyword_only = code.co_varnames[
        code.co_argcount : code.co_argcount + code.co_kwonlyargcount
    ]
    defaults = plain_function.__defaults__ or ()
    keyword_defaults = plain_function.__kwdefaults__ or {}
    return base_types.NodeSignature(
        is_args=bool(code.co_flags & inspect.CO_VARARGS),
        is_kwargs=bool(code.co_flags & inspect.CO_VARKEYWORDS),
        kwargs=positional + keyword_only,
        optional_kwargs=(
            positional[len(positional) - len(defaults) :]
            + tuple(name for name in keyword_only if name in keyword_defaults)
        ),
    )


def parameters(signature: base_types.NodeSignature) -> FrozenSet[str]:
    return frozenset(
        {
            *signature.kwargs,
            *signature.optional_kwargs,
            *(("**kwargs",) if signature.is_kwargs else ()),
        }
    )


is_unary = gamla.compose_left(parameters, gamla.len_equals(1))
