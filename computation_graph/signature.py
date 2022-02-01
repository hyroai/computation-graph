import functools
import inspect
from types import MappingProxyType
from typing import Callable, FrozenSet, Tuple

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


@gamla.before(
    gamla.compose_left(
        inspect.signature,
        gamla.attrgetter("parameters"),
        MappingProxyType.values,
        tuple,
    )
)
def from_callable(function_parameters: Tuple) -> base_types.NodeSignature:
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


def parameters(signature: base_types.NodeSignature) -> FrozenSet[str]:
    return frozenset(
        {
            *signature.kwargs,
            *signature.optional_kwargs,
            *(("**kwargs",) if signature.is_kwargs else ()),
        }
    )


is_unary = gamla.compose_left(parameters, gamla.len_equals(1))
