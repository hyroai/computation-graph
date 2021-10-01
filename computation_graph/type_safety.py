import typing
from typing import Any, Callable, Optional, Tuple, Union

import gamla


def _handle_union_on_left(type1, type2):
    return gamla.pipe(
        type1, typing.get_args, gamla.allmap(lambda t: _is_subtype((t, type2)))
    )


def _handle_union_on_right(type1, type2):
    return gamla.pipe(
        type2, typing.get_args, gamla.anymap(lambda t: _is_subtype((type1, t)))
    )


_handle_union = gamla.anyjuxt(
    gamla.alljuxt(
        gamla.compose_left(gamla.head, typing.get_origin, gamla.equals(Union)),
        gamla.star(_handle_union_on_left),
    ),
    gamla.alljuxt(
        gamla.compose_left(gamla.second, typing.get_origin, gamla.equals(Union)),
        gamla.star(_handle_union_on_right),
    ),
)


def _forward_ref(x):
    def _forward_ref(*args, **kwargs):
        return x()(*args, **kwargs)

    return _forward_ref


_handle_generics = gamla.alljuxt(
    gamla.allmap(typing.get_origin),
    gamla.compose_left(gamla.map(typing.get_origin), gamla.star(issubclass)),
    gamla.compose_left(
        gamla.map(typing.get_args),
        gamla.star(zip),
        gamla.allmap(_forward_ref(lambda: _is_subtype)),
    ),
)

_handle_plain_classes = gamla.alljuxt(
    gamla.complement(gamla.anymap(typing.get_origin)), gamla.star(issubclass)
)

_is_subtype: Callable[[Tuple[Any, Any]], bool] = gamla.anyjuxt(
    gamla.inside(Any),
    _handle_union,
    _handle_generics,
    gamla.allmap(gamla.equals(Ellipsis)),
    _handle_plain_classes,
)


def can_compose(destination: Callable, source: Callable, key: Optional[str]) -> bool:
    s = typing.get_type_hints(source)
    d = typing.get_type_hints(destination)
    if "return" not in s:
        return True
    if key:
        if key not in d:
            return True
        d = d[key]
    else:
        if "return" in d:
            del d["return"]
        if not d:
            return True
        assert len(d) == 1
        d = gamla.head(d.values())
    return _is_subtype((s["return"], d))
