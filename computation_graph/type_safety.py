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


_origin_equals = gamla.compose_left(gamla.equals, gamla.before(typing.get_origin))

_handle_union = gamla.case_dict(
    {
        gamla.compose_left(gamla.head, _origin_equals(Union)): gamla.star(
            _handle_union_on_left
        ),
        gamla.compose_left(gamla.second, _origin_equals(Union)): gamla.star(
            _handle_union_on_right
        ),
    }
)


def _rewrite_optional(x):
    return Union[None, typing.get_args(x)]


def _forward_ref(x):
    def _forward_ref(*args, **kwargs):
        return x()(*args, **kwargs)

    return _forward_ref


_handle_generics = gamla.alljuxt(
    gamla.compose_left(gamla.map(typing.get_origin), gamla.star(issubclass)),
    gamla.compose_left(
        gamla.map(typing.get_args),
        gamla.star(zip),
        gamla.allmap(_forward_ref(lambda: _is_subtype)),
    ),
)


_is_subtype: Callable[[Tuple[Any, Any]], bool] = gamla.compose_left(
    gamla.map(gamla.when(_origin_equals(Optional), _rewrite_optional)),
    tuple,
    gamla.case_dict(
        {
            gamla.anymap(_origin_equals(Union)): _handle_union,
            gamla.allmap(typing.get_origin): _handle_generics,
            gamla.inside(Ellipsis): gamla.allmap(gamla.equals(Ellipsis)),
            gamla.inside(Any): gamla.compose_left(gamla.second, gamla.equals(Any)),
            gamla.inside(None): gamla.allmap(
                gamla.contains(
                    [
                        None,
                        # Needed as `get_args` on `Union[None, x]` will give back a `NoneType`.
                        type(None),
                    ]
                )
            ),
            gamla.complement(gamla.anymap(typing.get_origin)): gamla.star(issubclass),
            gamla.just(True): gamla.just(False),
        }
    ),
)

is_subtype = gamla.compose_left(gamla.pack, _is_subtype)


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
    return is_subtype(s["return"], d)
