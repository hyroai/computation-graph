import typing
from typing import Any, Callable, Optional, Union

import gamla


@gamla.curry
def _is_subtype(type1, type2):
    if Any in [type1, type2]:
        return True
    if typing.get_origin(type1) == Union:
        return gamla.allmap(_is_subtype(type2))(typing.get_args(type1))
    if typing.get_origin(type2) == Union:
        return gamla.anymap(_is_subtype(type1))(typing.get_args(type2))
    if gamla.anymap(typing.get_origin)([type1, type2]):
        return (
            gamla.allmap(typing.get_origin)([type1, type2])
            and issubclass(typing.get_origin(type1), typing.get_origin(type2))
            and gamla.pipe(
                [type1, type2],
                gamla.map(typing.get_args),
                gamla.star(zip),
                gamla.allmap(gamla.star(_is_subtype)),
            )
        )
    return issubclass(type1, type2)


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
    return _is_subtype(s["return"], d)
