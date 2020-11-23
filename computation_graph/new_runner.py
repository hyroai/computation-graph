import dataclasses
import functools
from typing import Any, Callable, FrozenSet, Iterable, Tuple

import gamla

NodeId = Callable


@dataclasses.dataclass(frozen=True)
class Interval:
    start: float
    end: float


@dataclasses.dataclass(frozen=True)
class Fact:
    node: NodeId
    interval: Interval
    inputs: Tuple[Any, ...]
    value: Any


@dataclasses.dataclass(frozen=True)
class Attention:
    node: NodeId
    sources: Tuple[NodeId, ...]
    is_sequential: bool


def make_attention(node, sources: Iterable):
    return Attention(node, tuple(sources), False)


def sequential_attention(node, sources):
    return Attention(node, tuple(sources), True)


@dataclasses.dataclass(frozen=True)
class Task:
    node: NodeId
    intervals: Tuple[Interval, ...]
    values: Tuple[Fact, ...]


class RefuseProcessing(Exception):
    pass


@gamla.curry
def _is_coherent(
    fact_to_inputs: Callable[[Fact], FrozenSet[FrozenSet[Fact]]],
    task: Task,
) -> bool:
    return True


_is_already_done = gamla.compose_left(
    gamla.map(gamla.juxt(gamla.attrgetter("inputs"), gamla.attrgetter("node"))),
    gamla.star(zip),
    gamla.map(gamla.compose_left(frozenset, gamla.contains)),
    gamla.star(gamla.alljuxt),
)


_facts_to_values = gamla.compose_left(gamla.map(gamla.attrgetter("value")), tuple)
_facts_to_intervals = gamla.compose_left(gamla.map(gamla.attrgetter("interval")), tuple)


def _make_tasks(facts):
    (
        get_by_node,
        get_by_node_and_start,
        get_by_node_and_start_and_end,
        fact_to_inputs,
    ) = _index_facts(facts)

    is_already_done = _is_already_done(facts)

    @gamla.curry
    @functools.lru_cache(maxsize=None)
    def find_inputs(
        sources: Tuple[NodeId, ...],
        is_sequential: bool,
        inputs_so_far: Tuple[Fact, ...],
    ) -> FrozenSet[Tuple[Fact, ...]]:
        index = len(inputs_so_far)
        if index == len(sources):
            return frozenset([inputs_so_far])
        current_node = sources[index]
        if inputs_so_far:
            prev = gamla.last(inputs_so_far)
            first_interval_start = prev.interval.start
            first_interval_end = prev.interval.end
            if is_sequential:
                options = get_by_node_and_start((current_node, first_interval_end))
            else:
                options = get_by_node_and_start_and_end(
                    (current_node, first_interval_start, first_interval_end),
                )
        else:
            options = get_by_node(current_node)
        return gamla.pipe(
            options,
            gamla.map(
                gamla.compose_left(
                    lambda option: (*inputs_so_far, option),
                    find_inputs(sources, is_sequential),
                ),
            ),
            gamla.concat,
            frozenset,
        )

    def make_tasks(attention: Attention) -> Iterable[Task]:
        return gamla.pipe(
            (),
            find_inputs(attention.sources, attention.is_sequential),
            gamla.map(
                lambda inputs: Task(
                    attention.node,
                    _facts_to_intervals(inputs),
                    _facts_to_values(inputs),
                ),
            ),
            gamla.remove(is_already_done),
            gamla.filter(_is_coherent(fact_to_inputs)),
        )

    return make_tasks


def _runner(task: Task):
    return Fact(
        node=task.node,
        value=task.node(*task.values),
        inputs=task.values,
        interval=Interval(
            gamla.head(task.intervals).start,
            gamla.last(task.intervals).end,
        ),
    )


_index_facts = gamla.compose_left(
    gamla.juxt(
        gamla.groupby_many(lambda fact: [fact.node]),
        gamla.groupby_many(lambda fact: [(fact.node, fact.interval.start)]),
        gamla.groupby_many(
            lambda fact: [(fact.node, fact.interval.start, fact.interval.end)],
        ),
        # Fact to inputs.
        gamla.compose_left(
            gamla.groupby_many(gamla.attrgetter("inputs")),
            gamla.reverse_graph,
        ),
    ),
    gamla.map(
        gamla.compose_left(
            gamla.attrgetter("__getitem__"),
            gamla.excepts(KeyError, gamla.just(frozenset())),
        ),
    ),
)


@gamla.curry
def run(program: FrozenSet[Attention], facts: FrozenSet[Fact]) -> FrozenSet[Fact]:
    return gamla.pipe(
        program,
        gamla.mapcat(_make_tasks(facts)),
        # TODO(uri): Prevent redundant calls to `_runner`.
        gamla.map(gamla.excepts(RefuseProcessing, gamla.just(None), _runner)),
        gamla.remove(gamla.equals(None)),
        gamla.concat_with(facts),
        frozenset,
        gamla.ternary(gamla.len_equals(len(facts)), gamla.identity, run(program)),
    )


def test_parsing():
    def addition(a, _, c):
        return a + c

    def parse_plus(some_string):
        if some_string == "+":
            return some_string
        raise RefuseProcessing

    def parse_int(some_string):
        return int(some_string)

    def number(x):
        return x

    def is_larger_than_50(number):
        return number > 50

    def describe(is_large, number):
        description = "larger than 50" if is_large else "not larger than 50"
        return f"{number} is {description}"

    def char(x):
        return x

    def digit(x):
        if x in map(str, range(10)):
            return x
        raise RefuseProcessing

    def digits(x, y):
        return x + y

    gamla.pipe(
        run(
            frozenset(
                [
                    make_attention(parse_plus, [char]),
                    # Digits to numbers
                    make_attention(digit, [char]),
                    sequential_attention(digits, [digit, digit]),
                    sequential_attention(digits, [digits, digit]),
                    make_attention(parse_int, [digit]),
                    make_attention(parse_int, [digits]),
                    make_attention(number, [parse_int]),
                    # Number stuff
                    make_attention(number, [addition]),
                    make_attention(is_larger_than_50, [number]),
                    make_attention(describe, [is_larger_than_50, number]),
                    sequential_attention(addition, [number, parse_plus, number]),
                ],
            ),
            frozenset(
                [
                    *gamla.pipe(
                        "5+75+75+74+5+777+5+777+5+777+5",
                        enumerate,
                        gamla.map(
                            gamla.star(
                                lambda index, current_char: Fact(
                                    node=char,
                                    value=current_char,
                                    interval=Interval(index, index + 1),
                                    inputs=(),
                                ),
                            ),
                        ),
                    ),
                ],
            ),
        ),
        gamla.filter(lambda fact: fact.interval.start == 0),
        gamla.sort_by(lambda fact: fact.interval.end - fact.interval.start),
    )


def test_memory():
    def describe(phone_number):
        return f"Your phone number is {phone_number}"

    def char(x):
        return x

    def digit(x):
        if x in map(str, range(10)):
            return x
        raise RefuseProcessing

    def digits(x, y):
        return x + y

    def parse_phone_number(digits):
        if len(digits) != 9:
            raise RefuseProcessing
        return digits

    def remember(something):
        return something

    gamla.pipe(
        run(
            frozenset(
                [
                    # Digits to numbers
                    make_attention(digit, [char]),
                    sequential_attention(digits, [digit, digit]),
                    sequential_attention(digits, [digits, digit]),
                    make_attention(parse_phone_number, [digits]),
                    # memory
                    make_attention(remember, [parse_phone_number]),
                    make_attention(remember, [remember], delay=1),
                    # final answer
                    make_attention(describe, [parse_phone_number]),
                ],
            ),
            frozenset(
                [
                    *gamla.pipe(
                        "0123456789",
                        enumerate,
                        gamla.map(
                            gamla.star(
                                lambda index, current_char: Fact(
                                    node=char,
                                    value=current_char,
                                    interval=Interval(index, index + 1),
                                    inputs=(),
                                ),
                            ),
                        ),
                    ),
                ],
            ),
        ),
        gamla.filter(lambda fact: fact.interval.start == 0),
        gamla.sort_by(lambda fact: fact.interval.end - fact.interval.start),
    )
