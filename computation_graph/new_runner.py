import dataclasses
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
    def find_inputs(
        attention: Attention,
        inputs_so_far: Tuple[Fact, ...],
    ) -> FrozenSet[Tuple[Fact, ...]]:
        index = len(inputs_so_far)
        if index == len(attention.sources):
            return gamla.wrap_tuple(inputs_so_far)
        current_node = attention.sources[index]
        if inputs_so_far:
            prev = gamla.last(inputs_so_far)
            first_interval_start = prev.interval.start
            first_interval_end = prev.interval.end
            if attention.is_sequential:
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
                    find_inputs(attention),
                ),
            ),
            gamla.concat,
        )

    def make_tasks(attention: Attention) -> Iterable[Task]:
        return gamla.pipe(
            (),
            find_inputs(attention),
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
        gamla.map(gamla.excepts(RefuseProcessing, gamla.just(None), _runner)),
        gamla.remove(gamla.equals(None)),
        gamla.concat_with(facts),
        frozenset,
        gamla.ternary(gamla.len_equals(len(facts)), gamla.identity, run(program)),
    )


# TEST CASE


def addition(a, _, c):
    return a + c


def parse_plus(some_string):
    if some_string == "+":
        return some_string
    raise RefuseProcessing


def parse_number(some_string):
    try:
        return int(some_string)
    except ValueError:
        raise RefuseProcessing


def number(x):
    return x


def string(x, y):
    return x + y


def is_larger_than_50(number):
    return number > 50


def describe(is_large, number):
    description = "larger than 50" if is_large else "not larger than 50"
    return f"{number} is {description}"


run(
    frozenset(
        [
            make_attention(describe, [is_larger_than_50, number]),
            Attention(addition, (number, parse_plus, number), is_sequential=True),
            make_attention(number, [parse_number]),
            make_attention(is_larger_than_50, [number]),
            make_attention(number, [addition]),
            make_attention(parse_number, [string]),
            make_attention(parse_number, [string]),
            Attention(string, (string, string), is_sequential=True),
            make_attention(parse_plus, [string]),
        ],
    ),
    frozenset(
        [
            *gamla.pipe(
                "4+5+777",
                enumerate,
                gamla.map(
                    gamla.star(
                        lambda index, current_char: Fact(
                            node=string,
                            value=current_char,
                            interval=Interval(index, index + 1),
                            inputs=(),
                        ),
                    ),
                ),
            ),
        ],
    ),
)
