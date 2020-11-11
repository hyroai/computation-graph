import dataclasses
from typing import Any, Collection, FrozenSet, Tuple

import gamla

NodeId = int


@dataclasses.dataclass(frozen=True)
class Interval:
    start: float
    end: float


@dataclasses.dataclass(frozen=True)
class Fact:
    node: NodeId
    interval: Interval
    dependencies: FrozenSet["Fact"]
    output: Any


@dataclasses.dataclass(frozen=True)
class Attention:
    target: NodeId
    sources: Tuple[NodeId, ...]
    is_sequential: bool
    silence: Tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class Task:
    attention: Attention
    inputs: Tuple


class RefuseProcessing(Exception):
    pass


def _make_tasks_helper(nodes_to_find, is_sequential):
    pass


def _make_tasks(facts_index, attention: Attention):
    get_by_node, get_by_start, get_by_node_and_start = facts_index
    sources = attention.sources
    for x in sources:
        facts_index.get_by_node()


_index_facts = gamla.juxt(
    gamla.groupby_many(lambda fact: [fact.node]),
    gamla.groupby_many(lambda fact: [fact.interval.start]),
    gamla.groupby_many(lambda fact: [(fact.node, fact.interval.start)]),
    # Fact to dependencies.
    gamla.compose_left(gamla.groupby_many(gamla.attrgetter("dependencies")), gamla.reverse_graph)
)


def run(program: Collection[Attention], facts: Collection[Fact], node_id_to_callable):
    already_run = set()
    while True:
        new_tasks = (
            frozenset(gamla.mapcat(_make_tasks(_index_facts(facts)), program))
            - already_run
        )
        if not new_tasks:
            break
        for task in new_tasks:
            try:
                facts.add(
                    Fact(value=node_id_to_callable(task.attention.target)(*task.inputs), ...)
                )
                already_run.add(task)
            except RefuseProcessing:
                continue
    return facts


# TEST CASE


def addition(a, c):
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


def head_word(sentence):
    first_space = sentence.index(" ")
    return sentence[:first_space]


def tail_words(sentence):
    first_space = sentence.index(" ")
    remainder = sentence[first_space:]
    if remainder:
        return remainder
    raise RefuseProcessing


def number(x):
    return x


program = [
    Attention(addition, [number, parse_plus, number], silence=[1], is_sequential=True),
    Attention(number, [parse_number]),
    Attention(number, [addition]),
    Attention(parse_number, [head_word]),
    Attention(parse_plus, [head_word]),
    Attention(head_word, [tail_words]),
    Attention(tail_words, [tail_words]),
]


Fact(
    node=tail_words,
    value="4 + 5 + 7",
    interval=Interval(0, 1),
    dependencies=frozenset(),
)
