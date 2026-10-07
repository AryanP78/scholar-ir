"""Rule-based structured query parser (Query parsers, Scoring lecture; Westlaw-style syntax).

Syntax
    transformer retrieval               free-text terms (ranked, OR semantics)
    a AND b, a OR b, NOT a, ( ... )     Boolean operators (upper case) and grouping
    "dense passage retrieval"           phrase (positional index)
    a NEAR/5 b                          proximity: a and b within 5 positions (a, b may be phrases)
    "query expansion" NEAR/5            the phrase's words within 5 positions of each other, any order
    title:transformer                   zone restriction (title | abstract)
    abstract:"contrastive learning"     zone-restricted phrase
    year:2020  year:>=2020  year:<2015  year:2018..2022      parametric filters
    cat:cs.IR                           category filter (several cat: filters are OR-ed)

Semantics of juxtaposition (no operator between operands): bare words are *optional* (they only
rank); phrases, NEAR clauses and parenthesised/Boolean groups are *required*; NOT clauses exclude.
So  `"dense retrieval" passages NOT image`  = docs containing the phrase, without "image",
ranked by all positive terms.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Union

ZONES = ("title", "abstract")


@dataclass
class Term:
    text: str
    zone: str = "all"


@dataclass
class Phrase:
    text: str
    zone: str = "all"


@dataclass
class Near:
    left: "Node"
    right: "Node | None"
    k: int


@dataclass
class And:
    children: list["Node"]


@dataclass
class Or:
    children: list["Node"]


@dataclass
class Not:
    child: "Node"


@dataclass
class Seq:
    """Juxtaposed operands (see module docstring)."""
    children: list["Node"]


Node = Union[Term, Phrase, Near, And, Or, Not, Seq]


@dataclass
class Filters:
    year_min: int | None = None
    year_max: int | None = None
    categories: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return self.year_min is None and self.year_max is None and not self.categories


@dataclass
class ParsedQuery:
    raw: str
    root: Node | None
    filters: Filters

    @property
    def constrained(self) -> bool:
        """True if the Boolean part restricts the candidate set beyond 'contains any query term'."""
        return _is_constraining(self.root)

    def to_string(self) -> str:
        return render(self.root) + render_filters(self.filters)


class QuerySyntaxError(ValueError):
    pass


_TOKEN = re.compile(
    r'\s*(?:(?P<lp>\()|(?P<rp>\))|(?P<near>NEAR/(?P<k>\d+))|(?P<field>(?:title|abstract|year|cat):)'
    r'|"(?P<quoted>[^"]*)"|(?P<word>[^\s()"]+))')


def _lex(q: str) -> list[tuple[str, str]]:
    toks, i = [], 0
    while i < len(q):
        m = _TOKEN.match(q, i)
        if not m or m.end() == i:
            if q[i:].strip() == "":
                break
            raise QuerySyntaxError(f"cannot parse near: {q[i:i+20]!r}")
        i = m.end()
        if m.group("lp"):
            toks.append(("(", "("))
        elif m.group("rp"):
            toks.append((")", ")"))
        elif m.group("near"):
            toks.append(("NEAR", m.group("k")))
        elif m.group("field"):
            toks.append(("FIELD", m.group("field")[:-1]))
        elif m.group("quoted") is not None:
            toks.append(("PHRASE", m.group("quoted")))
        elif m.group("word"):
            w = m.group("word")
            toks.append((w, w) if w in ("AND", "OR", "NOT") else ("WORD", w))
    return toks


_YEAR = re.compile(r"^(>=|<=|>|<)?(\d{4})(?:\.\.(\d{4}))?$")


def _apply_year(f: Filters, spec: str) -> None:
    m = _YEAR.match(spec)
    if not m:
        raise QuerySyntaxError(f"bad year filter {spec!r}")
    op, a, b = m.group(1), int(m.group(2)), m.group(3)
    if b:
        f.year_min, f.year_max = a, int(b)
    elif op == ">=":
        f.year_min = a
    elif op == ">":
        f.year_min = a + 1
    elif op == "<=":
        f.year_max = a
    elif op == "<":
        f.year_max = a - 1
    else:
        f.year_min = f.year_max = a


class _Parser:
    def __init__(self, toks):
        self.toks, self.i = toks, 0

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else (None, None)

    def take(self):
        t = self.peek()
        self.i += 1
        return t

    def parse_or(self):
        kids = [self.parse_and()]
        while self.peek()[0] == "OR":
            self.take()
            kids.append(self.parse_and())
        return kids[0] if len(kids) == 1 else Or(kids)

    def parse_and(self):
        kids = [self.parse_seq()]
        while self.peek()[0] == "AND":
            self.take()
            kids.append(self.parse_seq())
        return kids[0] if len(kids) == 1 else And(kids)

    def parse_seq(self):
        kids = []
        while self.peek()[0] not in (None, ")", "OR", "AND"):
            kids.append(self.parse_unary())
        if not kids:
            raise QuerySyntaxError("empty expression")
        return kids[0] if len(kids) == 1 and not isinstance(kids[0], Not) else Seq(kids)

    def parse_unary(self):
        if self.peek()[0] == "NOT":
            self.take()
            return Not(self.parse_unary())
        return self.parse_near()

    def parse_near(self):
        left = self.parse_primary()
        while self.peek()[0] == "NEAR":
            k = int(self.take()[1])
            nxt = self.peek()[0]
            right = self.parse_primary() if nxt in ("WORD", "PHRASE", "FIELD", "(") else None
            left = Near(left, right, k)
        return left

    def parse_primary(self):
        kind, val = self.take()
        if kind == "(":
            node = self.parse_or()
            if self.take()[0] != ")":
                raise QuerySyntaxError("missing )")
            return node
        if kind == "PHRASE":
            return Phrase(val)
        if kind == "WORD":
            return Term(val)
        if kind == "FIELD":
            k2, v2 = self.take()
            if k2 == "PHRASE":
                return Phrase(v2, zone=val)
            if k2 == "WORD":
                return Term(v2, zone=val)
            if k2 == "(":
                node = self.parse_or()
                if self.take()[0] != ")":
                    raise QuerySyntaxError("missing )")
                return _set_zone(node, val)
            raise QuerySyntaxError(f"bad value after {val}:")
        raise QuerySyntaxError(f"unexpected token {val!r}")


def _set_zone(node, zone):
    if isinstance(node, (Term, Phrase)):
        node.zone = zone
    elif isinstance(node, Near):
        _set_zone(node.left, zone)
        if node.right is not None:
            _set_zone(node.right, zone)
    elif isinstance(node, Not):
        _set_zone(node.child, zone)
    elif isinstance(node, (And, Or, Seq)):
        for c in node.children:
            _set_zone(c, zone)
    return node


def parse(query: str) -> ParsedQuery:
    """Parse a query string into a Boolean/phrase AST plus parametric filters."""
    toks = _lex(query)
    filters = Filters()
    kept = []
    i = 0
    while i < len(toks):
        kind, val = toks[i]
        if kind == "FIELD" and val in ("year", "cat"):
            if i + 1 >= len(toks) or toks[i + 1][0] not in ("WORD", "PHRASE"):
                raise QuerySyntaxError(f"missing value after {val}:")
            v = toks[i + 1][1]
            if val == "year":
                _apply_year(filters, v)
            else:
                filters.categories.append(v)
            i += 2
            continue
        kept.append(toks[i])
        i += 1
    root = None
    if kept:
        p = _Parser(kept)
        root = p.parse_or()
        if p.i != len(kept):
            raise QuerySyntaxError(f"unexpected {kept[p.i][1]!r}")
    return ParsedQuery(raw=query, root=root, filters=filters)


def _is_constraining(node) -> bool:
    if node is None:
        return False
    if isinstance(node, Term):
        return False
    if isinstance(node, Seq):
        return any(not isinstance(c, Term) for c in node.children)
    return True


def positive_clauses(node, negated: bool = False):
    """Yield (Term|Phrase, zone) leaves that are not under a NOT — these drive ranking."""
    if node is None:
        return
    if isinstance(node, (Term, Phrase)):
        if not negated:
            yield node
    elif isinstance(node, Near):
        yield from positive_clauses(node.left, negated)
        if node.right is not None:
            yield from positive_clauses(node.right, negated)
    elif isinstance(node, Not):
        yield from positive_clauses(node.child, not negated)
    else:
        for c in node.children:
            yield from positive_clauses(c, negated)


def render(node) -> str:
    if node is None:
        return ""
    z = lambda n: "" if n.zone == "all" else f"{n.zone}:"
    if isinstance(node, Term):
        return f"{z(node)}{node.text}"
    if isinstance(node, Phrase):
        return f'{z(node)}"{node.text}"'
    if isinstance(node, Near):
        return f"({render(node.left)} NEAR/{node.k}{' ' + render(node.right) if node.right else ''})"
    if isinstance(node, Not):
        return f"NOT {render(node.child)}"
    if isinstance(node, And):
        return "(" + " AND ".join(render(c) for c in node.children) + ")"
    if isinstance(node, Or):
        return "(" + " OR ".join(render(c) for c in node.children) + ")"
    return " ".join(render(c) for c in node.children)


def render_filters(f: Filters) -> str:
    out = []
    if f.year_min is not None and f.year_max is not None:
        out.append(f"year:{f.year_min}" if f.year_min == f.year_max else f"year:{f.year_min}..{f.year_max}")
    elif f.year_min is not None:
        out.append(f"year:>={f.year_min}")
    elif f.year_max is not None:
        out.append(f"year:<={f.year_max}")
    out += [f"cat:{c}" for c in f.categories]
    return (" " + " ".join(out)) if out else ""
