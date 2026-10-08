"""Elapsed time is measured on a monotonic clock, never as the difference of two wall-clock readings.

``time.time()`` reads the wall clock, which NTP (or macOS's ``timed``) may step at any moment, so
the difference of two readings is the elapsed time plus whatever correction landed in between. A
4.014 s step back in the middle of a recall logged it as taking -1.872 s, and OpenTelemetry rejects
a negative histogram value: it warned and dropped six samples across
``hindsight.recall.phase.duration``, ``hindsight.operation.duration`` and ``hindsight.http.duration``.

``time.perf_counter()`` cannot be stepped, so ``end - start`` is never negative. ``time.time()`` is
still right wherever the value is itself a point in time: a token expiry, a file mtime, an id.

A subtraction or a comparison fails this test when both sides are clock readings and the wall clock
is one of them -- on both ends, or on one end with ``perf_counter``/``monotonic`` on the other, which
is what a half-done conversion looks like and is off by about 1.8e9 s rather than a few. Comparisons
cover deadlines (``while time.monotonic() < deadline``): a reading plus an offset is still a reading.
Readings are followed across the package:

- a call through the ``time`` module under any alias, or ``from time import ...``;
- a local or module-level name, a ``self`` attribute, or a function parameter whose bindings are all
  readings (``None``/number placeholders aside). A parameter takes what the package's call sites pass
  to it; a call is matched to its function by name when the name is unique, else within the same
  class (``self.f()``) or module (``f()``), and not at all when that is still ambiguous;
- a field: a keyword argument ``started_at=<reading>``, an assignment ``obj.started_at = <reading>``
  or a ``field(default_factory=time.perf_counter)`` makes ``<anything>.started_at`` a reading;
- a string key -- the request scope's ``hs_*`` stamps are set in observability.py and http.py and
  subtracted in ``api_recall`` -- read back with ``[...]`` or ``.get(...)``.

- an element of a tuple kept in ``self.x[key]`` and unpacked on the way out
  (``expires_at, value = self._entries.get(key)``).

Not followed: ``**kwargs``, dict literals, ``time.time_ns()`` and ``datetime`` arithmetic -- no
duration or deadline here is built from them.

A subtraction of two wall-clock points in time that is right as written goes in ``EXEMPT`` with the
reason.
"""

from __future__ import annotations

import ast
import functools
from dataclasses import dataclass
from pathlib import Path

import pytest

import hindsight_api

PACKAGE_ROOT = Path(hindsight_api.__file__).parent

WALL = "wall"
MONOTONIC = "monotonic"
_CLOCK_OF = {"time": WALL, "perf_counter": MONOTONIC, "monotonic": MONOTONIC}

Function = ast.FunctionDef | ast.AsyncFunctionDef
Kinds = frozenset[str]

#: (module, the subtraction as ``ast.unparse`` prints it) -> why it is right on the wall clock.
EXEMPT: dict[tuple[str, str], str] = {
    (
        "engine/providers/xai_oauth_auth.py",
        "time.time() - credential.obtained_at",
    ): "obtained_at is persisted with the credential and read back by later processes; only the wall clock "
    "is shared across them",
}


class _Clocks:
    """How one module spells each clock: ``time.time()``, ``tm.time()`` or a bare ``time()``."""

    def __init__(self, tree: ast.AST) -> None:
        self.modules: set[str] = set()
        self.functions: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.modules |= {alias.asname or alias.name for alias in node.names if alias.name == "time"}
            elif isinstance(node, ast.ImportFrom) and node.module == "time":
                for alias in node.names:
                    if alias.name in _CLOCK_OF:
                        self.functions[alias.asname or alias.name] = _CLOCK_OF[alias.name]

    def of_ref(self, node: ast.AST) -> str | None:
        """The clock a function reference names: ``time.perf_counter`` or a bare ``perf_counter``."""
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in self.modules:
            return _CLOCK_OF.get(node.attr)
        if isinstance(node, ast.Name):
            return self.functions.get(node.id)
        return None

    def of_call(self, node: ast.AST) -> str | None:
        if not isinstance(node, ast.Call) or node.args or node.keywords:
            return None
        return self.of_ref(node.func)


@dataclass
class _Module:
    path: str
    tree: ast.Module
    clocks: _Clocks


@dataclass
class _Unit:
    """An outermost function and everything nested in it; nested functions read its names."""

    module: _Module
    node: Function
    owner: ast.ClassDef | None


@dataclass(frozen=True)
class _Unpacked:
    """``a, b = source``: the ``index``-th element of ``source``."""

    source: ast.AST
    index: int


def _is_placeholder(node: ast.AST) -> bool:
    """``None``, a number, or ``float("-inf")`` -- an initial value that is not a reading yet."""
    if isinstance(node, ast.UnaryOp):
        node = node.operand
    if isinstance(node, ast.Constant):
        return node.value is None or isinstance(node.value, (int, float))
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "float"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Constant)
    )


def _string_key(node: ast.AST) -> str | None:
    """``x["k"]`` or ``x.get("k")`` -> ``"k"``."""
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
        return node.slice.value
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    ):
        return node.args[0].value
    return None


def _self_attr(node: ast.AST) -> str | None:
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
        return node.attr
    return None


def _is_method(function: Function, owner: ast.ClassDef | None) -> bool:
    if owner is None or function not in owner.body:
        return False
    return not any(isinstance(d, ast.Name) and d.id == "staticmethod" for d in function.decorator_list)


def _default_factory_clock(value: ast.AST | None, clocks: _Clocks) -> str | None:
    """``field(default_factory=time.perf_counter)`` -> ``"monotonic"``."""
    if not (isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "field"):
        return None
    for kw in value.keywords:
        if kw.arg == "default_factory":
            return clocks.of_ref(kw.value)
    return None


class _Package:
    """Which clock every name, attribute, parameter, field and string key carries, across the given modules."""

    _ROUNDS = 20

    def __init__(self, sources: dict[str, str]) -> None:
        self.modules = [
            _Module(path, tree, _Clocks(tree)) for path, tree in ((p, ast.parse(s)) for p, s in sources.items())
        ]
        self.units: list[_Unit] = []
        self.defs: dict[str, list[tuple[Function, ast.ClassDef | None, str]]] = {}
        for module in self.modules:
            self._index(module.tree, module, owner=None, unit=None)
        self.params: dict[tuple[int, str], set[str]] = {}
        self.keys: dict[str, set[str]] = {}
        self.fields: dict[str, set[str]] = {}
        self.attrs: dict[tuple[int, str], Kinds] = {}
        self.declared: set[tuple[int, str]] = set()
        self.containers: dict[tuple[int, str], set[str]] = {}
        self.container_items: dict[tuple[int, str, int], set[str]] = {}
        self.returns: dict[int, Kinds] = {}
        self.module_names: dict[str, dict[str, Kinds]] = {}
        self.names: dict[int, dict[str, Kinds]] = {}
        self.bound: dict[int, set[str]] = {}
        for _ in range(self._ROUNDS):
            before = self._state()
            self._propagate()
            if self._state() == before:
                break
        else:
            raise AssertionError(f"clock propagation did not settle in {self._ROUNDS} rounds")

    def _state(self) -> str:
        return repr(
            (
                sorted((k, sorted(v)) for k, v in self.params.items()),
                sorted((k, sorted(v)) for k, v in self.keys.items()),
                sorted((k, sorted(v)) for k, v in self.fields.items()),
                sorted(self.attrs.items()),
                sorted((k, sorted(v)) for k, v in self.containers.items()),
                sorted((k, sorted(v)) for k, v in self.container_items.items()),
                sorted(self.returns.items()),
                sorted((p, sorted(n.items())) for p, n in self.module_names.items()),
            )
        )

    def _index(self, node: ast.AST, module: _Module, owner: ast.ClassDef | None, unit: _Unit | None) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                self._index(child, module, owner=child, unit=unit)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.defs.setdefault(child.name, []).append((child, owner, module.path))
                if unit is None:
                    inner = _Unit(module, child, owner)
                    self.units.append(inner)
                    self._index(child, module, owner=owner, unit=inner)
                else:
                    self._index(child, module, owner=owner, unit=unit)
            else:
                self._index(child, module, owner=owner, unit=unit)

    # ── what a single expression reads ───────────────────────────────────────────────────────────
    def kind(self, unit: _Unit, names: dict[str, Kinds], node: ast.AST) -> Kinds | None:
        clock = unit.module.clocks.of_call(node)
        if clock:
            return frozenset({clock})
        if isinstance(node, ast.Call):
            container = self._container_read(unit, node)
            if container is not None:
                return container
            returned = [self.returns[id(f)] for f, _ in self._callees(unit, node) if id(f) in self.returns]
            if returned:
                return frozenset().union(*returned)
        if isinstance(node, ast.IfExp):
            # `time.monotonic() + t if t > 0 else None`: the branch that is not a placeholder decides.
            branches = [b for b in (node.body, node.orelse) if not _is_placeholder(b)]
            kinds = [self.kind(unit, names, b) for b in branches]
            return frozenset().union(*kinds) if kinds and all(kinds) else None
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            # A reading shifted by an offset (`time.monotonic() + timeout`) is still on that clock; the
            # difference of two readings is not, so only one side may be a reading.
            left, right = self.kind(unit, names, node.left), self.kind(unit, names, node.right)
            if (left is None) != (right is None) and not (isinstance(node.op, ast.Sub) and left is None):
                return left or right
            return None
        if isinstance(node, ast.Subscript) and not isinstance(node.slice, ast.Constant):
            attr = _self_attr(node.value)
            if attr is not None and unit.owner is not None and (id(unit.owner), attr) in self.containers:
                return frozenset(self.containers[(id(unit.owner), attr)])
        if isinstance(node, ast.Name):
            if node.id in names:
                return names[node.id]
            if node.id in self.bound.get(id(unit.node), set()):
                return None
            return self.module_names.get(unit.module.path, {}).get(node.id)
        attr = _self_attr(node)
        if attr is not None and unit.owner is not None and (id(unit.owner), attr) in self.declared:
            # The class owns this attribute, so what it binds decides -- not a same-named field elsewhere.
            return self.attrs.get((id(unit.owner), attr))
        if isinstance(node, ast.Attribute) and node.attr in self.fields:
            return frozenset(self.fields[node.attr])
        key = _string_key(node)
        if key is not None and key in self.keys:
            return frozenset(self.keys[key])
        return None

    def _settle(self, values: dict[str, list[ast.AST | tuple[int, str] | None]], read) -> dict[str, Kinds]:
        """Names whose every binding is a reading, given ``read`` for one value; iterated to a fixed point."""
        out: dict[str, Kinds] = {}
        for _ in range(10):
            changed = False
            for name, bound in values.items():
                kinds: set[str] = set()
                known = True
                for value in bound:
                    if value is None:
                        known = False
                        break
                    if isinstance(value, tuple):
                        passed = self.params.get(value)
                        if not passed:
                            known = False
                            break
                        kinds |= passed
                        continue
                    if _is_placeholder(value):
                        continue
                    k = read(out, value)
                    if k is None:
                        known = False
                        break
                    kinds |= k
                if known and kinds and out.get(name) != frozenset(kinds):
                    out[name] = frozenset(kinds)
                    changed = True
            if not changed:
                return out
        raise AssertionError("local clock propagation did not settle")

    @staticmethod
    def _bind_names(node: ast.AST, values: dict[str, list[ast.AST | tuple[int, str] | None]]) -> None:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    values.setdefault(target.id, []).append(node.value)
                elif isinstance(target, (ast.Tuple, ast.List)):
                    flat = all(isinstance(e, ast.Name) for e in target.elts)
                    for i, elt in enumerate(target.elts):
                        if flat:
                            values.setdefault(elt.id, []).append(_Unpacked(node.value, i))
                        else:
                            for name in ast.walk(elt):
                                if isinstance(name, ast.Name):
                                    values.setdefault(name.id, []).append(None)
        elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)) and isinstance(node.target, ast.Name):
            values.setdefault(node.target.id, []).append(node.value)
        elif isinstance(node, (ast.AugAssign, ast.For, ast.AsyncFor, ast.comprehension)):
            for elt in ast.walk(node.target):
                if isinstance(elt, ast.Name):
                    values.setdefault(elt.id, []).append(None)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    for elt in ast.walk(item.optional_vars):
                        if isinstance(elt, ast.Name):
                            values.setdefault(elt.id, []).append(None)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                values.setdefault((alias.asname or alias.name).split(".")[0], []).append(None)

    def _local_names(self, unit: _Unit) -> dict[str, Kinds]:
        values: dict[str, list[ast.AST | tuple[int, str] | None]] = {}
        for node in ast.walk(unit.node):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = node.args
                for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
                    values.setdefault(arg.arg, []).append((id(node), arg.arg))
                for arg in (args.vararg, args.kwarg):
                    if arg is not None:
                        values.setdefault(arg.arg, []).append(None)
            else:
                self._bind_names(node, values)
        self.bound[id(unit.node)] = set(values)

        def read(names: dict[str, Kinds], value) -> Kinds | None:
            if isinstance(value, _Unpacked):
                return self._element(unit, names, value.source, value.index, values)
            return self.kind(unit, names, value)

        return self._settle(values, read)

    def _element(self, unit: _Unit, names: dict[str, Kinds], source: ast.AST, index: int, values) -> Kinds | None:
        """The clock of element ``index`` of ``source``: a tuple, ``self.x[key]``/``self.x.get(key)`` filled
        with tuples, or a local name only ever bound to one of those."""
        if isinstance(source, ast.Tuple):
            return self.kind(unit, names, source.elts[index]) if index < len(source.elts) else None
        if isinstance(source, ast.Name):
            bound = values.get(source.id, [])
            kinds = [self._element(unit, names, v, index, values) if isinstance(v, ast.AST) else None for v in bound]
            return frozenset().union(*kinds) if kinds and all(kinds) else None
        attr = None
        if isinstance(source, ast.Subscript) and not isinstance(source.slice, ast.Constant):
            attr = _self_attr(source.value)
        elif (
            isinstance(source, ast.Call)
            and isinstance(source.func, ast.Attribute)
            and source.func.attr == "get"
            and source.args
            and not isinstance(source.args[0], ast.Constant)
        ):
            attr = _self_attr(source.func.value)
        if attr is None or unit.owner is None or (id(unit.owner), attr, index) not in self.container_items:
            return None
        return frozenset(self.container_items[(id(unit.owner), attr, index)])

    def _module_level_names(self, module: _Module) -> dict[str, Kinds]:
        values: dict[str, list[ast.AST | tuple[int, str] | None]] = {}
        for node in module.tree.body:
            self._bind_names(node, values)
        here = _Unit(module, ast.FunctionDef(name="<module>", args=ast.arguments(), body=[], decorator_list=[]), None)

        def read(names: dict[str, Kinds], value) -> Kinds | None:
            if isinstance(value, _Unpacked):
                return None
            clock = module.clocks.of_call(value)
            if clock:
                return frozenset({clock})
            if isinstance(value, ast.Name):
                return names.get(value.id)
            return self.kind(here, names, value) if not isinstance(value, ast.Name) else None

        return self._settle(values, read)

    # ── one propagation round ────────────────────────────────────────────────────────────────────
    def _propagate(self) -> None:
        for module in self.modules:
            self.module_names[module.path] = self._module_level_names(module)
        attr_values: dict[tuple[int, str], set[str] | None] = {}
        for module in self.modules:
            for cls in ast.walk(module.tree):
                if isinstance(cls, ast.ClassDef):
                    for stmt in cls.body:
                        if isinstance(stmt, (ast.AnnAssign, ast.Assign)):
                            for target in [stmt.target] if isinstance(stmt, ast.AnnAssign) else stmt.targets:
                                if isinstance(target, ast.Name):
                                    self.declared.add((id(cls), target.id))
                        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                            clock = _default_factory_clock(stmt.value, module.clocks)
                            if clock:
                                attr_values.setdefault((id(cls), stmt.target.id), set()).add(clock)
                                self.fields.setdefault(stmt.target.id, set()).add(clock)
        for unit in self.units:
            names = self.names[id(unit.node)] = self._local_names(unit)
            for fn in ast.walk(unit.node):
                if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    rets = [r.value for r in ast.walk(fn) if isinstance(r, ast.Return) and r.value is not None]
                    kinds = [self.kind(unit, names, v) for v in rets]
                    if rets and all(kinds):
                        self.returns[id(fn)] = frozenset().union(*kinds)
            for node in ast.walk(unit.node):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        self._bind(unit, names, target, node.value, attr_values)
                elif isinstance(node, ast.AnnAssign) and node.value is not None:
                    self._bind(unit, names, node.target, node.value, attr_values)
                elif isinstance(node, ast.Call):
                    if (
                        isinstance(node.func, ast.Attribute)
                        and node.func.attr == "setdefault"
                        and len(node.args) == 2
                        and isinstance(node.args[0], ast.Constant)
                        and isinstance(node.args[0].value, str)
                    ):
                        k = self.kind(unit, names, node.args[1])
                        if k:
                            self.keys.setdefault(node.args[0].value, set()).update(k)
                    for kw in node.keywords:
                        k = self.kind(unit, names, kw.value) if kw.arg else None
                        if k:
                            self.fields.setdefault(kw.arg, set()).update(k)
                    self._pass_arguments(unit, names, node)
        self.attrs = {key: frozenset(v) for key, v in attr_values.items() if v}

    def _bind(self, unit: _Unit, names: dict[str, Kinds], target: ast.AST, value: ast.AST, attr_values) -> None:
        if isinstance(target, ast.Attribute):
            k = None if _is_placeholder(value) else self.kind(unit, names, value)
            if k:
                self.fields.setdefault(target.attr, set()).update(k)
            attr = _self_attr(target)
            if attr is not None and unit.owner is not None:
                slot = (id(unit.owner), attr)
                self.declared.add(slot)
                if _is_placeholder(value):
                    attr_values.setdefault(slot, set())
                elif k is None or attr_values.get(slot, set()) is None:
                    attr_values[slot] = None
                else:
                    attr_values.setdefault(slot, set()).update(k)
            return
        key = _string_key(target)
        if key is not None:
            k = self.kind(unit, names, value)
            if k:
                self.keys.setdefault(key, set()).update(k)
            return
        if isinstance(target, ast.Subscript):
            attr = _self_attr(target.value)
            k = self.kind(unit, names, value)
            if attr is not None and unit.owner is not None and k:
                self.containers.setdefault((id(unit.owner), attr), set()).update(k)
            if attr is not None and unit.owner is not None and isinstance(value, ast.Tuple):
                for i, elt in enumerate(value.elts):
                    ek = self.kind(unit, names, elt)
                    if ek:
                        self.container_items.setdefault((id(unit.owner), attr, i), set()).update(ek)

    def _container_read(self, unit: _Unit, call: ast.Call) -> Kinds | None:
        """``self.x.get(key)`` with a computed key, where ``self.x[key] = <reading>`` fills it."""
        func = call.func
        if not (isinstance(func, ast.Attribute) and func.attr == "get" and call.args):
            return None
        if isinstance(call.args[0], ast.Constant):
            return None
        attr = _self_attr(func.value)
        if attr is None or unit.owner is None or (id(unit.owner), attr) not in self.containers:
            return None
        return frozenset(self.containers[(id(unit.owner), attr)])

    def _callees(self, unit: _Unit, call: ast.Call) -> list[tuple[Function, ast.ClassDef | None]]:
        """The functions this call can be, when that can be told; see ``_pass_arguments`` for the rest."""
        func = call.func
        name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else None
        found = self.defs.get(name or "", [])
        if len(found) == 1:
            return [(found[0][0], found[0][1])]
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id in ("self", "cls"):
            return [(f, o) for f, o, _ in found if o is not None and o is unit.owner]
        if isinstance(func, ast.Name):
            return [(f, o) for f, o, p in found if p == unit.module.path and o is None]
        return []

    def _pass_arguments(self, unit: _Unit, names: dict[str, Kinds], call: ast.Call) -> None:
        targets = self._callees(unit, call)
        if not targets and isinstance(call.func, ast.Attribute):
            # `store.expand(..., deadline=deadline)` on an interface with several implementations: a
            # keyword names the parameter, so it reaches every same-named method that declares it.
            # Positional arguments do not -- they would land on unrelated parameters.
            for target, _owner, _path in self.defs.get(call.func.attr, []):
                declared = {a.arg for a in [*target.args.posonlyargs, *target.args.args, *target.args.kwonlyargs]}
                for kw in call.keywords:
                    k = self.kind(unit, names, kw.value) if kw.arg in declared else None
                    if k:
                        self.params.setdefault((id(target), kw.arg), set()).update(k)
            return
        for target, owner in targets:
            params = [*target.args.posonlyargs, *target.args.args]
            if isinstance(call.func, ast.Attribute) and _is_method(target, owner):
                params = params[1:]
            passed: list[tuple[str, ast.AST]] = [(p.arg, a) for p, a in zip(params, call.args, strict=False)]
            known = {p.arg for p in [*params, *target.args.kwonlyargs]}
            passed += [(kw.arg, kw.value) for kw in call.keywords if kw.arg in known]
            for param, arg in passed:
                k = self.kind(unit, names, arg)
                if k:
                    self.params.setdefault((id(target), param), set()).update(k)

    def wall_clock_subtractions(self) -> list[tuple[str, int, str]]:
        offenders = []
        for unit in self.units:
            names = self.names[id(unit.node)]
            for node in ast.walk(unit.node):
                if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
                    sides = [node.left, node.right]
                elif isinstance(node, ast.Compare) and len(node.comparators) == 1:
                    sides = [node.left, node.comparators[0]]
                else:
                    continue
                left, right = (self.kind(unit, names, side) for side in sides)
                if left is not None and right is not None and WALL in left | right:
                    offenders.append((unit.module.path, node.lineno, ast.unparse(node)))
        return sorted(set(offenders))


def wall_clock_subtractions(sources: dict[str, str]) -> list[tuple[str, int, str]]:
    """(path, line, expression) of every subtraction that measures elapsed time against the wall clock."""
    return _Package(sources).wall_clock_subtractions()


def _package_sources() -> dict[str, str]:
    return {p.relative_to(PACKAGE_ROOT).as_posix(): p.read_text() for p in sorted(PACKAGE_ROOT.rglob("*.py"))}


@functools.cache
def _package_subtractions() -> list[tuple[str, int, str]]:
    """The whole-package analysis takes a few seconds; the tests below share one run."""
    return wall_clock_subtractions(_package_sources())


def test_no_elapsed_time_is_measured_on_the_wall_clock():
    offenders = [o for o in _package_subtractions() if (o[0], o[2]) not in EXEMPT]

    assert not offenders, (
        "These lines measure elapsed time against the wall clock (time.time() on one end or both), "
        "which a wall-clock step makes wrong or negative:\n  "
        + "\n  ".join(f"{path}:{line}  {expr}" for path, line, expr in offenders)
        + "\n\nRead both ends with time.perf_counter() instead, or -- if both ends really are points in "
        "time -- add the subtraction to EXEMPT with the reason."
    )


def test_every_exemption_still_applies():
    """An exemption whose subtraction is gone or no longer reads the wall clock is stale; drop it."""
    found = {(path, expr) for path, _, expr in _package_subtractions()}

    assert not set(EXEMPT) - found, f"Stale entries in EXEMPT: {sorted(set(EXEMPT) - found)}"


def test_the_check_reaches_the_request_scope_stamps():
    """The stamps are set in two files and subtracted in a third function; if the check stopped following
    string keys, reverting one of them would pass silently."""
    sources = _package_sources()
    sources["api/observability.py"] = sources["api/observability.py"].replace(
        'scope["hs_asgi_t0"] = time.perf_counter()', 'scope["hs_asgi_t0"] = time.time()'
    )

    assert any(path == "api/http.py" for path, _, _ in wall_clock_subtractions(sources))


def _offends(source: str) -> bool:
    return bool(wall_clock_subtractions({"m.py": source}))


@pytest.mark.parametrize(
    "source",
    [
        "import time\ndef f():\n    return time.time() - time.time()",
        "import time\ndef f():\n    t0 = time.time()\n    return time.time() - t0",
        "import time\ndef f():\n    t0 = time.time()\n    def done():\n        return time.time() - t0\n    return done",
        "import time\nclass C:\n    def start(self):\n        self.t0 = time.time()\n    def stop(self):\n        return time.time() - self.t0",
        "import time\nclass C:\n    def __init__(self, t0):\n        self.t0 = time.time()\n    def stop(self):\n        return time.time() - self.t0",
        "import time\nclass C:\n    def __init__(self):\n        self.t0 = None\n    def start(self):\n        self.t0 = time.time()\n    def stop(self):\n        return time.time() - self.t0",
        "import time\nclass C:\n    def __init__(self):\n        self.t0 = float('-inf')\n    def tick(self):\n        now = time.time()\n        if now - self.t0 > 30:\n            self.t0 = now",
        "import time\ndef f():\n    t0 = time.time()\n    return time.perf_counter() - t0",
        "import time\ndef f():\n    t0 = time.perf_counter()\n    return time.time() - t0",
        "import time\ndef f():\n    t0 = time.perf_counter()\n    a = time.perf_counter() - t0\n    t0 = time.time()\n    return time.perf_counter() - t0",
        "def f():\n    import time as tm\n    t0 = tm.time()\n    return tm.time() - t0",
        "from time import time\ndef f():\n    t0 = time()\n    return time() - t0",
        "import time\ndef took(start):\n    return time.perf_counter() - start\ndef f():\n    return took(time.time())",
        "import time\nclass C:\n    def _record(self, scope, start):\n        return time.perf_counter() - start\n    def call(self):\n        start = time.time()\n        return self._record('s', start)",
        "import time\ndef took(*, start_time):\n    return time.perf_counter() - start_time\ndef f():\n    return took(start_time=time.time())",
        "import time\ndef mw(scope):\n    scope['hs_t0'] = time.time()\ndef handler(request):\n    t0 = request.scope.get('hs_t0')\n    return time.perf_counter() - t0",
        "import time\n_START = time.time()\ndef uptime():\n    return time.monotonic() - _START",
        "import time\nclass Entry:\n    def __init__(self, created_at):\n        self.created_at = created_at\ndef put():\n    return Entry(created_at=time.time())\ndef age(entry):\n    return time.monotonic() - entry.created_at",
        "import time\nfrom dataclasses import dataclass, field\n@dataclass\nclass T:\n    started: float = field(default_factory=time.time)\n    def wall(self):\n        return time.perf_counter() - self.started",
        "import time\nclass S:\n    def due(self, job, every):\n        now = time.time()\n        last = self._last_run.get(job)\n        if last is not None and now - last < every:\n            return False\n        self._last_run[job] = now\n        return True",
        "import time\nclass C:\n    def _now(self):\n        return time.time()\n    def took(self, start):\n        return self._now() - start\n    def go(self):\n        return self.took(self._now())",
        "import time\ndef wait(timeout):\n    deadline = time.time() + timeout\n    while time.time() < deadline:\n        pass",
        "import time\ndef wait(timeout):\n    deadline = time.monotonic() + timeout\n    while time.time() < deadline:\n        pass",
        "import time\ndef wait(t):\n    deadline = time.time() + t if t > 0 else None\n    if deadline is not None and time.time() >= deadline:\n        return",
        "import time\nclass Cache:\n    def _now(self):\n        return time.time()\n    def put(self, k, v):\n        self._entries[k] = (self._now() + 60, v)\n    def get(self, k):\n        entry = self._entries.get(k)\n        expires_at, value = entry\n        if expires_at <= time.monotonic():\n            return None\n        return value",
        "import time\nclass Base:\n    def expand(self, ids, deadline=None):\n        raise NotImplementedError\nclass Pg(Base):\n    def expand(self, ids, deadline=None):\n        if deadline is not None and time.monotonic() >= deadline:\n            return\ndef job(store):\n    store.expand([1], deadline=time.time() + 5)",
    ],
    ids=[
        "inline",
        "local",
        "closure",
        "attribute",
        "attribute-named-like-a-parameter",
        "attribute-with-a-none-placeholder",
        "attribute-from-a-local-with-an-inf-placeholder",
        "mixed-wall-start",
        "mixed-wall-end",
        "one-of-several-bindings-reverted",
        "module-alias",
        "from-import",
        "positional-parameter",
        "method-parameter",
        "keyword-parameter",
        "string-key-across-functions",
        "module-level-name",
        "keyword-field-read-off-another-object",
        "dataclass-default-factory",
        "dict-keyed-by-a-variable",
        "function-returning-a-reading",
        "deadline-on-the-wall-clock",
        "deadline-mixed-clocks",
        "conditional-deadline",
        "keyword-to-an-interface-method",
        "expiry-kept-in-a-tuple",
    ],
)
def test_the_check_finds_a_wall_clock_duration(source):
    assert _offends(source)


@pytest.mark.parametrize(
    "source",
    [
        "import time\ndef f():\n    t0 = time.perf_counter()\n    return time.perf_counter() - t0",
        "import time\ndef f():\n    t0 = time.monotonic()\n    return time.monotonic() - t0",
        "import time\ndef f(expires_at):\n    return expires_at - time.time()",
        "import time, os\ndef f(path):\n    return time.time() - os.stat(path).st_mtime",
        "import time\ndef f(stamp):\n    t0 = time.time()\n    t0 = stamp\n    return time.time() - t0",
        "import time\ndef took(start):\n    return time.perf_counter() - start\ndef f():\n    return took(time.perf_counter())",
        "import time\ndef mw(scope):\n    scope['hs_t0'] = time.perf_counter()\ndef handler(request):\n    t0 = request.scope.get('hs_t0')\n    return time.perf_counter() - t0",
        # Two classes with a `_record` method each: a call in one must not reach the other's parameters.
        "import time\ndef wait(timeout):\n    deadline = time.monotonic() + timeout\n    while time.monotonic() < deadline:\n        pass",
        "import time\ndef fresh(token):\n    return token.expires_at > time.time()",
        "import time\nclass A:\n    def _record(self, start):\n        return time.perf_counter() - start\n    def go(self):\n        return self._record(time.perf_counter())\nclass B:\n    def _record(self, at):\n        self.events = [at]\n    def login(self):\n        self._record(time.time())",
    ],
    ids=[
        "perf-counter",
        "monotonic",
        "expiry",
        "mtime",
        "rebound-name",
        "parameter",
        "string-key",
        "monotonic-deadline",
        "persisted-expiry-compared-with-now",
        "same-method-name-in-another-class",
    ],
)
def test_the_check_leaves_points_in_time_alone(source):
    assert not _offends(source)
