"""Deterministic AST-to-AST refactoring engine (Phase 2, FR03, NFR01).

Applies real, semantics-preserving source-to-source transformations using
`ast.NodeTransformer`, `copy.deepcopy`, and `ast.unparse` — no runtime
execution, no placeholders, deterministic output:

    fix-mutable-defaults
        Replace list/dict/set defaults with None + an in-function guard
        (the canonical Fix F841-style mutable-default refactoring).

    merge-nested-conditionals
        Collapse `if a: if b:` chains into a single short-circuited
        `if a and b:` (only when no else-branch would change semantics).

    extract-function-helpers
        Extract provably independent statement runs into `_name_chunk`
        helpers that receive exactly the locals they read.  When no run is
        provably safe, fall back to a wholesale helper extraction that
        preserves signatures, defaults, decorators, async-ness, and
        generator behaviour.

The engine is shared by MockModelProvider (deterministic test provider) and
SmartASTProvider (CPU fallback used when no neural model is available).
"""

import ast
import copy
from typing import Any

from core.schemas import ASTFinding

_FUNCTION_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef)
_SCOPE_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
_MUTABLE_LITERALS = (ast.List, ast.Dict, ast.Set, ast.ListComp, ast.DictComp, ast.SetComp)
_MUTABLE_CALL_NAMES = {"list", "dict", "set", "bytearray"}

_RUN_BLOCKERS = (
    ast.Return,
    ast.Raise,
    ast.Break,
    ast.Continue,
    ast.Yield,
    ast.YieldFrom,
    ast.Nonlocal,
    ast.Global,
)

_DOC_SUFFIX = "_chunk"


# --------------------------------------------------------------------------- #
# AST helpers
# --------------------------------------------------------------------------- #
def _iter_scope_only(node: ast.AST):
    """Yield *node* and its descendants, never descending into nested scopes."""
    stack: list[tuple[ast.AST, bool]] = [(node, True)]
    while stack:
        current, is_root = stack.pop()
        yield current
        if not is_root and isinstance(current, _SCOPE_TYPES):
            continue
        for child in ast.iter_child_nodes(current):
            stack.append((child, False))


def _count_nodes(node: ast.AST) -> int:
    return sum(1 for _ in ast.walk(node))


def _scan_names(node: ast.AST) -> tuple[set[str], set[str]]:
    """Return (reads, writes) of a statement.

    Reads include closure reads from nested scopes (conservative: passing an
    extra argument can never break behaviour).  Writes are limited to the
    statement's own scope plus nested definition names so that extraction
    only happens when locals provably do not leak.
    """
    reads: set[str] = set()
    writes: set[str] = set()

    for current in ast.walk(node):
        if isinstance(current, ast.Name) and isinstance(current.ctx, ast.Load):
            reads.add(current.id)

    for current in _iter_scope_only(node):
        if current is not node and isinstance(current, _SCOPE_TYPES):
            if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                writes.add(current.name)
            continue
        if isinstance(current, ast.Name) and isinstance(current.ctx, (ast.Store, ast.Del)):
            writes.add(current.id)
        elif isinstance(current, (ast.Import, ast.ImportFrom)):
            for alias in current.names:
                writes.add((alias.asname or alias.name).split(".")[0])

    return reads, writes


def _contains_blockers(node: ast.AST) -> bool:
    return any(isinstance(item, _RUN_BLOCKERS) for item in _iter_scope_only(node))


def _contains_await(node: ast.AST) -> bool:
    return any(isinstance(item, ast.Await) for item in _iter_scope_only(node))


def _parameter_names(func: ast.AST) -> list[str]:
    if not isinstance(func, _FUNCTION_TYPES):
        return []
    names = [arg.arg for arg in list(func.args.posonlyargs) + list(func.args.args)]
    names.extend(arg.arg for arg in func.args.kwonlyargs)
    if func.args.vararg:
        names.append(func.args.vararg.arg)
    if func.args.kwarg:
        names.append(func.args.kwarg.arg)
    return names


def _build_call(name: str, func_args: ast.arguments) -> ast.Call:
    """Build `name(a, b, *args, kw=..., **kwargs)` mirroring a signature."""
    positional: list[ast.expr] = [
        ast.Name(id=arg.arg, ctx=ast.Load())
        for arg in list(func_args.posonlyargs) + list(func_args.args)
    ]
    keywords: list[ast.keyword] = [
        ast.keyword(arg=arg.arg, value=ast.Name(id=arg.arg, ctx=ast.Load()))
        for arg in func_args.kwonlyargs
    ]
    if func_args.vararg:
        positional.append(
            ast.Starred(value=ast.Name(id=func_args.vararg.arg, ctx=ast.Load()), ctx=ast.Load())
        )
    if func_args.kwarg:
        keywords.append(
            ast.keyword(arg=None, value=ast.Name(id=func_args.kwarg.arg, ctx=ast.Load()))
        )
    return ast.Call(func=ast.Name(id=name, ctx=ast.Load()), args=positional, keywords=keywords)


def _collect_defined_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.arg):
            names.add(node.arg)
    return names


def _docstring_statement(func: ast.AST) -> ast.stmt | None:
    if not isinstance(func, _FUNCTION_TYPES) or not func.body:
        return None
    first = func.body[0]
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        return copy.deepcopy(first)
    return None


# --------------------------------------------------------------------------- #
# Transformation 1: mutable default arguments
# --------------------------------------------------------------------------- #
class _MutableDefaultFixer(ast.NodeTransformer):
    """Replace mutable defaults with None and prepend `if x is None: x = ...` guards."""

    def __init__(self) -> None:
        self.count = 0

    @staticmethod
    def _is_mutable(node: ast.AST) -> bool:
        if isinstance(node, _MUTABLE_LITERALS):
            return True
        return bool(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _MUTABLE_CALL_NAMES
            and not node.args
            and not node.keywords
        )

    def _process(self, node: ast.AST) -> ast.AST:
        self.generic_visit(node)
        assert isinstance(node, _FUNCTION_TYPES)
        positional = list(node.args.posonlyargs) + list(node.args.args)
        defaults = node.args.defaults
        offset = len(positional) - len(defaults)
        guards: list[tuple[str, ast.AST]] = []
        new_defaults: list[ast.expr] = []
        for index, default in enumerate(defaults):
            if self._is_mutable(default):
                guards.append((positional[offset + index].arg, default))
                new_defaults.append(ast.Constant(value=None))
                self.count += 1
            else:
                new_defaults.append(default)
        node.args.defaults = new_defaults

        new_kw_defaults: list[ast.expr | None] = []
        for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults, strict=False):
            if default is not None and self._is_mutable(default):
                guards.append((arg.arg, default))
                new_kw_defaults.append(ast.Constant(value=None))
                self.count += 1
            else:
                new_kw_defaults.append(default)
        node.args.kw_defaults = new_kw_defaults

        if guards:
            guard_stmts: list[ast.stmt] = []
            for name, default in guards:
                source = f"if {name} is None:\n    {name} = {ast.unparse(default)}"
                guard_stmts.extend(ast.parse(source).body)
            insert_at = 1 if _docstring_statement(node) is not None else 0
            node.body[insert_at:insert_at] = guard_stmts
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        return self._process(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        return self._process(node)


def fix_mutable_defaults(code: str) -> tuple[str, int]:
    """Return (transformed_code, guard_count)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, 0
    fixer = _MutableDefaultFixer()
    transformed = fixer.visit(tree)
    if not fixer.count:
        return code, 0
    ast.fix_missing_locations(transformed)
    return ast.unparse(transformed), fixer.count


# --------------------------------------------------------------------------- #
# Transformation 2: nested conditional merging
# --------------------------------------------------------------------------- #
class _NestedIfMerger(ast.NodeTransformer):
    """Collapse `if a: if b: body` chains without else branches."""

    def __init__(self) -> None:
        self.count = 0

    @staticmethod
    def _flatten_and(left: ast.expr, right: ast.expr) -> ast.BoolOp:
        values: list[ast.expr] = []
        for side in (left, right):
            if isinstance(side, ast.BoolOp) and isinstance(side.op, ast.And):
                values.extend(side.values)
            else:
                values.append(side)
        return ast.BoolOp(op=ast.And(), values=values)

    def _merge_once(self, node: ast.If) -> bool:
        if len(node.body) != 1:
            return False
        inner = node.body[0]
        if not isinstance(inner, ast.If) or node.orelse or inner.orelse:
            return False
        node.test = self._flatten_and(node.test, inner.test)
        node.body = inner.body
        self.count += 1
        return True

    def visit_If(self, node: ast.If) -> ast.AST:
        self.generic_visit(node)
        while self._merge_once(node):
            pass
        return node


def merge_nested_conditionals(code: str) -> tuple[str, int]:
    """Return (transformed_code, merge_count)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, 0
    merger = _NestedIfMerger()
    transformed = merger.visit(tree)
    if not merger.count:
        return code, 0
    ast.fix_missing_locations(transformed)
    return ast.unparse(transformed), merger.count


# --------------------------------------------------------------------------- #
# Transformation 3: long function decomposition
# --------------------------------------------------------------------------- #
def _is_extractable_statement(stmt: ast.stmt) -> bool:
    if isinstance(
        stmt,
        (
            ast.Return,
            ast.Raise,
            ast.Break,
            ast.Continue,
            ast.Yield,
            ast.YieldFrom,
            ast.Nonlocal,
            ast.Global,
            ast.Import,
            ast.ImportFrom,
        ),
    ):
        return False
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
        # Bare strings / literals carry no behaviour worth extracting.
        return False
    if _contains_blockers(stmt):
        return False
    return True


def _find_safe_run(
    body: list[ast.stmt],
    min_statements: int,
    min_nodes: int,
) -> tuple[int, int] | None:
    """Locate the largest provably independent consecutive statement run.

    A run is safe when the names it writes are never read by statements
    after it, and it contains no control-flow terminators.  Reads outside the
    run are supplied to the helper as parameters, preserving behaviour.
    """
    suffix_reads: list[set[str]] = [set() for _ in range(len(body) + 1)]
    for index in range(len(body) - 1, -1, -1):
        reads, _ = _scan_names(body[index])
        suffix_reads[index] = suffix_reads[index + 1] | reads

    best: tuple[int, int] | None = None
    best_nodes = -1
    index = 0
    while index < len(body):
        if not _is_extractable_statement(body[index]):
            index += 1
            continue
        run_writes: set[str] = set()
        run_nodes = 0
        end = index
        while end < len(body) and _is_extractable_statement(body[end]):
            reads, writes = _scan_names(body[end])
            if (run_writes | writes) & suffix_reads[end + 1]:
                break
            run_writes |= writes
            run_nodes += _count_nodes(body[end])
            end += 1
        if end - index >= min_statements and run_nodes >= min_nodes and run_nodes > best_nodes:
            best = (index, end)
            best_nodes = run_nodes
        index = end if end > index else index + 1
    return best


def _run_parameter_names(
    func: ast.AST, body: list[ast.stmt], start: int, run: list[ast.stmt]
) -> list[str]:
    """Names the run reads that are bound before it (signature + earlier writes)."""
    available = set(_parameter_names(func))
    for stmt in body[:start]:
        _, writes = _scan_names(stmt)
        available |= writes
    needed: list[str] = []
    seen: set[str] = set()
    for stmt in run:
        reads, _ = _scan_names(stmt)
        for name in sorted(reads):
            if name in available and name not in seen:
                seen.add(name)
                needed.append(name)
    return needed


def _build_chunk_helper(
    name: str, parameters: list[str], statements: list[ast.stmt], is_async: bool
) -> ast.stmt:
    helper_cls = ast.AsyncFunctionDef if is_async else ast.FunctionDef
    helper = helper_cls(
        name=name,
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=param) for param in parameters],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        ),
        body=copy.deepcopy(statements),
        decorator_list=[],
        returns=None,
        type_comment=None,
    )
    doc = ast.Expr(
        value=ast.Constant(
            value=f"Extracted helper reducing the size of its parent function ({name})."
        )
    )
    helper.body.insert(0, doc)
    return helper


def _can_wholesale_extract(func: ast.AST) -> bool:
    """Reject bodies whose behaviour depends on their enclosing scope."""
    assert isinstance(func, _FUNCTION_TYPES)
    for node in ast.walk(func):
        if isinstance(node, ast.Nonlocal):
            return False
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "super" and not node.args and not node.keywords:
                return False
        if isinstance(node, ast.Name) and node.id == "__class__":
            return False
    if isinstance(func, ast.AsyncFunctionDef) and any(
        isinstance(node, (ast.Yield, ast.YieldFrom)) for node in ast.walk(func)
    ):
        # Async generators cannot be awaited without changing the return type.
        return False
    return True


def _wholesale_extract(func: ast.AST, helper_name: str) -> ast.stmt:
    """Move a full function body into a module-level helper, preserving semantics."""
    assert isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef))
    is_async = isinstance(func, ast.AsyncFunctionDef)
    helper_cls = ast.AsyncFunctionDef if is_async else ast.FunctionDef
    helper = helper_cls(
        name=helper_name,
        args=copy.deepcopy(func.args),
        body=copy.deepcopy(func.body),
        decorator_list=[],
        returns=copy.deepcopy(func.returns),
        type_comment=getattr(func, "type_comment", None),
    )
    call = _build_call(helper_name, func.args)
    if is_async:
        call = ast.Await(value=call)
    new_body: list[ast.stmt] = []
    doc = _docstring_statement(func)
    if doc is not None:
        new_body.append(doc)
    new_body.append(ast.Return(value=call))
    func.body = new_body
    ast.copy_location(helper, func)
    return helper


def _extract_from_function(
    func: ast.AST,
    make_name,
    max_helpers: int,
    min_run_statements: int,
    min_run_nodes: int,
) -> list[ast.stmt]:
    helpers: list[ast.stmt] = []
    while len(helpers) < max_helpers:
        run = _find_safe_run(func.body, min_run_statements, min_run_nodes)
        if run is None:
            break
        start, end = run
        run_statements = func.body[start:end]
        parameters = _run_parameter_names(func, func.body, start, run_statements)
        is_async_run = any(_contains_await(stmt) for stmt in run_statements)
        base = (
            f"_{func.name}{_DOC_SUFFIX}"
            if not helpers
            else f"_{func.name}{_DOC_SUFFIX}_{len(helpers) + 1}"
        )
        helper_name = make_name(base)
        helpers.append(_build_chunk_helper(helper_name, parameters, run_statements, is_async_run))
        call: ast.expr = ast.Call(
            func=ast.Name(id=helper_name, ctx=ast.Load()),
            args=[ast.Name(id=param, ctx=ast.Load()) for param in parameters],
            keywords=[],
        )
        if is_async_run:
            call = ast.Await(value=call)
        replacement = ast.Expr(value=call)
        ast.copy_location(replacement, run_statements[0])
        func.body = func.body[:start] + [replacement] + func.body[end:]

    if helpers:
        return helpers
    if _can_wholesale_extract(func):
        helper_name = make_name(f"_{func.name}{_DOC_SUFFIX}")
        helpers.append(_wholesale_extract(func, helper_name))
    return helpers


def extract_long_functions(
    code: str,
    max_function_lines: int = 30,
    max_helpers_per_function: int = 4,
    min_run_statements: int = 2,
    min_run_nodes: int = 6,
) -> tuple[str, int, list[str]]:
    """Decompose over-long top-level functions and class methods.

    Returns (transformed_code, helper_count, helper_names).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, 0, []

    defined = _collect_defined_names(tree)

    def make_name(base: str) -> str:
        candidate = base
        suffix = 2
        while candidate in defined:
            candidate = f"{base}_{suffix}"
            suffix += 1
        defined.add(candidate)
        return candidate

    module_body = tree.body
    insertions: dict[int, list[ast.stmt]] = {}
    helper_names: list[str] = []

    for index, statement in enumerate(module_body):
        targets: list[ast.AST] = []
        if isinstance(statement, _FUNCTION_TYPES):
            targets.append(statement)
        elif isinstance(statement, ast.ClassDef):
            targets.extend(
                member for member in statement.body if isinstance(member, _FUNCTION_TYPES)
            )
        for func in targets:
            length = (func.end_lineno or func.lineno) - func.lineno + 1
            if length <= max_function_lines:
                continue
            helpers = _extract_from_function(
                func, make_name, max_helpers_per_function, min_run_statements, min_run_nodes
            )
            if helpers:
                insertions.setdefault(index, []).extend(helpers)
                helper_names.extend(helper.name for helper in helpers)

    if not insertions:
        return code, 0, []

    new_body: list[ast.stmt] = []
    for index, statement in enumerate(module_body):
        new_body.extend(insertions.get(index, []))
        new_body.append(statement)
    tree.body = new_body
    ast.fix_missing_locations(tree)
    return ast.unparse(tree), len(helper_names), helper_names


def _default_cleanup(code: str) -> str:
    """Add a module docstring when the module documents nothing at all."""
    stripped = code.strip()
    if not stripped:
        return code
    if '"""' in code or "'''" in code:
        return code + "\n"
    return f'"""Refactored module: structural cleanup and documentation pass."""\n\n{stripped}\n'


# --------------------------------------------------------------------------- #
# Engine facade
# --------------------------------------------------------------------------- #
class ASTRefactoringEngine:
    """Deterministic, rule-guided source-to-source refactoring engine."""

    def __init__(
        self,
        max_function_lines: int = 30,
        fix_mutable_defaults_enabled: bool = True,
        merge_nested_conditionals_enabled: bool = True,
        extract_helpers_enabled: bool = True,
    ) -> None:
        self.max_function_lines = max_function_lines
        self.fix_mutable_defaults_enabled = fix_mutable_defaults_enabled
        self.merge_nested_conditionals_enabled = merge_nested_conditionals_enabled
        self.extract_helpers_enabled = extract_helpers_enabled

    def transform(self, code: str, findings: list[ASTFinding]) -> tuple[str, list[dict[str, Any]]]:
        """Apply transformations targeting the rules present in *findings*.

        Returns (transformed_code, applied_steps) where every step records the
        transformation name and how many sites changed.
        """
        rules = {finding.rule for finding in findings}
        current = code
        steps: list[dict[str, Any]] = []

        if self.fix_mutable_defaults_enabled and "mutable-default-arg" in rules:
            current, mutated = fix_mutable_defaults(current)
            if mutated:
                steps.append({"transform": "fix-mutable-defaults", "count": mutated})

        if self.merge_nested_conditionals_enabled and rules & {
            "deep-nesting",
            "high-complexity",
            "high-cognitive-complexity",
        }:
            current, merged = merge_nested_conditionals(current)
            if merged:
                steps.append({"transform": "merge-nested-conditionals", "count": merged})

        if self.extract_helpers_enabled and "long-function" in rules:
            current, extracted, helper_names = extract_long_functions(
                current, self.max_function_lines
            )
            if extracted:
                steps.append(
                    {
                        "transform": "extract-function-helpers",
                        "count": extracted,
                        "helpers": helper_names,
                    }
                )

        if not steps:
            if rules:
                # No deterministic transformation is provably safe for these
                # rules; return the input untouched so the Review Agent can
                # judge it honestly instead of pretending to have fixed it.
                return current, []
            current = _default_cleanup(current)
            steps.append({"transform": "structural-cleanup", "count": 1})

        return current, steps
