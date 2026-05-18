from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Violation:
    rule_id: str
    line: int
    col: int
    message: str


def check_source(source: str) -> list[Violation]:
    tree = ast.parse(source)
    checker = _StateflowChecker()
    checker.visit(tree)
    return checker.violations


def check_path(path: Path | str) -> list[Violation]:
    p = Path(path)
    return check_source(p.read_text(encoding="utf-8"))


def _is_dbos_workflow_decorator(decorator: ast.expr) -> bool:
    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    if isinstance(target, ast.Attribute):
        return (
            isinstance(target.value, ast.Name)
            and target.value.id == "DBOS"
            and target.attr == "workflow"
        )
    return False


def _is_dbos_step_decorator(decorator: ast.expr) -> bool:
    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    if isinstance(target, ast.Attribute):
        return (
            isinstance(target.value, ast.Name)
            and target.value.id == "DBOS"
            and target.attr in ("step", "transaction")
        )
    return False


def _matches_module_call(
    call: ast.Call, modules: tuple[str, ...], attrs: set[str] | None = None
) -> bool:
    func = call.func
    if not isinstance(func, ast.Attribute):
        return False
    if not isinstance(func.value, ast.Name):
        return False
    if func.value.id not in modules:
        return False
    return not (attrs is not None and func.attr not in attrs)


class _StateflowChecker(ast.NodeVisitor):
    def __init__(self) -> None:
        self.violations: list[Violation] = []
        self._in_workflow: int = 0

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_func(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_func(node)

    def _visit_func(self, node: ast.AsyncFunctionDef | ast.FunctionDef) -> None:
        in_workflow = any(_is_dbos_workflow_decorator(d) for d in node.decorator_list)
        in_step = any(_is_dbos_step_decorator(d) for d in node.decorator_list)
        if in_workflow and not in_step:
            self._in_workflow += 1
            try:
                self.generic_visit(node)
            finally:
                self._in_workflow -= 1
        else:
            self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if self._in_workflow > 0:
            self._check_call(node)
        self.generic_visit(node)

    def _check_call(self, node: ast.Call) -> None:
        # STATEFLOW001 — datetime.now / datetime.utcnow
        if _matches_module_call(node, ("datetime",), {"now", "utcnow"}):
            self.violations.append(Violation(
                rule_id="STATEFLOW001",
                line=node.lineno, col=node.col_offset,
                message="datetime.now()/utcnow() outside @DBOS.step inside a workflow",
            ))
        # STATEFLOW002 — time.time / time.monotonic
        if _matches_module_call(node, ("time",), {"time", "monotonic", "perf_counter"}):
            self.violations.append(Violation(
                rule_id="STATEFLOW002",
                line=node.lineno, col=node.col_offset,
                message="time.time()/monotonic() outside @DBOS.step inside a workflow",
            ))
        # STATEFLOW003 — httpx.* / requests.*
        if _matches_module_call(node, ("httpx", "requests")):
            self.violations.append(Violation(
                rule_id="STATEFLOW003",
                line=node.lineno, col=node.col_offset,
                message="HTTP call outside @DBOS.step inside a workflow",
            ))
        # STATEFLOW004 — random.*
        if _matches_module_call(node, ("random",)):
            self.violations.append(Violation(
                rule_id="STATEFLOW004",
                line=node.lineno, col=node.col_offset,
                message="random.* outside @DBOS.step inside a workflow",
            ))
        # STATEFLOW005 — asyncio.sleep
        if _matches_module_call(node, ("asyncio",), {"sleep"}):
            self.violations.append(Violation(
                rule_id="STATEFLOW005",
                line=node.lineno, col=node.col_offset,
                message="asyncio.sleep() in workflow (use DBOS.sleep() instead)",
            ))
