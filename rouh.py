from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Dict

import ast
import math


# ---- file/path safety -------------------------------------------------

# skills can only be read from THIS directory (adjust if needed)
_ALLOWED_SKILLS_DIR = Path(__file__).resolve().parent


def _safe_resolve_skill_path(path: str | Path) -> Path:
    """
    Resolve a user-provided path and make sure it stays inside the
    allowed skills directory. This is what SAST wants for 'dynamic input in file path'.
    """
    base = _ALLOWED_SKILLS_DIR
    p = Path(path)

    # treat relative paths as relative to our base
    if not p.is_absolute():
        p = base / p

    resolved = p.resolve()

    # containment check – prevents ../../.. traversal
    if base not in resolved.parents and resolved != base:
        raise ValueError(f"Skill path {resolved} is outside allowed directory {base}")

    return resolved


# ---- tiny safe evaluators ---------------------------------------------
# We need this because SAST doesn't like exec/eval on user input.

_ALLOWED_MATH_FUNCS: Dict[str, Callable[..., Any]] = {
    # add more if you actually need them
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "pow": pow,
    "abs": abs,
}

_ALLOWED_CONSTS: Dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
}


class _SafeMathEval(ast.NodeVisitor):
    """
    Very small, very strict evaluator for math-like expressions.
    Supports: + - * / **, unary +/-, parentheses, calls to allowed math fns,
    and names from _ALLOWED_CONSTS.
    """

    def visit(self, node: ast.AST) -> Any:  # type: ignore[override]
        if isinstance(node, ast.Expression):
            return self.visit(node.body)

        # numbers
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value

        # names (pi, e)
        if isinstance(node, ast.Name):
            if node.id in _ALLOWED_CONSTS:
                return _ALLOWED_CONSTS[node.id]
            raise ValueError(f"name {node.id!r} not allowed")

        # binops
        if isinstance(node, ast.BinOp):
            left = self.visit(node.left)
            right = self.visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left ** right
            raise ValueError("operation not allowed")

        # unary ops
        if isinstance(node, ast.UnaryOp):
            operand = self.visit(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +operand
            if isinstance(node.op, ast.USub):
                return -operand
            raise ValueError("unary operation not allowed")

        # calls: only allowed math funcs, positional args only
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("only simple calls allowed")
            fn_name = node.func.id
            if fn_name not in _ALLOWED_MATH_FUNCS:
                raise ValueError(f"function {fn_name!r} not allowed")
            args = [self.visit(a) for a in node.args]
            return _ALLOWED_MATH_FUNCS[fn_name](*args)

        # everything else: no.
        raise ValueError(f"expression type {type(node).__name__} not allowed")


def safe_eval_math(expr: str) -> float:
    """
    Parse math-like expression safely, without eval().
    This replaces:
        eval(expression, {"__builtins__": {}}, math.__dict__)
    which SAST flags.
    """
    parsed = ast.parse(expr, mode="eval")
    return float(_SafeMathEval().visit(parsed))


# If you ALSO want to support "code execution" as a skill, make it template-ish,
# not exec-ish.
def safe_code_exec_snippet(code: str) -> str:
    """
    Super-conservative replacement for the old exec(...) sandbox.
    Instead of executing arbitrary Python, we only allow:
    - print-like templates: 'echo: something'
    - or math-like expressions.

    Adjust to your use-case, but keep a clear allowlist – SAST will like that.
    """
    code = code.strip()

    # pattern 1: pretend "echo: hi"
    if code.startswith("echo:"):
        return code.split(":", 1)[1].strip()

    # pattern 2: try math
    try:
        value = safe_eval_math(code)
        return str(value)
    except Exception:
        # don't leak Python internals
        raise ValueError("code snippet not allowed / not understood")


# ---- your registry / skills -------------------------------------------

class SkillRegistry:
    def __init__(self) -> None:
        self._skills: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, fn: Callable[..., Any]) -> None:
        self._skills[name] = fn

    def get(self, name: str) -> Callable[..., Any]:
        return self._skills[name]

    def list(self) -> Mapping[str, Callable[..., Any]]:
        return dict(self._skills)


def search_stub(query: str) -> str:
    return f"Search results for {query!r}"


def web_fetch(url: str) -> str:
    # you can add URL validation here if the scanner later complains.
    return f"Fetched content from {url!r}"


def code_exec_sandbox(code: str) -> str:
    """
    OLD VERSION (flagged):
        allowed = {"__builtins__": {...}}
        exec(code, allowed, allowed)

    NEW VERSION:
        route to a safe, allowlisted interpreter.
    """
    return safe_code_exec_snippet(code)


def math_solver(expression: str) -> float:
    """
    OLD VERSION (flagged):
        return float(eval(expression, {"__builtins__": {}}, math.__dict__))

    NEW VERSION:
        AST-based evaluator – no eval(), no builtins.
    """
    return safe_eval_math(expression)


def rag_query(index: Mapping[str, str], query: str, k: int = 3) -> list[str]:
    # keeping your earlier stub
    return list(index.values())[:k]


def kb_lookup(entity: str) -> str:
    return f"Info about {entity!r}"


# Example of a skill that *loads* from disk but sanitizes the path.
def load_skill_from_file(path: str) -> str:
    """
    This is to fix the L68 'unsanitized dynamic input in file path found'.
    """
    safe_path = _safe_resolve_skill_path(path)
    return safe_path.read_text(encoding="utf-8")
