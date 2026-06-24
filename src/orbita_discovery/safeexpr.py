"""Small safe expression evaluator for declarative tabular hypotheses."""
from __future__ import annotations

import ast
import operator
from typing import Any

_BIN = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg, ast.Not: operator.not_}
_CMP = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}
_FUNCS = {"abs": abs, "min": min, "max": max, "len": len, "round": round}


class UnsafeExpression(ValueError):
    pass


def evaluate(expression: str, record: dict[str, Any]) -> Any:
    tree = ast.parse(expression, mode="eval")
    return _eval(tree.body, record)


def _eval(node: ast.AST, env: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        if node.id in _FUNCS:
            return _FUNCS[node.id]
        raise UnsafeExpression(f"Unknown name: {node.id}")
    if isinstance(node, ast.List):
        return [_eval(x, env) for x in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_eval(x, env) for x in node.elts)
    if isinstance(node, ast.Dict):
        return {_eval(k, env): _eval(v, env) for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        return _BIN[type(node.op)](_eval(node.left, env), _eval(node.right, env))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand, env))
    if isinstance(node, ast.BoolOp):
        values = [_eval(x, env) for x in node.values]
        return all(values) if isinstance(node.op, ast.And) else any(values)
    if isinstance(node, ast.Compare):
        left = _eval(node.left, env)
        for op_node, comp_node in zip(node.ops, node.comparators):
            right = _eval(comp_node, env)
            fn = _CMP.get(type(op_node))
            if fn is None or not fn(left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        fn = _FUNCS[node.func.id]
        args = [_eval(a, env) for a in node.args]
        kwargs = {kw.arg: _eval(kw.value, env) for kw in node.keywords if kw.arg}
        return fn(*args, **kwargs)
    if isinstance(node, ast.Subscript):
        value = _eval(node.value, env)
        key = _eval(node.slice, env)
        return value[key]
    raise UnsafeExpression(f"Disallowed syntax: {type(node).__name__}")
