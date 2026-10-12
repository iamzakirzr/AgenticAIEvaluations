"""Tools for the reference agent: a calculator, doc search and an order lookup."""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from evalkit.sut.rag_app.retriever import TfidfRetriever, load_corpus


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema
    fn: Callable[..., str]

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def calculate(expression: str) -> str:
    """Arithmetic only. eval() on model output is how agents get owned."""

    def walk(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            if isinstance(node.op, ast.Pow) and abs(walk(node.right)) > 10:
                raise ValueError("exponent too large")
            return _OPS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](walk(node.operand))
        raise ValueError(f"unsupported expression element: {type(node).__name__}")

    try:
        value = walk(ast.parse(expression, mode="eval").body)
    except (SyntaxError, ValueError, ZeroDivisionError) as exc:
        return f"error: {exc}"
    return f"{value:g}" if isinstance(value, float) else str(value)


# Fictional billing records. Tests reference these ids.
ORDERS: dict[str, dict[str, Any]] = {
    "INV-1001": {
        "plan": "Pro",
        "billing": "monthly",
        "seats": 1,
        "amount_usd": 8.0,
        "charged_on": "2026-10-01",
        "status": "paid",
    },
    "INV-1002": {
        "plan": "Team",
        "billing": "annual",
        "seats": 5,
        "amount_usd": 672.0,
        "charged_on": "2026-08-20",
        "status": "paid",
    },
    "INV-1003": {
        "plan": "Pro",
        "billing": "monthly",
        "seats": 1,
        "amount_usd": 8.0,
        "charged_on": "2026-09-02",
        "status": "refunded",
    },
}


def get_order(order_id: str) -> str:
    order = ORDERS.get(order_id.strip().upper())
    if order is None:
        return f"error: no order with id {order_id}"
    return ", ".join(f"{k}={v}" for k, v in order.items())


def default_tools(retriever: TfidfRetriever | None = None) -> list[Tool]:
    retriever = retriever or TfidfRetriever(load_corpus())

    def search_docs(query: str) -> str:
        hits = retriever.search(query, k=2)
        return "\n".join(c.render() for c, _ in hits) or "no matching documentation"

    return [
        Tool(
            "calculator",
            "Evaluate an arithmetic expression, e.g. '14 * 3 * 12 * 0.8'.",
            {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
            calculate,
        ),
        Tool(
            "search_docs",
            "Search the Nimbus Notes documentation (plans, pricing, refunds, security, "
            "support, accounts).",
            {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            search_docs,
        ),
        Tool(
            "get_order",
            "Look up a billing record by invoice id, e.g. 'INV-1001'.",
            {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
            get_order,
        ),
    ]
