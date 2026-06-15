"""
Calculator 工具 - 安全的数学表达式计算器

支持基本的四则运算、幂运算、取余等。
使用 AST 解析确保安全性（不使用 eval）。
"""

import ast
import math
import operator
from agent.tools.base import BaseTool, ToolResult


# ============================================================
# 安全运算符映射表
# 只允许这些运算操作，防止任意代码执行
# ============================================================
SAFE_OPERATORS = {
    ast.Add: operator.add,       # +
    ast.Sub: operator.sub,       # -
    ast.Mult: operator.mul,      # *
    ast.Div: operator.truediv,   # /
    ast.FloorDiv: operator.floordiv,  # //
    ast.Mod: operator.mod,       # %
    ast.Pow: operator.pow,       # **
    ast.USub: operator.neg,      # 一元负号 -x
    ast.UAdd: operator.pos,      # 一元正号 +x
}

# ============================================================
# 安全数学函数白名单
# ============================================================
SAFE_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "pi": math.pi,
    "e": math.e,
}


def _safe_eval(node: ast.AST) -> float:
    """
    递归解析 AST 节点并安全计算数学表达式

    工作原理：
    1. 将表达式字符串解析为 Python AST（抽象语法树）
    2. 递归遍历 AST 节点
    3. 只允许白名单中的运算符和函数
    4. 遇到不允许的操作立即抛出异常

    Args:
        node: AST 节点

    Returns:
        计算结果（浮点数）

    Raises:
        ValueError: 遇到不支持的操作
    """
    # 数字常量节点（如 3, 3.14）
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"不支持的常量类型: {type(node.value).__name__}")

    # 二元运算节点（如 3 + 4, 2 ** 10）
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise ValueError(f"不支持的运算符: {op_type.__name__}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        # 防止超大幂运算
        if op_type == ast.Pow and right > 1000:
            raise ValueError("幂运算指数不能超过 1000")
        return SAFE_OPERATORS[op_type](left, right)

    # 一元运算节点（如 -5, +3）
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise ValueError(f"不支持的一元运算符: {op_type.__name__}")
        operand = _safe_eval(node.operand)
        return SAFE_OPERATORS[op_type](operand)

    # 函数调用节点（如 sqrt(9), abs(-5)）
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("不支持的函数调用形式")
        func_name = node.func.id
        if func_name not in SAFE_FUNCTIONS:
            raise ValueError(f"不支持的函数: {func_name}")
        args = [_safe_eval(arg) for arg in node.args]
        func = SAFE_FUNCTIONS[func_name]
        if callable(func):
            return func(*args)
        return func  # 常量如 pi, e

    # 名称节点（如 pi, e 等数学常量）
    if isinstance(node, ast.Name):
        if node.id in SAFE_FUNCTIONS:
            val = SAFE_FUNCTIONS[node.id]
            if not callable(val):
                return val
        raise ValueError(f"不支持的变量名: {node.id}")

    # Expression 包装节点
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)

    raise ValueError(f"不支持的表达式节点: {type(node).__name__}")


class CalculatorTool(BaseTool):
    """
    数学计算器工具

    接受一个数学表达式字符串，使用 AST 安全解析计算结果。
    支持: +, -, *, /, //, %, **, sqrt, sin, cos, tan, log, abs, round, min, max, pi, e

    使用示例：
        expression: "2 ** 10 + sqrt(144)"
        返回: "计算结果: 1036.0"
    """

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return (
            "安全的数学表达式计算器。输入一个数学表达式字符串，返回计算结果。"
            "支持四则运算、幂运算(**)、取余(%)、以及 sqrt/sin/cos/tan/log/abs/round/min/max 函数。"
            "数学常量 pi 和 e 也可使用。"
            "示例: '2 ** 10 + sqrt(144)', '(3 + 5) * 2'"
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "要计算的数学表达式，例如 '2 + 3 * 4' 或 'sqrt(144) + pi'",
                }
            },
            "required": ["expression"],
        }

    def execute(self, expression: str) -> ToolResult:
        """
        执行数学表达式计算

        执行流程：
        1. 接收表达式字符串
        2. 使用 ast.parse() 解析为 AST
        3. 递归遍历 AST 进行安全计算
        4. 返回结果或错误信息

        Args:
            expression: 数学表达式字符串

        Returns:
            ToolResult: 包含计算结果或错误信息
        """
        try:
            # 使用 ast.parse 解析表达式（mode='eval' 限制为单表达式）
            tree = ast.parse(expression.strip(), mode="eval")
            result = _safe_eval(tree)

            # 如果结果是整数，去掉小数部分
            if result == int(result) and abs(result) < 1e15:
                result = int(result)

            return ToolResult(
                success=True,
                result=f"计算结果: {expression} = {result}",
                metadata={"expression": expression, "value": result},
            )
        except (ValueError, TypeError, ZeroDivisionError) as e:
            return ToolResult(
                success=False,
                result="",
                error=f"计算失败: {e}. 表达式: '{expression}'",
            )
        except RecursionError:
            return ToolResult(
                success=False,
                result="",
                error="表达式过于复杂，超出计算深度限制",
            )
