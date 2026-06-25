"""
============================================================
Calculator 工具 — 安全的数学表达式计算器
============================================================

核心设计：使用 Python AST（抽象语法树）解析而非 eval()。

为什么不用 eval()：
    eval() 可以执行任意 Python 代码，是安全黑洞。
    例如 eval("__import__('os').system('rm -rf /')") 会执行系统命令。
    AST 模式只解析表达式结构，不执行，从根本上杜绝了代码注入。

实现原理：
    1. ast.parse(expression, mode='eval') 将表达式解析为 AST
       mode='eval' 限制为单一表达式，不能执行语句
    2. 递归遍历 AST 节点
    3. 只允许白名单中的运算符和函数
    4. 遇到不允许的操作立即抛出异常

安全措施：
    - 白名单运算符：只有 +, -, *, /, //, %, ** 等基础运算符
    - 白名单函数：只有 sqrt, sin, cos, tan, log, abs 等数学函数
    - 幂运算限制：指数不得超过 1000（防止 DDOS）
    - 纯函数：相同输入永远相同输出（可以放心缓存）

支持的操作：
    运算符: +, -, *, /, //, %, **, 一元 +/-,
    函数: sqrt, sin, cos, tan, log, log10, abs, round, min, max
    常量: pi, e
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
# 只有白名单中的函数允许在表达式中调用
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
    "pi": math.pi,      # 常量（非 callable）
    "e": math.e,         # 常量（非 callable）
}


def _safe_eval(node: ast.AST) -> float:
    """
    递归解析 AST 节点并安全计算数学表达式。

    工作原理：
    1. 接收 AST 节点
    2. 根据节点类型（数字常量/二元运算/函数调用等）分派处理
    3. 递归计算子节点
    4. 只允许白名单中的运算

    Args:
        node: AST 节点

    Returns:
        计算结果（浮点数）

    Raises:
        ValueError: 遇到不支持的操作
    """
    # ---- 数字常量节点（如 3, 3.14） ----
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"不支持的常量类型: {type(node.value).__name__}")

    # ---- 二元运算节点（如 3 + 4, 2 ** 10） ----
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise ValueError(f"不支持的运算符: {op_type.__name__}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        # 防止超大幂运算（计算量指数级增长）
        if op_type == ast.Pow and right > 1000:
            raise ValueError("幂运算指数不能超过 1000")
        return SAFE_OPERATORS[op_type](left, right)

    # ---- 一元运算节点（如 -5, +3） ----
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise ValueError(f"不支持的一元运算符: {op_type.__name__}")
        operand = _safe_eval(node.operand)
        return SAFE_OPERATORS[op_type](operand)

    # ---- 函数调用节点（如 sqrt(9), abs(-5)） ----
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
        return func  # 常量如 pi, e（直接返回值）

    # ---- 名称节点（如 pi, e 等数学常量） ----
    if isinstance(node, ast.Name):
        if node.id in SAFE_FUNCTIONS:
            val = SAFE_FUNCTIONS[node.id]
            if not callable(val):
                return val
        raise ValueError(f"不支持的变量名: {node.id}")

    # ---- Expression 包装节点（AST 根节点） ----
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)

    raise ValueError(f"不支持的表达式节点: {type(node).__name__}")


class CalculatorTool(BaseTool):
    """
    数学计算器工具。

    接受一个数学表达式字符串，使用 AST 安全解析计算结果。
    支持: +, -, *, /, //, %, **, sqrt, sin, cos, tan, log, abs, round, min, max, pi, e

    这是一个纯函数工具：相同的输入永远产生相同的输出。
    因此启用了缓存（TTL=300s），避免重复计算相同表达式。
    """

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def timeout(self) -> float:
        # 纯计算任务，5 秒足够
        return 5.0

    @property
    def cache_ttl(self) -> float:
        """纯函数，缓存 5 分钟。"""
        return 300.0

    @property
    def quota_limit(self) -> int:
        """
        计算器工具的单会话配额上限（token 数）。

        计算结果通常很短（几十个字符），所以配额设得较小。
        500 tokens 约等于 750 字符，可覆盖数百次简单计算。
        修改为 0 可取消限额，仅统计使用量。
        """
        return 500

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
        执行数学表达式计算。

        执行流程：
        1. 使用 ast.parse() 解析为 AST（mode='eval' 限制为单一表达式）
        2. 递归遍历 AST 进行安全计算
        3. 返回结果或错误信息

        Args:
            expression: 数学表达式字符串

        Returns:
            ToolResult: 包含计算结果或错误信息
        """
        try:
            # mode='eval' 限制为单一表达式，防止执行语句
            tree = ast.parse(expression.strip(), mode="eval")
            result = _safe_eval(tree)

            # 如果结果是整数，去掉小数部分（如 4.0 → 4）
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
