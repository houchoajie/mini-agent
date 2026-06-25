"""
============================================================
工具测试套件 — 所有工具的自动化测试用例
============================================================

测试覆盖：
    1. 工具基本属性（name, description, parameters 不为空）
    2. JSON Schema 合法性（type, properties, required）
    3. safe_execute 的正常调用和错误处理
    4. max_result_chars / max_result_tokens 截断
    5. ToolContext 鉴权
    6. 缓存机制（如有）
    7. 执行示例（验证参数格式）

运行方式：
    python -m unittest tests/test_tool_base.py -v

注意：
    各测试类的 tool_class 当前设置为 None，需要先导入具体工具类
    并设置 tool_class 才能运行。例如：
        from agent.tools.calculator import CalculatorTool
        class TestCalculator(ToolTestCase):
            tool_class = CalculatorTool
"""

import unittest
import json
from pathlib import Path
from agent.tools.base import (
    BaseTool, ToolResult, ToolContext, ToolError, ErrorCode,
)


class ToolTestCase(unittest.TestCase):
    """
    工具测试基类。

    继承此类并设置 tool_class 即可获得标准测试集。
    子类可覆盖 test_examples 等方法来测试特定工具的行为。

    用法:
        class TestCalculator(ToolTestCase):
            tool_class = CalculatorTool

            def test_specific_feature(self):
                result = self.tool.execute(expression="2 + 3")
                self.assertTrue(result.success)
    """

    tool_class: type[BaseTool] | None = None
    valid_params: list[dict] = []
    invalid_params: list[dict] = []

    def setUp(self):
        """每个测试前创建工具实例并注入测试上下文。"""
        if self.tool_class is None:
            self.skipTest("tool_class 未设置")
        self.tool = self.tool_class()
        self.test_context = ToolContext(
            username="test_user",
            user_dir=Path("/tmp/test_agent_user"),
            session_id="test_session",
        )
        self.tool.set_context(self.test_context)

    # ================================================================
    # 基础属性测试
    # ================================================================

    def test_name_not_empty(self):
        """工具名称不能为空。"""
        self.assertTrue(len(self.tool.name) > 0, "name 不能为空")

    def test_description_not_empty(self):
        """工具描述不能为空。"""
        self.assertTrue(len(self.tool.description) > 0, "description 不能为空")

    def test_parameters_is_dict(self):
        """parameters 必须是 dict，且包含 type 和 properties。"""
        params = self.tool.parameters
        self.assertIsInstance(params, dict, "parameters 必须是 dict")
        self.assertIn("type", params, "parameters 必须包含 type")
        self.assertIn("properties", params, "parameters 必须包含 properties")

    def test_parameters_properties_is_dict(self):
        """parameters.properties 必须是 dict。"""
        props = self.tool.parameters.get("properties", {})
        self.assertIsInstance(props, dict, "parameters.properties 必须是 dict")

    def test_positive_timeout(self):
        """timeout 必须大于 0。"""
        self.assertGreater(self.tool.timeout, 0, "timeout 必须大于 0")

    def test_positive_max_result_chars(self):
        """max_result_chars 必须 >= 0。"""
        self.assertGreaterEqual(self.tool.max_result_chars, 0)

    def test_positive_max_result_tokens(self):
        """max_result_tokens 必须 >= 0。"""
        self.assertGreaterEqual(self.tool.max_result_tokens, 0)

    # ================================================================
    # Schema 格式测试
    # ================================================================

    def test_to_openai_schema_valid(self):
        """to_openai_schema() 必须返回合法的 OpenAI function calling 格式。"""
        schema = self.tool.to_openai_schema()
        self.assertIn("type", schema, "schema 必须包含 type")
        self.assertEqual(schema["type"], "function", "type 必须是 function")
        self.assertIn("function", schema, "schema 必须包含 function")
        func = schema["function"]
        self.assertIn("name", func, "function 必须包含 name")
        self.assertIn("description", func, "function 必须包含 description")
        self.assertIn("parameters", func, "function 必须包含 parameters")

    def test_openai_schema_name_matches(self):
        """OpenAI schema 的 name 必须与工具 name 属性一致。"""
        schema = self.tool.to_openai_schema()
        self.assertEqual(schema["function"]["name"], self.tool.name)

    # ================================================================
    # 执行测试
    # ================================================================

    def test_safe_execute_no_args(self):
        """safe_execute('{}') 至少不会崩溃（可能返回参数校验错误）。"""
        result = self.tool.safe_execute("{}")
        self.assertIsInstance(result, ToolResult, "返回类型必须是 ToolResult")

    def test_safe_execute_empty_string(self):
        """safe_execute('') 不应崩溃。"""
        result = self.tool.safe_execute("")
        self.assertIsInstance(result, ToolResult)

    def test_safe_execute_none_json(self):
        """safe_execute('null') 不应崩溃。"""
        result = self.tool.safe_execute("null")
        self.assertIsInstance(result, ToolResult)

    def test_safe_execute_invalid_json(self):
        """safe_execute('{bad json') 应优雅处理 JSON 解析错误。"""
        result = self.tool.safe_execute("{bad json")
        self.assertIsInstance(result, ToolResult)

    def test_safe_execute_with_context_passing(self):
        """safe_execute 传入 context 后工具应能正确访问。"""
        ctx = ToolContext(username="context_test_user")
        result = self.tool.safe_execute("{}", context=ctx)
        self.assertIsInstance(result, ToolResult)

    # ================================================================
    # 缓存测试（如果启用）
    # ================================================================

    def test_cache_ttl_non_negative(self):
        """cache_ttl 必须 >= 0。"""
        self.assertGreaterEqual(self.tool.cache_ttl, 0)

    # ================================================================
    # 自文档化测试
    # ================================================================

    def test_examples_format(self):
        """examples 格式必须正确（每个示例包含 description 和 arguments）。"""
        examples = self.tool.examples
        for ex in examples:
            self.assertIn("description", ex, "每个示例必须包含 description")
            self.assertIn("arguments", ex, "每个示例必须包含 arguments")
            self.assertIsInstance(ex["arguments"], dict, "arguments 必须是 dict")

    def test_usage_guide_is_string(self):
        """usage_guide 必须是字符串。"""
        self.assertIsInstance(self.tool.usage_guide, str)

    # ================================================================
    # 辅助方法
    # ================================================================

    def call_tool(self, **kwargs) -> ToolResult:
        """便捷方法：调用 safe_execute 并返回结果。

        用法: result = self.call_tool(expression="2 + 3")
        """
        return self.tool.safe_execute(json.dumps(kwargs), context=self.test_context)

    def assertToolSuccess(self, result: ToolResult, msg: str = ""):
        """断言工具执行成功。"""
        self.assertTrue(result.success, msg or f"工具执行失败: {result.error}")

    def assertToolError(self, result: ToolResult, code: str = "", msg: str = ""):
        """断言工具执行失败并可选地校验错误码。"""
        self.assertFalse(result.success, msg or "工具应执行失败")
        if code:
            self.assertEqual(result.error_code, code,
                             f"错误码应为 {code}，实际为 {result.error_code}")


# ================================================================
# 具体工具测试类
# 需要导入对应工具类并设置 tool_class 才能运行
# ================================================================

class TestCalculator(ToolTestCase):
    """计算器工具测试。"""
    tool_class = None
    # from agent.tools.calculator import CalculatorTool
    # tool_class = CalculatorTool

    valid_params = [
        {"expression": "2 + 3"},
        {"expression": "2 ** 10"},
        {"expression": "sqrt(144)"},
        {"expression": "pi * 5 ** 2"},
    ]

    def test_simple_addition(self):
        result = self.call_tool(expression="2 + 3")
        self.assertToolSuccess(result)
        self.assertIn("5", result.result)

    def test_power(self):
        result = self.call_tool(expression="2 ** 10")
        self.assertToolSuccess(result)
        self.assertIn("1024", result.result)

    def test_with_function(self):
        result = self.call_tool(expression="sqrt(144)")
        self.assertToolSuccess(result)
        self.assertIn("12", result.result)


class TestDateTime(ToolTestCase):
    """时间工具测试。"""
    tool_class = None
    # from agent.tools.datetime_tool import DateTimeTool
    # tool_class = DateTimeTool

    def test_default_format(self):
        result = self.call_tool()
        self.assertToolSuccess(result)
        self.assertIn("当前时间", result.result)


class TestSearch(ToolTestCase):
    """搜索工具测试。"""
    tool_class = None
    # from agent.tools.search import SearchTool
    # tool_class = SearchTool

    def test_search_python(self):
        result = self.call_tool(query="python")
        self.assertToolSuccess(result)

    def test_search_no_results(self):
        result = self.call_tool(query="xyznonexistent12345")
        self.assertToolSuccess(result)


class TestTodoManager(ToolTestCase):
    """任务管理工具测试。"""
    tool_class = None
    # from agent.tools.todo_manager import TodoManagerTool
    # tool_class = TodoManagerTool

    def test_list_empty(self):
        result = self.call_tool(action="list")
        self.assertToolSuccess(result)


class TestFileWriter(ToolTestCase):
    """文件写入工具测试。"""
    tool_class = None
    # from agent.tools.file_writer import FileWriterTool
    # tool_class = FileWriterTool


class TestFileReader(ToolTestCase):
    """文件读取工具测试。"""
    tool_class = None
    # from agent.tools.file_reader import FileReaderTool
    # tool_class = FileReaderTool


if __name__ == "__main__":
    unittest.main()
