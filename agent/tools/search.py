"""
============================================================
Search 工具 — Mock（模拟）搜索引擎
============================================================

当前实现：返回预定义的 Mock 搜索结果，无需真实网络请求。

为什么用 Mock：
    1. 减少依赖：真实搜索需要 API Key、网络请求处理、结果解析
    2. 演示目的：Mock 数据足够展示 Agent 的"搜索-分析"能力
    3. 可预测性：测试时结果可控，不依赖外部服务
    4. 零成本：不需要支付搜索 API 费用

Mock 数据库覆盖的主题：
    - Python（官方文档、教程）
    - AI Agent（概念介绍、ReAct 论文）
    - 天气（北京、上海）
    - 大模型（LLM 概述）
    - Function Calling（OpenAI 指南）

搜索逻辑：
    1. 将查询转为小写
    2. 双向匹配：查询包含关键词 或 关键词包含在查询中
    3. 收集所有匹配结果，限制最大返回条数
    4. 格式化输出

TODO: 后续可替换为真实搜索引擎（如 DuckDuckGo、SerpAPI、Bing Search）
"""

import time
from agent.tools.base import BaseTool, ToolResult


# ============================================================
# Mock 搜索数据库
# 键为关键词，值为搜索结果列表
# 每条结果包含 title, snippet(摘要), url(链接)
# ============================================================
MOCK_SEARCH_DB: dict[str, list[dict[str, str]]] = {
    "python": [
        {
            "title": "Python 官方文档",
            "snippet": "Python 是一种解释型、高级、通用型编程语言。Python 的设计哲学强调代码的可读性和简洁性。",
            "url": "https://docs.python.org/3/",
        },
        {
            "title": "Python 教程 - 廖雪峰",
            "snippet": "Python 入门教程，涵盖基础语法、高级特性、网络编程等内容。",
            "url": "https://www.liaoxuefeng.com/wiki/1016959663602400",
        },
    ],
    "agent": [
        {
            "title": "什么是 AI Agent",
            "snippet": "AI Agent（智能代理）是一种能够感知环境、做出决策并采取行动以实现目标的系统。它通常包含感知、推理、规划和执行等核心模块。",
            "url": "https://example.com/ai-agent-intro",
        },
        {
            "title": "ReAct: 推理与行动的协同",
            "snippet": "ReAct 是一种将推理(Reasoning)和行动(Acting)交替进行的范式，Agent 通过思考链来决定下一步行动。",
            "url": "https://example.com/react-paper",
        },
    ],
    "weather": [
        {
            "title": "今日天气预报",
            "snippet": "北京：晴，25°C，湿度 40%，微风。适合户外活动。",
            "url": "https://example.com/weather/beijing",
        },
        {
            "title": "一周天气预报",
            "snippet": "上海：周一多云 22°C，周二小雨 18°C，周三晴 24°C。",
            "url": "https://example.com/weather/shanghai",
        },
    ],
    "大模型": [
        {
            "title": "大语言模型概述",
            "snippet": "大语言模型(LLM)是基于 Transformer 架构的大规模预训练语言模型，如 GPT-4、Claude 等，具备强大的自然语言理解和生成能力。",
            "url": "https://example.com/llm-overview",
        },
    ],
    "function calling": [
        {
            "title": "OpenAI Function Calling 指南",
            "snippet": "Function Calling 允许模型在需要时调用外部函数，将自然语言请求转换为结构化的函数调用参数。",
            "url": "https://platform.openai.com/docs/guides/function-calling",
        },
    ],
}

# 默认结果 — 当关键词未命中时使用
DEFAULT_RESULT = [
    {
        "title": "搜索结果",
        "snippet": "未找到与查询高度相关的结果，建议尝试更具体的关键词。",
        "url": "https://example.com/search",
    }
]


class SearchTool(BaseTool):
    """
    Mock 搜索引擎工具。

    模拟搜索引擎行为，根据关键词返回预定义的搜索结果。
    支持部分匹配：查询中包含数据库中的关键词即可命中。
    """

    @property
    def name(self) -> str:
        return "search"

    @property
    def timeout(self) -> float:
        return 5.0

    @property
    def quota_limit(self) -> int:
        """
        搜索工具的单会话配额上限（token 数）。

        搜索结果通常有数百字符，将配额设为 2000 tokens。
        修改为 0 可取消限额，仅统计使用量。
        """
        return 2000

    @property
    def description(self) -> str:
        return (
            "搜索信息查询。输入搜索关键词，返回相关的搜索结果（标题、摘要、链接）。"
            "可用于查询技术文档、概念解释、天气等信息。"
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索查询关键词",
                },
                "max_results": {
                    "type": "integer",
                    "description": "最大返回结果数，默认 3",
                    "default": 3,
                },
            },
            "required": ["query"],
        }

    def execute(self, query: str, max_results: int = 3) -> ToolResult:
        """
        执行 Mock 搜索。

        搜索逻辑：
        1. 将查询转为小写
        2. 遍历 Mock 数据库，检查查询是否包含任何关键词
        3. 收集所有匹配结果
        4. 截断到 max_results 数量
        5. 格式化返回

        Args:
            query: 搜索查询字符串
            max_results: 最大返回结果数

        Returns:
            ToolResult: 格式化的搜索结果文本
        """
        query_lower = query.lower().strip()
        results: list[dict[str, str]] = []

        # 遍历 Mock 数据库进行关键词匹配（双向匹配）
        for keyword, entries in MOCK_SEARCH_DB.items():
            # 双向匹配：查询包含关键词 或 关键词包含在查询中
            # 例如 query="Python 教程" → 匹配 keyword="python"
            #      query="py" → 不匹配，因为 "py" 在 "python" 中但不构成完整词
            if keyword.lower() in query_lower or query_lower in keyword.lower():
                results.extend(entries)

        # 如果没有匹配结果，返回默认结果
        if not results:
            results = DEFAULT_RESULT.copy()

        # 限制结果数量
        results = results[:max_results]

        # 格式化输出
        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(f"[{i}] {r['title']}")
            formatted.append(f"    摘要: {r['snippet']}")
            formatted.append(f"    链接: {r['url']}")
            formatted.append("")

        output = f"搜索 \"{query}\" 的结果（共 {len(results)} 条）：\n" + "\n".join(formatted)

        return ToolResult(
            success=True,
            result=output,
            metadata={"query": query, "result_count": len(results)},
        )
