"""
对话记忆管理 - 支持历史摘要压缩

当对话轮次增多时，消息历史会越来越长，导致：
1. Token 消耗急剧增加
2. LLM 处理变慢
3. 可能超出上下文窗口限制

ConversationMemory 通过自动压缩早期消息为摘要来解决这个问题：
- 当消息数超过阈值时，将早期消息交给 LLM 生成摘要
- 保留最近的 N 条消息 + 早期摘要
- 摘要本身也会持久化到 Session 中

使用示例：
    memory = ConversationMemory(max_messages=50, keep_recent=20)

    # 在每次 LLM 调用前检查是否需要压缩
    if memory.should_compress(session.messages):
        compressed = memory.compress(session.messages, llm_client)
        # compressed 格式: [system_msg, summary_msg, ...recent_msgs]
"""


class ConversationMemory:
    """
    对话记忆管理器

    管理策略：
    1. 始终保留 system 消息（在最前面）
    2. 当非 system 消息数超过 max_messages 时触发压缩
    3. 压缩后保留最近 keep_recent 条消息
    4. 早期消息被压缩为一段摘要文本

    Attributes:
        max_messages: 触发压缩的消息数阈值
        keep_recent: 压缩后保留的最近消息数
        _summary: 当前累积的对话摘要文本
    """

    def __init__(self, max_messages: int = 50, keep_recent: int = 20):
        """
        初始化记忆管理器

        Args:
            max_messages: 触发压缩的消息数阈值，默认 50
                          当非 system 消息超过此数量时自动压缩
            keep_recent: 压缩后保留的最近消息数，默认 20
                         保留越多上下文越完整，但 token 消耗也越大
        """
        self.max_messages = max_messages
        self.keep_recent = keep_recent
        self._summary: str = ""

    def should_compress(self, messages: list[dict]) -> bool:
        """
        判断是否需要压缩消息历史

        只统计非 system 消息的数量（system 消息不参与压缩判断）。

        Args:
            messages: 当前对话消息列表

        Returns:
            True 表示需要压缩，False 表示不需要
        """
        non_system = [m for m in messages if m.get("role") != "system"]
        return len(non_system) > self.max_messages

    def compress(self, messages: list[dict], llm_client=None) -> list[dict]:
        """
        压缩消息历史

        压缩策略：
        1. 分离 system 消息和非 system 消息
        2. 如果非 system 消息未超阈值，直接返回原消息
        3. 将早期消息格式化为文本摘要
        4. 如果有 LLM 客户端，用 LLM 生成高质量摘要
        5. 如果没有 LLM，使用简单截断作为降级方案
        6. 返回: [system_msg] + [summary_msg] + [recent_msgs]

        Args:
            messages: 当前对话消息列表
            llm_client: LLM 客户端实例（可选），用于生成高质量摘要

        Returns:
            压缩后的消息列表，格式与输入相同
        """
        # 分离 system 消息和非 system 消息
        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]

        # 如果未超阈值，直接返回
        if len(non_system) <= self.max_messages:
            return messages

        # 划分：早期消息（待压缩）和最近消息（保留原文）
        recent = non_system[-self.keep_recent:]
        old = non_system[:-self.keep_recent]

        # 将早期消息格式化为可读文本
        old_text_parts = []
        for m in old:
            role = m.get("role", "unknown")
            content = m.get("content") or ""
            # 截断过长的单条消息
            if len(content) > 200:
                content = content[:200] + "..."
            old_text_parts.append(f"[{role}]: {content}")
        old_text = "\n".join(old_text_parts)

        # 生成摘要
        if llm_client:
            # 使用 LLM 生成高质量摘要
            prompt = self._build_summary_prompt(old_text)
            try:
                response = llm_client.chat(
                    messages=[{"role": "user", "content": prompt}],
                    tools=None,  # 摘要生成不需要工具
                )
                self._summary = response.get("content") or old_text[:500]
            except Exception:
                # LLM 调用失败时降级为简单截断
                self._summary = old_text[:500] + "\n[摘要生成失败，使用截断文本]"
        else:
            # 无 LLM 时的简单截断方案
            self._summary = old_text[:500] + "\n[早期对话已压缩]"

        # 构建压缩后的消息列表
        result = system_msgs.copy()

        # 添加摘要消息（作为 system 角色，确保 LLM 能看到）
        if self._summary:
            result.append({
                "role": "system",
                "content": f"[对话历史摘要] {self._summary}",
            })

        # 添加保留的最近消息
        result.extend(recent)

        return result

    def _build_summary_prompt(self, old_text: str) -> str:
        """
        构建摘要生成的 Prompt

        Args:
            old_text: 格式化后的早期对话文本

        Returns:
            摘要生成的完整 Prompt
        """
        if self._summary:
            # 如果已有旧摘要，要求更新
            return (
                f"以下是之前的对话摘要：\n{self._summary}\n\n"
                f"以下是新增的对话内容：\n{old_text}\n\n"
                f"请将以上信息合并，生成一段更新后的简洁摘要（保留关键上下文，不超过 300 字）。"
            )
        else:
            # 首次生成摘要
            return (
                f"以下是之前的对话历史：\n{old_text}\n\n"
                f"请将以上对话压缩为一段简洁的摘要（保留关键信息、用户意图和重要结论，不超过 300 字）。"
            )

    def get_summary(self) -> str:
        """获取当前摘要文本"""
        return self._summary

    def clear_summary(self) -> None:
        """清空摘要（用于新会话或手动重置）"""
        self._summary = ""
