"""
============================================================
对话记忆管理 — 扁平摘要 + 滑动窗口（方案A）
============================================================

解决的问题：
    当对话轮次增多时，消息历史会越来越长，导致：
    1. Token 消耗急剧增加（每次 LLM 调用都要把全部历史传过去）
    2. LLM 处理变慢（长上下文的计算复杂度是 O(n²)）
    3. 可能超出上下文窗口限制

方案：扁平摘要 + 滑动窗口（业界主流方案）

压缩策略：
    1. 始终保留 system 消息（在最前面）
    2. 当非 system 消息数超过阈值（默认 50）时触发压缩
    3. 压缩后保留最近 N 条消息原文（默认 20，即滑动窗口）
    4. 早期所有消息压缩为**一段扁平摘要**，覆盖之前所有历史
    5. 每次压缩时：已有摘要 + 新掉出窗口的消息 → 重新生成单一摘要（替换旧摘要）

为什么用"扁平摘要"而非"多轮追加摘要"：
    - 多轮追加：摘要逐渐膨胀为 S1 + S2 + ... + Sn，每轮摘要都重复概述早期内容
    - 扁平摘要：始终只有 1 段摘要，不重复，token 利用率最高
    - 业界主流方案（ChatGPT、Claude、Gemini）均采用此策略

消息结构示例：
    第1次压缩后: [system] + [摘要] + [最近20条]
    第2次压缩后: [system] + [新摘要(覆盖了旧摘要+新掉出窗口的消息)] + [最近20条]
    第3次压缩后: [system] + [新摘要] + [最近20条]
    ...

降级策略：
    - LLM 摘要生成失败 → 保留旧摘要 + 截断新消息（保证系统不崩溃）
    - 无 LLM 客户端 → 直接截断（离线可用）
"""


class ConversationMemory:
    """
    对话记忆管理器 — 扁平摘要 + 滑动窗口策略。

    管理策略：
    1. 始终保留 system 消息（在最前面）
    2. 当非 system 消息数超过 max_messages 时触发压缩
    3. 压缩后保留最近 keep_recent 条消息（原始内容，滑动窗口）
    4. 早期所有消息压缩为**一段扁平摘要**，覆盖之前所有历史
    5. 每次压缩时：旧摘要 + 新掉出窗口的消息 → 重新生成单一摘要（替换旧摘要）

    为什么用扁平摘要代替多轮摘要：
        - 避免多轮摘要的信息冗余（多轮摘要中每轮都重复概述早期内容）
        - 减少 token 消耗：1 段摘要 vs N 段摘要
        - 更简单的结构：LLM 看到的始终是 [system] + [1段摘要] + [最近消息]

    Attributes:
        max_messages: 触发压缩的消息数阈值（默认 50）
        keep_recent: 压缩后保留的最近消息数（默认 20）
        _summary: 当前唯一的一段扁平摘要（覆盖所有已压缩的旧历史）
        _compress_count: 已发生的压缩轮次
    """

    # 扁平摘要的长度上限（字符数）
    # 每次压缩都替换旧摘要，不会累积，所以固定长度即可
    SUMMARY_LENGTH = 800

    def __init__(self, max_messages: int = 50, keep_recent: int = 20):
        """
        初始化记忆管理器。

        Args:
            max_messages: 触发压缩的消息数阈值，默认 50。
                          当非 system 消息超过此数量时自动压缩。
            keep_recent: 压缩后保留的最近消息数，默认 20。
                         保留越多上下文越完整，但 token 消耗也越大。
                         20 条通常可以覆盖最近 2-3 轮完整对话。
        """
        self.max_messages = max_messages
        self.keep_recent = keep_recent
        self._summary: str = ""          # 单一扁平摘要，覆盖所有已压缩的旧历史
        self._compress_count = 0          # 已发生的压缩轮次

    # ---- 公共方法 ----

    def should_compress(self, messages: list[dict]) -> bool:
        """
        判断是否需要压缩消息历史。

        只统计非 system 消息的数量（system prompt 始终需要保留）。

        Args:
            messages: 当前对话消息列表

        Returns:
            True 表示需要压缩，False 表示不需要
        """
        non_system = [m for m in messages if m.get("role") != "system"]
        return len(non_system) > self.max_messages

    async def compress(self, messages: list[dict], llm=None) -> list[dict]:
        """
        压缩消息历史（扁平摘要模式）— 异步版本。

        压缩流程：
        1. 分离原始 system 消息和已有的历史摘要
        2. 从非 system 消息中分出早期消息（待压缩）和最近消息（保留原文）
        3. 将已有摘要 + 早期消息 → 用 LLM 生成为一段新摘要（替换旧摘要）
        4. 返回: [原始system] + [新摘要] + [最近消息]

        Args:
            messages: 当前对话消息列表
            llm: LLM 客户端实例（可选），用于生成高质量摘要。
                 不传时使用简单截断降级方案。

        Returns:
            压缩后的消息列表，格式与输入相同
        """
        # ---- 第 1 步：分离 system、已有摘要、非 system 消息 ----
        original_system = [
            m for m in messages
            if m.get("role") == "system" and m.get("type") != "history_summary"
        ]
        existing_summary_msgs = [
            m for m in messages
            if m.get("type") == "history_summary"
        ]

        # 从 session 中恢复摘要状态（重启场景）
        # 当从磁盘加载已有会话时，_summary 可能为空
        # 但 existing_summary_msgs 可能包含之前压缩的摘要
        if not self._summary and existing_summary_msgs:
            self._rebuild_from_existing(existing_summary_msgs)

        # ---- 第 2 步：分离待压缩的旧消息和保留的最近消息 ----
        non_system = [m for m in messages if m.get("role") != "system"]

        # 防御性检查：未超阈值则直接返回（一般不会触发，因为 should_compress 已检查）
        if len(non_system) <= self.max_messages:
            return messages

        # 划分窗口：早期消息（待压缩）和最近消息（保留原文）
        recent = non_system[-self.keep_recent:]
        old = non_system[:-self.keep_recent]

        # ---- 第 3 步：将早期消息格式化为可读文本（给 LLM 做摘要用） ----
        old_text_parts = []
        for m in old:
            role = m.get("role", "unknown")
            content = m.get("content") or ""
            # 截断过长的单条消息，避免摘要 prompt 太长
            # 为什么不用简单截断 content[:200] + "..."：
            #   直接按字符截断会切在代码/句子/JSON 中间，LLM 看到半截信息，
            #   可能导致生成的摘要误解原文含义。
            # 策略：在最近的换行符处截断（保证行完整性），
            #   并告知 LLM 被截掉了多少内容。
            if len(content) > 200:
                truncated = content[:200]
                last_newline = truncated.rfind("\n")
                if last_newline > 0:
                    cut_at = last_newline
                else:
                    # 没有换行符时，在最近的空格处截断（保证词完整）
                    last_space = truncated.rfind(" ")
                    cut_at = last_space if last_space > 0 else 200
                remaining = len(content) - cut_at
                content = content[:cut_at] + f"\n...(省略 {remaining} 字符)"
            old_text_parts.append(f"[{role}]: {content}")
        old_text = "\n".join(old_text_parts)

        # ---- 第 4 步：生成新的扁平摘要 ----
        max_len = self._get_summary_max_length()

        new_summary = await self._generate_summary(
            old_text=old_text,
            max_length=max_len,
            llm=llm,
            existing_summary=self._summary or None,
        )

        # ---- 第 5 步：更新内部状态 ----
        self._summary = new_summary
        self._compress_count += 1

        # ---- 第 6 步：构建压缩后的消息列表 ----
        result = original_system.copy()

        # 追加扁平摘要（仅一段，替换所有旧摘要）
        result.append({
            "role": "system",
            "content": f"[对话历史摘要] {new_summary}",
            "type": "history_summary",
        })

        # 追加保留的最近消息（滑动窗口）
        result.extend(recent)

        return result

    def get_summary(self) -> str:
        """获取当前扁平摘要。"""
        return self._summary

    def get_all_summaries(self) -> list[str]:
        """
        获取所有历史摘要（兼容旧接口）。

        扁平模式下只有一段摘要，但保留此方法供外部调用。
        """
        return [self._summary] if self._summary else []

    def get_compress_count(self) -> int:
        """获取已发生的压缩轮次。"""
        return self._compress_count

    def clear_summary(self) -> None:
        """
        清空所有摘要（用于新会话或手动重置）。

        重置后后续压缩将从零开始生成新摘要。
        """
        self._summary = ""
        self._compress_count = 0

    # ---- 内部方法 ----

    def _get_summary_max_length(self) -> int:
        """
        获取摘要最大长度（固定值）。

        扁平摘要始终只有一段，每次压缩都是替换而非追加，
        所以长度固定即可，无需递增。
        """
        return self.SUMMARY_LENGTH

    async def _generate_summary(
        self,
        old_text: str,
        max_length: int,
        llm=None,
        existing_summary: str | None = None,
    ) -> str:
        """
        生成（或更新）扁平摘要。

        核心思路：
        - 首次压缩：直接压缩原始对话内容
        - 后续压缩：将旧摘要与新掉出窗口的消息合并重生成一段新摘要
        - LLM 不可用或失败时降级为文本截断

        Args:
            old_text: 格式化后的早期对话文本（即将被压缩的消息）
            max_length: 摘要长度上限（字数）
            llm: LLM 客户端实例（可选）
            existing_summary: 已有的摘要文本（可选），首次压缩时为 None

        Returns:
            新的摘要文本
        """
        if llm:
            prompt = self._build_summary_prompt(old_text, max_length, existing_summary)
            try:
                response = await llm.chat_async(
                    messages=[{"role": "user", "content": prompt}],
                    tools=None,  # 摘要不需要工具
                )
                return response.get("content") or self._fallback_text(
                    old_text, max_length, existing_summary
                )
            except Exception:
                # LLM 调用失败时降级为文本截断
                # 为什么降级而非抛异常：摘要失败不影响主流程
                return self._fallback_text(old_text, max_length, existing_summary)
        else:
            # 无 LLM 时的简单截断方案
            return self._fallback_text(old_text, max_length, existing_summary)

    def _build_summary_prompt(
        self,
        old_text: str,
        max_length: int,
        existing_summary: str | None = None,
    ) -> str:
        """
        构建摘要生成的 Prompt。

        核心设计：
        - 有旧摘要时：让 LLM 将新内容"合并"进旧摘要，而非从头重写
        - 无旧摘要时（首次压缩）：直接压缩原始对话
        - 明确给定额度，让 LLM 在限制内最大化信息密度
        """
        if existing_summary:
            return (
                f"以下是已有的对话摘要（概括了更早期的对话内容）：\n"
                f"{existing_summary}\n\n"
                f"以下是需要合并到摘要中的新对话内容：\n{old_text}\n\n"
                f"请将已有摘要和新内容合并为一段新的完整摘要，"
                f"保留关键信息、用户意图、重要结论和上下文。"
                f"如果新内容与已有摘要有重叠，优先保留新内容中的细节。"
                f"不超过 {max_length} 字。"
            )
        else:
            return (
                f"以下是需要压缩的对话内容：\n{old_text}\n\n"
                f"请将以上对话压缩为一段简洁的摘要，"
                f"保留关键信息、用户意图、重要结论和上下文，"
                f"不超过 {max_length} 字。"
            )

    def _fallback_text(
        self,
        old_text: str,
        max_length: int,
        existing_summary: str | None = None,
    ) -> str:
        """
        降级方案：LLM 不可用或失败时使用文本截断。

        策略：
        - 优先保留新掉出窗口的消息（它们是最近上下文，且从未被摘要过）
        - 旧摘要是已经压缩过的旧信息，仅在空间充裕时做简短提及
        - 无旧摘要（首次压缩）时直接截取旧消息

        为什么新消息优先于旧摘要：
        旧摘要在之前压缩时已存在于 context 中供 LLM 阅读，
        而新掉出窗口的消息是本次才需要被保留的，丢失代价更大。
        """
        # 旧摘要仅作简短提及（最多 80 字），主体留给新消息
        note = ""
        if existing_summary:
            note = f"[更早期摘要: {existing_summary[:80]}...]\n"

        remaining = max_length - len(note)

        # 始终保留 13 字的空间给末尾标记
        marker = "\n[摘要降级-文本截断]"
        remaining -= len(marker)

        if remaining >= 50:
            return note + old_text[:remaining] + marker
        # 空间不够时舍弃旧摘要引用，但保留标记
        return old_text[:max_length - len(marker)] + marker

    def _rebuild_from_existing(self, existing_summaries: list[dict]):
        """
        从已有的历史摘要消息中恢复内部状态。

        用于会话重启场景：从 session 文件中加载消息后，
        重建 _summary 和 _compress_count。

        扁平模式下理论上只有一段摘要，但如果有多个（升级兼容），
        取最后一段作为当前摘要。

        Args:
            existing_summaries: 消息列表中 type="history_summary" 的消息
        """
        if existing_summaries:
            # 取最后一段摘要（扁平模式下只有一段，这里做兼容处理）
            last = existing_summaries[-1]
            content = last.get("content", "")
            # 提取摘要文本（去掉前缀标记，如 "[对话历史摘要] "）
            if "] " in content:
                self._summary = content.split("] ", 1)[1]
            else:
                self._summary = content
            self._compress_count = len(existing_summaries)
