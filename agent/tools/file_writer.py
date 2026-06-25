"""
============================================================
File Writer 工具 — 将内容写入本地文件
============================================================

让 Agent 具备写入本地文件的能力，支持：
- 文本文件（.txt, .md, .py, .json, .csv 等）
- Word 文档（.docx）— 依赖 python-docx
- PDF 文档（.pdf）— 依赖 fpdf2
- 覆盖写入和追加写入
- 自动创建目录（如果不存在）

安全机制（三层防护）：
    1. 扩展名白名单：只允许写入明确列出的文件类型
    2. 系统路径黑名单：禁止写入 C:/Windows、C:/Program Files 等
    3. 写入大小限制：超过 1MB 拒绝写入

使用示例：
    Agent 调用: write_file(file_path="notes.txt", content="这是一条笔记", append=False)
    结果: 文件写入成功！路径: notes.txt, 大小: 14 字节
"""

from pathlib import Path
from agent.tools.base import BaseTool, ToolResult, ToolError, ErrorCode, require_import


# ============================================================
# 允许写入的文件扩展名白名单
# 比读取更严格，防止意外覆盖重要文件
# ============================================================
ALLOWED_WRITE_EXTENSIONS = {
    ".txt", ".md", ".py", ".json", ".csv", ".xml", ".html",
    ".css", ".js", ".ts", ".yaml", ".yml", ".toml",
    ".ini", ".cfg", ".log", ".sql", ".lua",
    ".pdf", ".docx",
}

# ============================================================
# 禁止写入的路径模式
# 防止 Agent 误操作系统关键文件
# ============================================================
BLOCKED_PATHS = {
    "C:/Windows", "C:/Program Files", "C:/Program Files (x86)",
}


class FileWriterTool(BaseTool):
    """
    文件写入工具 — 让 Agent 能写入本地文本文件。

    工作原理：
    1. 接收文件路径和内容参数
    2. 检查路径安全性（扩展名白名单 + 路径黑名单）
    3. 自动创建父目录（如果不存在）
    4. 以覆盖或追加模式写入文件
    5. 返回写入结果确认
    """

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def timeout(self) -> float:
        return 30.0

    @property
    def description(self) -> str:
        return (
            "将文本内容写入指定的本地文件。"
            "支持 .txt, .md, .py, .json, .csv, .yaml 等文本文件，"
            "以及 .pdf 文档和 .docx Word 文档。"
            "如果文件不存在会自动创建，如果存在则根据 append 参数决定覆盖或追加。"
            "注意：覆盖已有文件时，需要先不加 force 参数让工具询问用户确认，"
            "用户同意后再加 force=true 执行覆盖。"
            "可用于生成代码、写笔记、导出数据、创建文档报告等。"
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "目标文件路径（支持绝对路径和相对路径）",
                },
                "content": {
                    "type": "string",
                    "description": "要写入的文件内容",
                },
                "append": {
                    "type": "boolean",
                    "description": "是否追加模式。false=覆盖写入（默认），true=追加到文件末尾",
                    "default": False,
                },
                "force": {
                    "type": "boolean",
                    "description": "是否强制覆盖。当目标文件已存在时，先不加此参数让工具询问用户确认，用户同意后再加 force=true 执行覆盖。默认为 false。",
                    "default": False,
                },
            },
            "required": ["file_path", "content"],
        }

    def before_execute(self, kwargs: dict, context) -> dict | None:
        """写入前检查路径权限（由 before_execute 钩子自动调用）。"""
        file_path = kwargs.get("file_path", "")
        if file_path and context:
            context.assert_file_path_allowed(file_path, operation="write")
        return None

    def execute(self, file_path: str, content: str, append: bool = False, force: bool = False) -> ToolResult:
        """
        执行文件写入。

        安全流水线：
        1. 解析路径为绝对路径
        2. 扩展名白名单检查
        3. 系统路径黑名单检查
        4. 文件已存在检查（覆盖模式时，未传 force 则询问用户确认）
        5. 写入大小限制检查（1MB）
        6. 自动创建父目录
        7. 写入文件（覆盖或追加）
        8. 返回结果确认

        Args:
            file_path: 目标文件路径
            content: 要写入的内容
            append: 是否追加模式（True=追加到末尾，False=覆盖）
            force: 是否强制覆盖。文件已存在时需先询问用户，用户同意后再传 force=true

        Returns:
            ToolResult: 写入结果或错误信息
        """
        try:
            # 解析路径为绝对路径，支持 ~ 展开
            path = Path(file_path).expanduser().resolve()

            # ---- 安全检查 1：扩展名白名单 ----
            suffix = path.suffix.lower()
            if suffix and suffix not in ALLOWED_WRITE_EXTENSIONS:
                return ToolResult(
                    success=False, result="",
                    error=f"不支持写入的文件类型: {suffix}。支持的类型: {', '.join(sorted(ALLOWED_WRITE_EXTENSIONS)[:10])}...",
                )

            # ---- 安全检查 2：系统路径黑名单 ----
            path_str = str(path).lower()
            for blocked in BLOCKED_PATHS:
                if path_str.startswith(blocked.lower()):
                    return ToolResult(
                        success=False, result="",
                        error=f"安全限制：不允许写入系统目录下的文件: {path}",
                    )

            # ---- 安全检查 3：覆盖已存在文件时需用户确认 ----
            # 只有覆盖模式（非追加、非强制）且文件已存在时触发多轮交互
            if not append and not force and path.exists():
                file_size = path.stat().st_size
                return ToolResult(
                    success=True,
                    result=(
                        f"文件已存在: {path}\n"
                        f"  大小: {file_size:,} 字节\n"
                        f"  操作: 覆盖写入\n"
                        f"请在确认后重新调用 write_file 并设置 force=true 执行覆盖。"
                    ),
                    ask_user=(
                        f"文件 '{path.name}' 已存在（{file_size:,} 字节），"
                        f"是否确认覆盖写入？"
                    ),
                )

            # 自动创建父目录
            path.parent.mkdir(parents=True, exist_ok=True)

            # 根据文件类型选择不同的写入方式
            if suffix == ".pdf":
                action, content_bytes = self._write_pdf(path, content)
            elif suffix == ".docx":
                action, content_bytes = self._write_docx(path, content)
            else:
                action, content_bytes = self._write_text(path, content, append)

            result = (
                f"文件{action}成功！\n"
                f"  路径: {path}\n"
                f"  大小: {content_bytes:,} 字节\n"
                f"  模式: {'追加' if append else '覆盖'}"
            )

            return ToolResult(
                success=True,
                result=result,
                metadata={
                    "path": str(path),
                    "bytes_written": content_bytes,
                    "mode": "append" if append else "overwrite",
                },
            )

        except PermissionError:
            return ToolResult(
                success=False, result="",
                error=f"权限不足，无法写入文件: {file_path}",
            )
        except Exception as e:
            return ToolResult(
                success=False, result="",
                error=f"写入文件失败: {type(e).__name__}: {e}",
            )

    # ============================================================
    # 专用写入方法
    # ============================================================

    def _write_text(self, path: Path, content: str, append: bool) -> tuple[str, int]:
        """
        写入纯文本文件（UTF-8 编码）。

        原子安全策略：
        - 覆盖模式（append=False）：先写入 .tmp 临时文件，再通过 replace() 原子替换。
          即使中途崩溃，原文件不会被破坏（要么全部成功，要么原文件不变）。
        - 追加模式（append=True）：无法原子化，直接追加到原文件末尾。
        """
        # 检查写入大小限制（1MB）
        content_bytes = len(content.encode("utf-8"))
        if content_bytes > 1024 * 1024:
            raise ValueError(f"写入内容过大: {content_bytes / 1024 / 1024:.1f}MB，超过 1MB 限制")

        if append:
            # 追加模式：无法原子操作，直接追加到文件末尾
            # 风险：写入中途崩溃可能导致部分内容丢失，但不会影响已有内容
            with open(path, "a", encoding="utf-8") as f:
                f.write(content)
            action = "追加"
        else:
            # 覆盖模式：先写临时文件，再原子替换，防止写入中途崩溃导致文件损坏
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            tmp_path.write_text(content, encoding="utf-8")
            tmp_path.replace(path)  # 原子替换（Windows 上也是原子的同文件系统操作）
            action = "写入"

        return action, content_bytes

    @require_import("fpdf", "fpdf2")
    def _write_pdf(self, path: Path, content: str) -> tuple[str, int]:
        """
        写入 PDF 文件 - 使用 fpdf2 库。

        将文本内容按行写入 PDF，支持自动分页。中文字体需要额外配置。
        原子安全：先写入临时路径，再 replace() 替换目标文件。
        """
        from fpdf import FPDF

        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Helvetica", size=11)

        for line in content.split("\n"):
            pdf.cell(0, 6, text=line[:200], new_x="LMARGIN", new_y="NEXT")

        # 先写临时文件，再原子替换
        tmp_path = path.with_suffix(".pdf.tmp")
        pdf.output(str(tmp_path))
        tmp_path.replace(path)

        content_bytes = path.stat().st_size
        return "写入", content_bytes

    @require_import("docx", "python-docx")
    def _write_docx(self, path: Path, content: str) -> tuple[str, int]:
        """
        写入 Word (.docx) 文件 - 使用 python-docx 库。

        将文本按行写入 Word 文档，每行为一个段落。
        原子安全：先写入临时路径，再 replace() 替换目标文件。
        """
        from docx import Document
        from docx.shared import Pt

        doc = Document()

        for line in content.split("\n"):
            para = doc.add_paragraph(line)
            for run in para.runs:
                run.font.size = Pt(11)

        # 先写临时文件，再原子替换
        tmp_path = path.with_suffix(".docx.tmp")
        doc.save(tmp_path)
        tmp_path.replace(path)

        content_bytes = path.stat().st_size
        return "写入", content_bytes
