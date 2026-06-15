"""
File Writer 工具 - 写入内容到本地文件

让 Agent 具备写入本地文件的能力，支持：
- 文本文件（.txt, .md, .py, .json, .csv 等）
- Word 文档（.docx）
- PDF 文档（.pdf）
- 覆盖写入和追加写入
- 自动创建目录（如果不存在）
- 写入结果确认

使用示例：
    file_path: "C:/Users/xxx/notes.txt"
    content: "这是一条笔记"
    append: false
    返回: "文件写入成功"
"""

from pathlib import Path
from agent.tools.base import BaseTool, ToolResult


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
    文件写入工具 - 让 Agent 能写入本地文本文件

    工作原理：
    1. 接收文件路径和内容参数
    2. 检查路径安全性（扩展名白名单 + 路径黑名单）
    3. 自动创建父目录（如果不存在）
    4. 以覆盖或追加模式写入文件
    5. 返回写入结果确认

    安全机制：
    - 扩展名白名单过滤
    - 系统路径黑名单保护
    - UTF-8 编码写入
    - 写入大小限制（1MB）
    """

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return (
            "将文本内容写入指定的本地文件。"
            "支持 .txt, .md, .py, .json, .csv, .yaml 等文本文件，"
            "以及 .pdf 文档和 .docx Word 文档。"
            "如果文件不存在会自动创建，如果存在则根据 append 参数决定覆盖或追加。"
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
            },
            "required": ["file_path", "content"],
        }

    def execute(self, file_path: str, content: str, append: bool = False) -> ToolResult:
        """
        执行文件写入

        执行流程：
        1. 解析路径为绝对路径
        2. 安全检查：扩展名白名单
        3. 安全检查：系统路径黑名单
        4. 安全检查：写入大小限制
        5. 自动创建父目录
        6. 写入文件（覆盖或追加）
        7. 返回结果确认

        Args:
            file_path: 目标文件路径
            content: 要写入的内容
            append: 是否追加模式

        Returns:
            ToolResult: 写入结果或错误信息
        """
        try:
            # 解析路径
            path = Path(file_path).expanduser().resolve()

            # 检查扩展名白名单
            suffix = path.suffix.lower()
            if suffix and suffix not in ALLOWED_WRITE_EXTENSIONS:
                return ToolResult(
                    success=False, result="",
                    error=f"不支持写入的文件类型: {suffix}。支持的类型: {', '.join(sorted(ALLOWED_WRITE_EXTENSIONS)[:10])}...",
                )

            # 检查系统路径黑名单
            path_str = str(path).lower()
            for blocked in BLOCKED_PATHS:
                if path_str.startswith(blocked.lower()):
                    return ToolResult(
                        success=False, result="",
                        error=f"安全限制：不允许写入系统目录下的文件: {path}",
                    )

            # 自动创建父目录
            path.parent.mkdir(parents=True, exist_ok=True)

            # ============================================================
            # 根据文件类型选择不同的写入方式
            # ============================================================
            if suffix == ".pdf":
                action, content_bytes = self._write_pdf(path, content)
            elif suffix == ".docx":
                action, content_bytes = self._write_docx(path, content)
            else:
                action, content_bytes = self._write_text(path, content, append)

            # 返回结果
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
        """写入纯文本文件"""
        # 检查写入大小限制（1MB）
        content_bytes = len(content.encode("utf-8"))
        if content_bytes > 1024 * 1024:
            raise ValueError(f"写入内容过大: {content_bytes / 1024 / 1024:.1f}MB，超过 1MB 限制")

        mode = "a" if append else "w"
        with open(path, mode, encoding="utf-8") as f:
            f.write(content)

        action = "追加" if append else "写入"
        return action, content_bytes

    def _write_pdf(self, path: Path, content: str) -> tuple[str, int]:
        """写入 PDF 文件 - 使用 fpdf2"""
        from fpdf import FPDF

        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Helvetica", size=11)

        # 按行写入 PDF（自动分页）
        for line in content.split("\n"):
            # 空行也写入以保持格式
            pdf.cell(0, 6, text=line[:200], new_x="LMARGIN", new_y="NEXT")

        pdf.output(str(path))

        content_bytes = path.stat().st_size
        return "写入", content_bytes

    def _write_docx(self, path: Path, content: str) -> tuple[str, int]:
        """写入 Word (.docx) 文件 - 使用 python-docx"""
        from docx import Document
        from docx.shared import Pt

        doc = Document()

        # 按行写入（空行作为段落分隔）
        for line in content.split("\n"):
            para = doc.add_paragraph(line)
            # 设置默认字号
            for run in para.runs:
                run.font.size = Pt(11)

        doc.save(path)

        content_bytes = path.stat().st_size
        return "写入", content_bytes
