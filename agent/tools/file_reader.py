"""
File Reader 工具 - 读取本地文件内容

让 Agent 具备读取本地文件的能力，支持：
- 文本文件（.txt, .md, .py, .json, .csv 等）
- PDF 文档（.pdf）
- Word 文档（.docx）
- 限制最大读取行数/字符数，防止超大文件撑爆上下文
- 安全的文件存在性检查

使用示例：
    file_path: "C:/Users/xxx/data.csv"
    max_lines: 100
    返回: 文件内容文本
"""

from pathlib import Path
from agent.tools.base import BaseTool, ToolResult


# ============================================================
# 允许读取的文件扩展名白名单
# 防止 Agent 尝试读取二进制文件（如图片、可执行文件等）
# ============================================================
ALLOWED_EXTENSIONS = {
    ".txt", ".md", ".py", ".json", ".csv", ".xml", ".html",
    ".css", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".go",
    ".rs", ".rb", ".php", ".sh", ".bat", ".yaml", ".yml",
    ".toml", ".ini", ".cfg", ".conf", ".log", ".sql",
    ".r", ".swift", ".kt", ".scala", ".lua", ".env",
    ".gitignore", ".dockerfile",
    ".pdf", ".docx",
}


class FileReaderTool(BaseTool):
    """
    文件读取工具 - 让 Agent 能读取本地文本文件

    工作原理：
    1. 接收文件路径参数
    2. 检查文件是否存在、是否为文本文件
    3. 检查扩展名是否在白名单中
    4. 按行读取文件内容，最多读取 max_lines 行
    5. 返回格式化的文件内容

    安全机制：
    - 扩展名白名单过滤
    - 最大行数限制（默认 200 行）
    - UTF-8 编码强制
    - 文件大小保护（超过 10MB 拒绝读取）
    """

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return (
            "读取指定路径的文件内容。"
            "支持 .txt, .md, .py, .json, .csv, .xml, .yaml 等文本文件，"
            "以及 .pdf 文档和 .docx Word 文档。"
            "可用于分析代码、查看配置、读取数据文件、检查日志、阅读文档等。"
            "文件过大时会自动截断，可通过 max_lines 参数控制最大行数。"
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "要读取的文件路径（支持绝对路径和相对路径）",
                },
                "max_lines": {
                    "type": "integer",
                    "description": "最大读取行数，默认 200。文件超过此行数会自动截断。",
                    "default": 200,
                },
            },
            "required": ["file_path"],
        }

    def execute(self, file_path: str, max_lines: int = 200) -> ToolResult:
        """
        执行文件读取

        执行流程：
        1. 将路径转为 Path 对象并解析为绝对路径
        2. 检查文件是否存在
        3. 检查是否为文件（非目录）
        4. 检查扩展名白名单
        5. 检查文件大小（< 10MB）
        6. 逐行读取，限制最大行数
        7. 返回格式化结果

        Args:
            file_path: 文件路径字符串
            max_lines: 最大读取行数

        Returns:
            ToolResult: 包含文件内容或错误信息
        """
        try:
            # 解析路径为绝对路径（支持 ~ 展开）
            path = Path(file_path).expanduser().resolve()

            # 检查文件是否存在
            if not path.exists():
                return ToolResult(
                    success=False, result="",
                    error=f"文件不存在: {path}",
                )

            # 检查是否为文件（排除目录）
            if not path.is_file():
                return ToolResult(
                    success=False, result="",
                    error=f"路径不是文件: {path}",
                )

            # 检查扩展名白名单（无扩展名的文件如 .env 也允许）
            suffix = path.suffix.lower()
            if suffix and suffix not in ALLOWED_EXTENSIONS:
                return ToolResult(
                    success=False, result="",
                    error=f"不支持的文件类型: {suffix}。支持的类型: {', '.join(sorted(ALLOWED_EXTENSIONS)[:12])}...",
                )

            # 检查文件大小（10MB 保护）
            file_size = path.stat().st_size
            if file_size > 10 * 1024 * 1024:
                return ToolResult(
                    success=False, result="",
                    error=f"文件过大: {file_size / 1024 / 1024:.1f}MB，超过 10MB 限制",
                )

            # ============================================================
            # 根据文件类型选择不同的读取方式
            # ============================================================
            suffix = path.suffix.lower()

            if suffix == ".pdf":
                content = self._read_pdf(path, max_lines)
            elif suffix == ".docx":
                content = self._read_docx(path, max_lines)
            else:
                # 文本文件: 使用原有方式
                content = self._read_text(path, max_lines)

            # 封装结果
            lines = content.splitlines()
            total = len(lines)
            return ToolResult(
                success=True,
                result=content,
                metadata={
                    "path": str(path),
                    "total_lines": total,
                    "file_size": file_size,
                    "displayed_lines": min(total, max_lines),
                    "file_type": suffix if suffix else "text",
                },
            )

        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            return ToolResult(
                success=False, result="",
                error=f"读取文件失败: {error_msg}",
            )

    # ============================================================
    # 专用读取方法
    # ============================================================

    def _read_text(self, path: Path, max_lines: int) -> str:
        """读取纯文本文件"""
        content = path.read_text(encoding="utf-8")
        lines = content.splitlines()
        total_lines = len(lines)

        truncated_note = ""
        if total_lines > max_lines:
            lines = lines[:max_lines]
            truncated_note = f"\n\n[已截断: 文件共 {total_lines} 行，仅显示前 {max_lines} 行]"

        return (
            f"文件: {path}\n"
            f"大小: {path.stat().st_size:,} 字节 | 行数: {total_lines}\n"
            f"{'=' * 50}\n"
            + "\n".join(lines)
            + truncated_note
        )

    def _read_pdf(self, path: Path, max_lines: int) -> str:
        """读取 PDF 文件 - 使用 PyMuPDF (fitz) 提取文本"""
        import fitz  # PyMuPDF

        doc = fitz.open(path)
        total_pages = len(doc)
        lines = []

        for page_num in range(total_pages):
            page = doc[page_num]
            text = page.get_text()
            page_lines = text.splitlines()
            # 过滤空行
            page_lines = [l for l in page_lines if l.strip()]

            if page_lines:
                lines.append(f"--- 第 {page_num + 1} 页 ---")
                lines.extend(page_lines)

            # 检查是否超过行数限制
            if len(lines) > max_lines:
                lines = lines[:max_lines]
                lines.append(f"\n[已截断: PDF 共 {total_pages} 页，仅显示前 {max_lines} 行]")
                break

        doc.close()

        total_lines = len(lines)
        return (
            f"文件: {path}\n"
            f"大小: {path.stat().st_size:,} 字节 | PDF 页数: {total_pages} | 显示行数: {total_lines}\n"
            f"{'=' * 50}\n"
            + "\n".join(lines)
        )

    def _read_docx(self, path: Path, max_lines: int) -> str:
        """读取 Word (.docx) 文件 - 使用 python-docx 提取文本"""
        from docx import Document

        doc = Document(path)
        lines = []

        # 提取段落文本
        for para in doc.paragraphs:
            if para.text.strip():
                lines.append(para.text)
                if len(lines) > max_lines:
                    lines = lines[:max_lines]
                    lines.append(f"\n[已截断: 仅显示前 {max_lines} 行]")
                    break

        # 如果段落不够，尝试提取表格内容
        if len(lines) < max_lines:
            for table in doc.tables:
                if len(lines) > max_lines:
                    break
                lines.append("")
                lines.append("--- 表格 ---")
                for row in table.rows:
                    cells = [cell.text.strip() for cell in row.cells]
                    lines.append(" | ".join(cells))
                    if len(lines) > max_lines:
                        lines = lines[:max_lines]
                        lines.append(f"\n[已截断: 仅显示前 {max_lines} 行]")
                        break

        total_lines = len(lines)
        return (
            f"文件: {path}\n"
            f"大小: {path.stat().st_size:,} 字节 | 段落数: {len(doc.paragraphs)} | 显示行数: {total_lines}\n"
            f"{'=' * 50}\n"
            + "\n".join(lines)
        )
