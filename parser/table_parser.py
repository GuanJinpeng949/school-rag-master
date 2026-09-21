"""表格解析器 - 从PDF中提取结构化表格数据

核心功能:
1. 使用pdfplumber提取PDF中的表格
2. 使用Camelot作为备选方案（对有线表格效果更好）
3. 输出Markdown格式的表格文本

设计说明:
- 学校官网中规章制度、通知等常以PDF附件形式发布
- 这些PDF中往往包含重要表格（如招生计划、课程安排等）
- 单独的表格解析器可以将表格转为结构化文本，便于检索
"""
import re
from pathlib import Path
from typing import Optional

from loguru import logger

from parser.base import BaseParser, ContentType, ParsedDocument


class TableParser(BaseParser):
    """表格专用解析器"""

    @property
    def supported_types(self) -> list[ContentType]:
        return [ContentType.TABLE]

    def parse(self, file_path: str | Path, metadata: Optional[dict] = None) -> list[ParsedDocument]:
        """从PDF文件中提取表格"""
        file_path = Path(file_path)
        metadata = metadata or {}

        if not file_path.exists():
            logger.warning(f"文件不存在: {file_path}")
            return []

        source_url = metadata.get("url", "")
        source_site = metadata.get("source_site", "")
        file_hash = metadata.get("file_hash", file_path.stem[:16])
        title = metadata.get("title", file_path.stem)

        # 提取表格
        tables = self._extract_tables_pdfplumber(file_path)
        if not tables:
            tables = self._extract_tables_camelot(file_path)

        if not tables:
            logger.debug(f"未找到表格: {file_path.name}")
            return []

        # 每个表格生成一个文档
        docs = []
        for i, table_text in enumerate(tables):
            doc_id = self._generate_doc_id(source_url, file_hash, chunk_index=i)
            doc = ParsedDocument(
                doc_id=doc_id,
                text=table_text,
                source_url=source_url,
                source_site=source_site,
                title=f"{title} - 表格{i+1}",
                content_type=ContentType.TABLE,
                publish_date=metadata.get("publish_date", ""),
                author=metadata.get("author", ""),
                file_path=str(file_path),
                file_hash=file_hash,
                chunk_index=i,
                total_chunks=len(tables),
                attachments=metadata.get("attachments", []),
                extra={"table_index": i + 1, "total_tables": len(tables)},
            )
            docs.append(doc)

        return docs

    def _extract_tables_pdfplumber(self, file_path: Path) -> list[str]:
        """使用pdfplumber提取表格"""
        try:
            import pdfplumber
        except ImportError:
            logger.debug("pdfplumber未安装")
            return []

        tables_md = []
        try:
            with pdfplumber.open(str(file_path)) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    tables = page.extract_tables()
                    for table in tables:
                        if not table or len(table) < 2:
                            continue
                        md = self._table_to_markdown(table)
                        if md:
                            tables_md.append(md)
        except Exception as e:
            logger.debug(f"pdfplumber表格提取失败: {file_path} - {e}")

        return tables_md

    def _extract_tables_camelot(self, file_path: Path) -> list[str]:
        """使用Camelot提取表格（备选方案）"""
        try:
            import camelot
        except ImportError:
            logger.debug("Camelot未安装")
            return []

        tables_md = []
        try:
            # 先尝试流模式（适用于无边框表格）
            tables = camelot.read_pdf(str(file_path), flavor="stream", pages="all")
            if not tables:
                # 再尝试格子模式（适用于有边框表格）
                tables = camelot.read_pdf(str(file_path), flavor="lattice", pages="all")

            for table in tables:
                df = table.df
                if df.empty or len(df) < 2:
                    continue
                md = self._dataframe_to_markdown(df)
                if md:
                    tables_md.append(md)
        except Exception as e:
            logger.debug(f"Camelot表格提取失败: {file_path} - {e}")

        return tables_md

    def _table_to_markdown(self, table: list[list]) -> str:
        """将二维列表转为Markdown表格"""
        if not table:
            return ""

        cleaned = []
        for row in table:
            cleaned_row = [(cell or "").strip().replace("\n", " ").replace("|", "｜") for cell in row]
            cleaned.append(cleaned_row)

        max_cols = max(len(row) for row in cleaned) if cleaned else 0
        if max_cols == 0:
            return ""

        for row in cleaned:
            while len(row) < max_cols:
                row.append("")

        lines = []
        lines.append("| " + " | ".join(cleaned[0]) + " |")
        lines.append("| " + " | ".join(["---"] * max_cols) + " |")
        for row in cleaned[1:]:
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)

    def _dataframe_to_markdown(self, df) -> str:
        """将pandas DataFrame转为Markdown表格"""
        try:
            # 清理单元格内容
            df = df.fillna("")
            lines = []

            # 表头
            header = "| " + " | ".join(str(col).replace("|", "｜") for col in df.columns) + " |"
            separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"
            lines.append(header)
            lines.append(separator)

            # 数据行
            for _, row in df.iterrows():
                line = "| " + " | ".join(str(v).replace("|", "｜") for v in row) + " |"
                lines.append(line)

            return "\n".join(lines)
        except Exception:
            return ""
