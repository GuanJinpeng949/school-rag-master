"""PDF解析器 - 从PDF文件中提取文本和表格

核心功能:
1. 文本提取：使用PyMuPDF(fitz)提取PDF中的文本内容
2. 表格提取：使用pdfplumber提取PDF中的表格为结构化数据
3. 多页文档：每页生成独立的文本段，同时合并为完整文档
4. OCR兜底：对扫描件图片页使用PaddleOCR识别文字

设计说明:
- PDF在爬虫阶段只保存了文件，没有提取内容
- 解析器负责完整的文本和表格提取
- 对于扫描件PDF，自动启用OCR识别
"""
import re
from pathlib import Path
from typing import Optional

from loguru import logger

from parser.base import BaseParser, ContentType, ParsedDocument


# 公文标题的常见结尾词，用于判断跨行标题是否已经结束
_TITLE_SUFFIXES = (
    "的通知", "的通报", "的意见", "的批复", "的决定", "的函",
    "通知", "办法", "规定", "细则", "方案", "规则", "制度",
)


class PDFParser(BaseParser):
    """PDF文档解析器"""

    # 文本页的最小字符数阈值（少于此值认为是扫描件/图片页）
    MIN_TEXT_CHARS = 50

    @property
    def supported_types(self) -> list[ContentType]:
        return [ContentType.PDF]

    def parse(self, file_path: str | Path, metadata: Optional[dict] = None) -> list[ParsedDocument]:
        """解析PDF文件，提取文本和表格

        Returns:
            通常返回1个 ParsedDocument（完整文档），
            如果文档很长可能分多个（每个section一个）
        """
        file_path = Path(file_path)
        metadata = metadata or {}

        if not file_path.exists():
            logger.warning(f"PDF文件不存在: {file_path}")
            return []

        source_url = metadata.get("url", "")
        source_site = metadata.get("source_site", "")
        file_hash = metadata.get("file_hash", file_path.stem[:16])
        title = metadata.get("title", file_path.stem)

        # 提取文本
        full_text = self._extract_text(file_path)
        if not full_text:
            logger.debug(f"PDF文本提取为空，尝试OCR: {file_path.name}")
            full_text = self._extract_with_ocr(file_path)

        # 提取表格
        tables_text = self._extract_tables(file_path)

        # 合并文本和表格
        if tables_text:
            full_text = f"{full_text}\n\n---\n\n【PDF表格内容】\n\n{tables_text}" if full_text else tables_text

        if not full_text:
            logger.debug(f"PDF内容为空: {file_path.name}")
            return []

        # 爬虫给出的 title 通常就是文件名（如 AB8EAC68...256DF5.pdf），
        # 展示给用户没有意义，改用正文中第一行有意义的内容
        if self._is_filename_title(title, file_path):
            title = self._guess_title_from_text(full_text) or title

        doc_id = self._generate_doc_id(source_url, file_hash)

        doc = ParsedDocument(
            doc_id=doc_id,
            text=full_text,
            source_url=source_url,
            source_site=source_site,
            title=title,
            content_type=ContentType.PDF,
            publish_date=metadata.get("publish_date", ""),
            author=metadata.get("author", ""),
            file_path=str(file_path),
            file_hash=file_hash,
            chunk_index=0,
            total_chunks=1,
            attachments=metadata.get("attachments", []),
            extra={"page_count": self._get_page_count(file_path)},
        )

        return [doc]

    @staticmethod
    def _is_filename_title(title: str, file_path: Path) -> bool:
        """判断标题是否只是文件名"""
        if not title:
            return True
        return title == file_path.name or title.lower().endswith(".pdf")

    @staticmethod
    def _guess_title_from_text(text: str) -> str:
        """从正文中推断标题

        学校官网的公文版式很固定：第1行是发文机关标志（如「中国矿业大学文件」），
        第2行是发文字号（如「中矿委〔2022〕24 号」），第3行起才是真正的标题，
        且标题常跨多行、以「各…：」这类主送机关行结束。非公文版式则首行即标题。
        """
        lines = []
        for raw in text.splitlines():
            line = re.sub(r"\s+", " ", raw).strip()
            if len(line) < 6:
                continue
            if re.fullmatch(r"[—–\-\s]*\d+[—–\-\s]*", line):       # 页码，如 "- 2 -"
                continue
            if "阅知" in line or "注意保管" in line:                # 保密提示
                continue
            lines.append(line)
            if len(lines) >= 8:
                break

        if not lines:
            return ""

        if not lines[0].endswith("文件"):
            return lines[0][:60]

        # 公文版式：跳过发文机关标志与发文字号，拼接到主送机关行为止
        picked = []
        for line in lines[1:]:
            if len(line) <= 20 and line.endswith("号") and re.search(r"\d{4}", line):
                continue  # 发文字号
            if line.endswith(("：", ":")):
                break  # 主送机关，标题到此结束
            picked.append(line)
            joined = "".join(picked)
            if joined.endswith(_TITLE_SUFFIXES) or len(joined) >= 30:
                break

        title = "".join(picked)
        # 标题与正文被排版在同一行时，截到标题结束处
        for suffix in _TITLE_SUFFIXES:
            idx = title.find(suffix)
            if idx >= 6:
                return title[:idx + len(suffix)][:60]
        return (title or lines[0])[:60]

    def _get_page_count(self, file_path: Path) -> int:
        """获取PDF页数"""
        try:
            import fitz
            doc = fitz.open(str(file_path))
            count = len(doc)
            doc.close()
            return count
        except Exception:
            return 0

    def _extract_text(self, file_path: Path) -> str:
        """使用PyMuPDF提取PDF文本"""
        try:
            import fitz
        except ImportError:
            logger.warning("PyMuPDF未安装，无法提取PDF文本")
            return ""

        try:
            doc = fitz.open(str(file_path))
            text_parts = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text("text", flags=fitz.TEXT_PRESERVE_WHITESPACE)

                if page_text.strip():
                    text_parts.append(page_text.strip())

            doc.close()
            return "\n\n".join(text_parts)

        except Exception as e:
            logger.error(f"PDF文本提取失败: {file_path} - {e}")
            return ""

    def _extract_tables(self, file_path: Path) -> str:
        """使用pdfplumber提取PDF表格"""
        try:
            import pdfplumber
        except ImportError:
            logger.debug("pdfplumber未安装，跳过表格提取")
            return ""

        try:
            tables_md = []
            with pdfplumber.open(str(file_path)) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    tables = page.extract_tables()
                    for table_num, table in enumerate(tables):
                        if not table or len(table) < 2:
                            continue
                        md = self._table_to_markdown(table)
                        if md:
                            tables_md.append(f"第{page_num+1}页 表格{table_num+1}:\n{md}")

            return "\n\n".join(tables_md)

        except Exception as e:
            logger.debug(f"PDF表格提取失败: {file_path} - {e}")
            return ""

    def _table_to_markdown(self, table: list[list]) -> str:
        """将表格数据转为Markdown格式"""
        if not table:
            return ""

        # 清理单元格内容
        cleaned = []
        for row in table:
            cleaned_row = []
            for cell in row:
                text = (cell or "").strip().replace("\n", " ").replace("|", "｜")
                cleaned_row.append(text)
            cleaned.append(cleaned_row)

        # 统一列数
        max_cols = max(len(row) for row in cleaned)
        for row in cleaned:
            while len(row) < max_cols:
                row.append("")

        # 生成Markdown
        lines = []
        lines.append("| " + " | ".join(cleaned[0]) + " |")
        lines.append("| " + " | ".join(["---"] * max_cols) + " |")
        for row in cleaned[1:]:
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)

    def _extract_with_ocr(self, file_path: Path) -> str:
        """对扫描件PDF使用OCR提取文字"""
        try:
            import fitz
            from PIL import Image
            import io
        except ImportError:
            logger.debug("PyMuPDF/Pillow未安装，无法进行OCR")
            return ""

        try:
            from paddleocr import PaddleOCR
        except ImportError:
            logger.debug("PaddleOCR未安装，无法进行OCR")
            return ""

        try:
            ocr = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
            doc = fitz.open(str(file_path))
            text_parts = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                # 先检查是否有文本
                page_text = page.get_text("text").strip()
                if len(page_text) >= self.MIN_TEXT_CHARS:
                    continue  # 有足够文本，不需要OCR

                # 将页面渲染为图片
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x放大提高OCR精度
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))

                # OCR识别
                result = ocr.ocr(img, cls=True)
                if result and result[0]:
                    page_texts = []
                    for line in result[0]:
                        if line and len(line) >= 2:
                            page_texts.append(line[1][0])  # 提取文字
                    if page_texts:
                        text_parts.append("\n".join(page_texts))

            doc.close()
            return "\n\n".join(text_parts)

        except Exception as e:
            logger.debug(f"PDF OCR提取失败: {file_path} - {e}")
            return ""
