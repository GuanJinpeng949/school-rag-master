"""文档解析模块 - 将原始文件解析为结构化文档

支持的格式:
- HTML: BeautifulSoup正文提取 + readability-lxml兜底
- PDF: PyMuPDF文本提取 + pdfplumber表格提取
- 图片: PaddleOCR文字识别
- 表格: PDF表格结构化提取

所有解析器继承 BaseParser，返回统一的 ParsedDocument 列表。
"""
from parser.base import BaseParser, ParsedDocument
from parser.router import ParserRouter

__all__ = ["BaseParser", "ParsedDocument", "ParserRouter"]
