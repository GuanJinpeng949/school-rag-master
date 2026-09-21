"""解析路由 - 根据文件类型/MIME分发到对应解析器

核心功能:
1. 自动检测文件类型（基于扩展名和路径）
2. 路由到对应的解析器
3. 支持从爬虫JSON结果批量解析
4. 统一的解析入口

设计说明:
- Router是解析模块的统一入口，索引模块只需调用Router即可
- Router会根据文件类型自动选择合适的解析器
- 支持从crawl_full.json批量加载和解析
"""
import json
from pathlib import Path
from typing import Optional

from loguru import logger

from parser.base import BaseParser, ContentType, ParsedDocument
from parser.html_parser import HTMLParser
from parser.pdf_parser import PDFParser
from parser.image_parser import ImageParser
from parser.table_parser import TableParser


class ParserRouter:
    """解析路由器 - 统一入口"""

    def __init__(self):
        self._parsers: dict[ContentType, BaseParser] = {}
        self._register_parsers()

    def _register_parsers(self):
        """注册所有解析器"""
        parsers = [
            HTMLParser(),
            PDFParser(),
            ImageParser(),
            TableParser(),
        ]
        for parser in parsers:
            for content_type in parser.supported_types:
                self._parsers[content_type] = parser

    def detect_content_type(self, file_path: str | Path) -> ContentType:
        """根据文件路径检测内容类型"""
        file_path = Path(file_path)
        ext = file_path.suffix.lower()
        parent = file_path.parent.name.lower()

        # 根据父目录判断
        if parent == "html":
            return ContentType.HTML
        elif parent == "pdf":
            return ContentType.PDF
        elif parent == "images":
            return ContentType.IMAGE

        # 根据扩展名判断
        ext_map = {
            ".htm": ContentType.HTML,
            ".html": ContentType.HTML,
            ".jsp": ContentType.HTML,
            ".pdf": ContentType.PDF,
            ".jpg": ContentType.IMAGE,
            ".jpeg": ContentType.IMAGE,
            ".png": ContentType.IMAGE,
            ".gif": ContentType.IMAGE,
            ".bmp": ContentType.IMAGE,
            ".webp": ContentType.IMAGE,
            ".doc": ContentType.DOC,
            ".docx": ContentType.DOC,
            ".xls": ContentType.DOC,
            ".xlsx": ContentType.DOC,
            ".ppt": ContentType.DOC,
            ".pptx": ContentType.DOC,
        }

        return ext_map.get(ext, ContentType.OTHER)

    def parse_file(self, file_path: str | Path, metadata: Optional[dict] = None) -> list[ParsedDocument]:
        """解析单个文件

        Args:
            file_path: 文件路径
            metadata: 可选的元数据（来自爬虫阶段）

        Returns:
            ParsedDocument 列表
        """
        content_type = self.detect_content_type(file_path)
        parser = self._parsers.get(content_type)

        if not parser:
            logger.warning(f"无对应解析器: content_type={content_type}, file={file_path}")
            return []

        try:
            return parser.parse(file_path, metadata)
        except Exception as e:
            logger.error(f"解析失败: {file_path} - {e}")
            return []

    def parse_crawl_results(self, json_path: str | Path, only_detail: bool = True,
                            only_with_body: bool = False) -> list[ParsedDocument]:
        """从爬虫结果JSON批量解析

        Args:
            json_path: 爬虫结果JSON文件路径（如 data/crawl_full.json）
            only_detail: 是否只解析详情页
            only_with_body: 是否只解析有正文的页面

        Returns:
            所有解析后的文档列表
        """
        json_path = Path(json_path)
        if not json_path.exists():
            logger.error(f"爬虫结果文件不存在: {json_path}")
            return []

        with open(json_path, "r", encoding="utf-8") as f:
            crawl_data = json.load(f)

        logger.info(f"加载爬虫结果: {len(crawl_data)} 条")

        all_docs = []
        parsed_count = 0
        skipped_count = 0
        failed_count = 0

        for item in crawl_data:
            # 过滤条件
            if only_detail and not item.get("metadata", {}).get("is_detail_page"):
                skipped_count += 1
                continue

            if only_with_body and not item.get("metadata", {}).get("body_text"):
                skipped_count += 1
                continue

            # 跳过有错误的条目
            if item.get("error"):
                skipped_count += 1
                continue

            file_path = item.get("file_path")
            if not file_path or not Path(file_path).exists():
                skipped_count += 1
                continue

            # 构建元数据
            metadata = {
                "url": item.get("url", ""),
                "title": item.get("title", ""),
                "source_site": item.get("metadata", {}).get("source_site", ""),
                "file_hash": item.get("file_hash", ""),
                "publish_date": item.get("metadata", {}).get("publish_date", ""),
                "author": item.get("metadata", {}).get("author", ""),
                "attachments": item.get("metadata", {}).get("attachments", []),
            }

            # 解析
            docs = self.parse_file(file_path, metadata)
            if docs:
                all_docs.extend(docs)
                parsed_count += 1
            else:
                failed_count += 1

        logger.info(
            f"解析完成: 成功={parsed_count}, 跳过={skipped_count}, "
            f"失败={failed_count}, 文档数={len(all_docs)}"
        )

        return all_docs

    # 使用data/raw 目录中的实际文件，不使用爬虫元数据
    def parse_all_raw_files(self, raw_dir: str | Path) -> list[ParsedDocument]:
        """扫描原始文件目录，解析所有文件

        Args:
            raw_dir: 原始文件根目录（如 data/raw）

        Returns:
            所有解析后的文档列表
        """
        raw_dir = Path(raw_dir)
        if not raw_dir.exists():
            logger.error(f"原始文件目录不存在: {raw_dir}")
            return []

        all_docs = []
        supported_exts = {
            ".htm", ".html", ".jsp",  # HTML
            ".pdf",  # PDF
            ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp",  # Image
        }

        for subdir in ["html", "pdf", "images"]:
            subdir_path = raw_dir / subdir
            if not subdir_path.exists():
                continue

            for file_path in subdir_path.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in supported_exts:
                    docs = self.parse_file(file_path)
                    all_docs.extend(docs)

        logger.info(f"原始文件解析完成: 文档数={len(all_docs)}")
        return all_docs
