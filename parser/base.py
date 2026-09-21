"""解析器基类 - 定义统一接口和数据模型"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class ContentType(str, Enum):
    """内容类型枚举"""
    HTML = "html"
    PDF = "pdf"
    IMAGE = "image"
    TABLE = "table"
    DOC = "doc"
    OTHER = "other"


@dataclass
class ParsedDocument:
    """解析后的结构化文档

    这是解析器的统一输出，供下游索引模块使用。
    一个原始文件可能产生多个 ParsedDocument（如 PDF 每页一个文档、
    HTML 正文+表格各一个文档）。
    """
    # 文档唯一ID（基于 source_url + chunk_index 生成）
    doc_id: str

    # 文本内容
    text: str

    # 元数据
    source_url: str = ""
    source_site: str = ""
    title: str = ""
    content_type: ContentType = ContentType.OTHER
    publish_date: str = ""
    author: str = ""
    file_path: str = ""
    file_hash: str = ""

    # 分块信息（同一个原始文件可能被分成多个文档）
    chunk_index: int = 0
    total_chunks: int = 1

    # 附件信息（原页面关联的附件列表）
    attachments: list[dict] = field(default_factory=list)

    # 额外元数据（格式特定的信息，如PDF页码、表格行列数等）
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """转换为字典，便于序列化"""
        return {
            "doc_id": self.doc_id,
            "text": self.text,
            "source_url": self.source_url,
            "source_site": self.source_site,
            "title": self.title,
            "content_type": self.content_type.value,
            "publish_date": self.publish_date,
            "author": self.author,
            "file_path": self.file_path,
            "file_hash": self.file_hash,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "attachments": self.attachments,
            "extra": self.extra,
        }


class BaseParser(ABC):
    """解析器基类

    所有解析器必须实现 parse 方法，接受文件路径和可选的元数据，
    返回 ParsedDocument 列表。
    """

    @abstractmethod
    def parse(self, file_path: str | Path, metadata: Optional[dict] = None) -> list[ParsedDocument]:
        """解析文件，返回结构化文档列表

        Args:
            file_path: 文件路径
            metadata: 可选的元数据（来自爬虫阶段的已知信息，如 url、title 等）

        Returns:
            ParsedDocument 列表（一个文件可能产生多个文档片段）
        """
        ...

    @property
    @abstractmethod
    def supported_types(self) -> list[ContentType]:
        """该解析器支持的内容类型"""
        ...

    def _generate_doc_id(self, source_url: str, file_hash: str, chunk_index: int = 0) -> str:
        """生成文档唯一ID

        使用 source_url + chunk_index 组合，确保同一来源的不同片段有不同ID
        """
        import hashlib
        raw = f"{source_url}#{chunk_index}" if source_url else f"{file_hash}#{chunk_index}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]
