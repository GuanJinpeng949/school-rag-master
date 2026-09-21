"""文本分块器 - 整文档优先 + 长文档语义分块

分块策略:
1. 整文档优先：短文档(<=doc_keep_size)直接作为一整块，保证完整性
2. 长文档分块：超过阈值的文档按语义边界切分，每个chunk携带来源URL
3. 每个chunk都继承原文档的元数据(含source_url)，便于RAG返回来源链接

数据分布:
- 53%文档<=512字 → 整文档一块
- 36%文档512~2048字 → 整文档一块
- 10%文档>2048字 → 需要分块
"""
import hashlib
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from parser.base import ParsedDocument


@dataclass
class Chunk:
    """文本分块"""
    # 分块唯一ID
    chunk_id: str
    # 分块文本
    text: str
    # 所属文档ID
    doc_id: str
    # 分块序号（从0开始）
    chunk_index: int
    # 该文档的总分块数
    total_chunks: int

    # 继承自原文档的元数据
    source_url: str = ""
    source_site: str = ""
    title: str = ""
    content_type: str = ""
    publish_date: str = ""
    author: str = ""
    file_path: str = ""
    file_hash: str = ""
    attachments: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "doc_id": self.doc_id,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "source_url": self.source_url,
            "source_site": self.source_site,
            "title": self.title,
            "content_type": self.content_type,
            "publish_date": self.publish_date,
            "author": self.author,
            "file_path": self.file_path,
            "file_hash": self.file_hash,
            "attachments": self.attachments,
            "extra": self.extra,
        }


class TextChunker:
    """文本分块器 - 整文档优先策略

    策略说明:
    - 文档 <= doc_keep_size → 整文档作为一块(保持完整性，方便返回原始链接)
    - 文档 > doc_keep_size → 按语义边界分块(每个chunk仍携带source_url)
    - BGE-large-zh 最大支持512 token (~1024字)，但实际embedding可以处理更长文本
    - 整文档粒度在检索时可能降低精确度，但保证了上下文完整性和来源可溯
    """

    # 中文语义分隔符（按优先级排列）
    SEPARATORS = ["\n\n", "\n", "。", "；", "！", "？", "，", " ", ""]

    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 120,
                 min_chunk_size: int = 50, doc_keep_size: int = 2048):
        """
        Args:
            chunk_size: 每个分块的最大字符数（仅用于长文档分块）
            chunk_overlap: 相邻分块的重叠字符数
            min_chunk_size: 最小分块大小
            doc_keep_size: 整文档保留阈值(<=此值不分块，整文档作为一块)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.doc_keep_size = doc_keep_size

    def chunk_document(self, doc: ParsedDocument) -> list[Chunk]:
        """将文档分块

        策略:
        1. 短文档(<=doc_keep_size) → 整文档一块，保持完整性
        2. 长文档(>doc_keep_size) → 按语义边界分块
        """
        text = doc.text
        if not text or len(text.strip()) < self.min_chunk_size:
            return []

        text_len = len(text.strip())

        # 策略1：短文档整文档保留
        if text_len <= self.doc_keep_size:
            return [self._create_chunk(doc, text, 0, 1)]

        # 策略2：长文档按语义边界分块
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(
                separators=self.SEPARATORS,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                is_separator_regex=False,
            )
            text_chunks = splitter.split_text(text)
        except ImportError:
            text_chunks = self._simple_split(text)

        # 过滤过短的分块，合并到前一个
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            if len(chunk_text.strip()) < self.min_chunk_size and i > 0 and chunks:
                # 合并到前一个分块
                chunks[-1] = Chunk(
                    chunk_id=chunks[-1].chunk_id,
                    text=chunks[-1].text + "\n" + chunk_text.strip(),
                    doc_id=chunks[-1].doc_id,
                    chunk_index=chunks[-1].chunk_index,
                    total_chunks=chunks[-1].total_chunks,
                    source_url=chunks[-1].source_url,
                    source_site=chunks[-1].source_site,
                    title=chunks[-1].title,
                    content_type=chunks[-1].content_type,
                    publish_date=chunks[-1].publish_date,
                    author=chunks[-1].author,
                    file_path=chunks[-1].file_path,
                    file_hash=chunks[-1].file_hash,
                    attachments=chunks[-1].attachments,
                    extra=chunks[-1].extra,
                )
                continue
            chunks.append(self._create_chunk(doc, chunk_text, i, len(text_chunks)))

        # 修正 total_chunks
        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        return chunks

    def chunk_documents(self, docs: list[ParsedDocument]) -> list[Chunk]:
        """批量分块"""
        all_chunks = []
        whole_doc_count = 0
        split_doc_count = 0

        for doc in docs:
            text_len = len(doc.text.strip()) if doc.text else 0
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
            if chunks:
                if chunks[0].total_chunks == 1:
                    whole_doc_count += 1
                else:
                    split_doc_count += 1

        logger.info(
            f"分块完成: {len(docs)} 个文档 → {len(all_chunks)} 个分块 "
            f"(整文档保留: {whole_doc_count}, 需要切分: {split_doc_count})"
        )
        return all_chunks

    def _create_chunk(self, doc: ParsedDocument, text: str,
                      chunk_index: int, total_chunks: int) -> Chunk:
        """创建分块

        改进：对长文档的分块（chunk_index > 0 或 total_chunks > 1），
        在文本前添加文档标题作为上下文标题，帮助嵌入模型和LLM理解
        chunk 所属的文档主题。
        """
        raw = f"{doc.doc_id}#{chunk_index}"
        chunk_id = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]

        # 为多chunk文档添加上下文标题
        # 当文档被切分为多个chunk时，单个chunk可能缺少上下文信息
        # 在文本前加上"标题：xxx\n"可以帮助嵌入和LLM理解所属主题
        chunk_text = text.strip()
        if total_chunks > 1 and doc.title:
            # 避免重复添加（原文开头可能已包含标题）
            title_prefix = f"标题：{doc.title}\n"
            if not chunk_text.startswith(doc.title):
                chunk_text = title_prefix + chunk_text

        return Chunk(
            chunk_id=chunk_id,
            text=chunk_text,
            doc_id=doc.doc_id,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            source_url=doc.source_url,
            source_site=doc.source_site,
            title=doc.title,
            content_type=doc.content_type.value,
            publish_date=doc.publish_date,
            author=doc.author,
            file_path=doc.file_path,
            file_hash=doc.file_hash,
            attachments=doc.attachments,
            extra=doc.extra,
        )

    def _simple_split(self, text: str) -> list[str]:
        """简单分块（LangChain不可用时的回退方案）"""
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + self.chunk_size
            # 查找分隔符，避免跨分隔符切分
            if end < text_len:
                best_pos = end
                for sep in self.SEPARATORS:
                    if not sep:
                        continue
                    # 从当前起点往后50%到125%查找分隔符
                    pos = text.rfind(sep, start + self.chunk_size // 2, end + self.chunk_size // 4)
                    if pos > start:
                        best_pos = pos + len(sep)
                        break
                end = best_pos

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # 从终点往前一个chunk_overlap长度作为下一个起点，保留上下文。
            start = end - self.chunk_overlap
            if start <= 0 or start >= text_len:
                start = end

        return chunks
