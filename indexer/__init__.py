"""索引模块 - 文本分块、向量嵌入、向量存储、元数据管理、BM25检索、重排序

处理流程:
1. ParsedDocument → 文本分块(chunker) → 多个文本片段
2. 文本片段 → BGE嵌入(embedder) → 向量
3. 向量 + 元数据 → Chroma向量库(vector_store)
4. 元数据 → SQLite(metadata_store)
5. 全量文档 → BM25关键词索引(bm25_search)
6. 检索结果 → Cross-Encoder重排序(reranker)
"""
from indexer.chunker import TextChunker, Chunk
from indexer.embedder import Embedder
from indexer.vector_store import VectorStore
from indexer.metadata_store import MetadataStore
from indexer.bm25_search import BM25Search
from indexer.reranker import Reranker

__all__ = [
    "TextChunker", "Chunk", "Embedder", "VectorStore",
    "MetadataStore", "BM25Search", "Reranker",
]
