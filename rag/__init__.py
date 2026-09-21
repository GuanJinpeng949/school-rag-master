"""RAG核心模块 - 检索增强生成

处理流程:
1. 用户查询 → Embedding → 向量检索(retriever) → Top-K相关文档
2. 检索结果 + Prompt模板(generator) → LLM生成回答
3. 回答 + 来源链接 → 返回给用户
"""
from rag.retriever import Retriever, SearchResult
from rag.generator import Generator
from rag.chain import RAGChain

__all__ = ["Retriever", "SearchResult", "Generator", "RAGChain"]
