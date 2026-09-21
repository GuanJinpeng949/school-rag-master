"""依赖注入 - 数据库会话、RAG Chain实例"""

import os

from loguru import logger

# 离线模式环境变量（在import前设置）
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from config.settings import settings
from rag.chain import RAGChain
from rag.retriever import Retriever
from indexer.vector_store import VectorStore
from indexer.metadata_store import MetadataStore
from indexer.bm25_search import BM25Search
from indexer.reranker import Reranker


# 全局单例（懒加载）
_rag_chain: RAGChain = None
_vector_store: VectorStore = None
_metadata_store: MetadataStore = None
_bm25_search: BM25Search = None
_reranker: Reranker = None


def get_rag_chain() -> RAGChain:
    """获取RAGChain单例（配置驱动）"""
    global _rag_chain
    if _rag_chain is None:
        logger.info("初始化RAGChain...")
        retriever = Retriever(
            hybrid=settings.retrieval.hybrid_search,
            rrf_k=settings.retrieval.rrf_k,
            use_reranker=settings.retrieval.use_reranker,
            score_threshold=settings.retrieval.score_threshold,
            default_top_k=settings.retrieval.default_top_k,
            max_chunks_per_doc=settings.retrieval.max_chunks_per_doc,
            use_query_rewrite=settings.retrieval.use_query_rewrite,
            use_query_decompose=settings.retrieval.use_query_decompose,
            bm25_search=get_bm25_search(),
            reranker=get_reranker(),
        )
        _rag_chain = RAGChain(retriever=retriever)
    return _rag_chain


def get_bm25_search() -> BM25Search:
    """获取BM25Search单例"""
    global _bm25_search
    if _bm25_search is None:
        _bm25_search = BM25Search()
    return _bm25_search


def get_reranker() -> Reranker:
    """获取Reranker单例"""
    global _reranker
    if _reranker is None:
        _reranker = Reranker(model_name=settings.retrieval.reranker_model)
    return _reranker


def get_vector_store() -> VectorStore:
    """获取VectorStore单例"""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def get_metadata_store() -> MetadataStore:
    """获取MetadataStore单例"""
    global _metadata_store
    if _metadata_store is None:
        _metadata_store = MetadataStore()
    return _metadata_store
