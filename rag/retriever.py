"""检索器 - Hybrid Search(向量+BM25) + RRF融合 + 重排序 + 结果去重

核心功能:
1. 查询预处理：查询改写（口语→书面）+ 多查询分解（复杂问题拆分）
2. 向量检索（BGE嵌入 + Chroma向量库）
3. BM25关键词检索（jieba分词 + rank_bm25）
4. RRF融合（Reciprocal Rank Fusion，合并两路检索结果）
5. Cross-Encoder重排序（可选，提升Top-K精度）
6. 按文档去重（同一文档保留Top-N个最相关chunk）
7. 相似度阈值过滤
8. 返回结构化的SearchResult列表

检索策略:
- 纯向量模式(hybrid=False): 查询预处理 → 向量检索 → 阈值过滤 → 去重
- 混合检索模式(hybrid=True): 查询预处理 → 向量+BM25 → RRF融合 → 重排序 → 去重
"""
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from indexer.embedder import Embedder
from indexer.vector_store import VectorStore
from indexer.bm25_search import BM25Search
from indexer.reranker import Reranker


@dataclass
class SearchResult:
    """检索结果条目"""
    # 文本内容
    text: str
    # 来源URL
    source_url: str
    # 来源站点
    source_site: str
    # 文档标题
    title: str
    # 内容类型(html/pdf等)
    content_type: str
    # 发布日期
    publish_date: str = ""
    # 相似度分数(0~1, 越高越相似)
    score: float = 0.0
    # 文档ID
    doc_id: str = ""
    # 分块序号
    chunk_index: int = 0
    # 总分块数
    total_chunks: int = 1
    # 分数来源标记（vector/bm25/rrf/rerank）
    score_source: str = "vector"

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "source_url": self.source_url,
            "source_site": self.source_site,
            "title": self.title,
            "content_type": self.content_type,
            "publish_date": self.publish_date,
            "score": round(self.score, 4),
            "doc_id": self.doc_id,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "score_source": self.score_source,
        }


class Retriever:
    """检索器 - Hybrid Search + RRF融合 + 重排序"""

    def __init__(self, embedder: Optional[Embedder] = None,
                 vector_store: Optional[VectorStore] = None,
                 bm25_search: Optional[BM25Search] = None,
                 reranker: Optional[Reranker] = None,
                 default_top_k: int = 5,
                 score_threshold: float = 0.3,
                 deduplicate: bool = True,
                 max_chunks_per_doc: int = 2,
                 hybrid: bool = True,
                 rrf_k: int = 60,
                 use_reranker: bool = True,
                 use_query_rewrite: bool = True,
                 use_query_decompose: bool = True):
        """
        Args:
            embedder: 嵌入器实例
            vector_store: 向量库实例
            bm25_search: BM25检索实例
            reranker: 重排序器实例
            default_top_k: 默认返回结果数
            score_threshold: 相似度阈值(低于此值的结果被过滤)
            deduplicate: 是否按文档去重
            max_chunks_per_doc: 每文档最多保留的chunk数
            hybrid: 是否启用混合检索（向量+BM25）
            rrf_k: RRF融合常数(默认60，越小排名靠前的权重越大)
            use_reranker: 是否启用重排序
            use_query_rewrite: 是否启用查询改写（口语→书面，补充上下文）
            use_query_decompose: 是否启用多查询分解（复杂问题拆分）
        """
        self.embedder = embedder or Embedder()
        self.vector_store = vector_store or VectorStore()
        self.bm25_search = bm25_search or BM25Search()
        self.reranker = reranker or Reranker()
        self.default_top_k = default_top_k
        self.score_threshold = score_threshold
        self.deduplicate = deduplicate
        self.max_chunks_per_doc = max_chunks_per_doc
        self.hybrid = hybrid
        self.rrf_k = rrf_k
        self.use_reranker = use_reranker
        self.use_query_rewrite = use_query_rewrite
        self.use_query_decompose = use_query_decompose

        # BM25索引是否已构建
        self._bm25_initialized = False

    def _preprocess_query(self, query: str) -> list[str]:
        """查询预处理：改写 + 多查询分解

        流程:
        1. 查询改写：口语→书面语，补充上下文（如"怎么选课"→"大学选课流程和选课系统操作方法"）
        2. 多查询分解：复杂查询拆分为多个子查询分别检索
        3. 返回最终查询列表（至少包含原始查询作为保底）

        Args:
            query: 用户原始查询

        Returns:
            查询列表（可能包含原始查询、改写查询、分解子查询）
        """
        queries = []
        rewritten_query = query  # 默认使用原始查询

        # Step 1: 查询改写
        if self.use_query_rewrite:
            rewritten = self._rewrite_query(query)
            if rewritten and rewritten != query:
                rewritten_query = rewritten
                queries.append(rewritten_query)
                logger.info(f"查询改写: '{query[:30]}' → '{rewritten_query[:30]}'")

        # Step 2: 多查询分解
        if self.use_query_decompose:
            sub_queries = self._decompose_query(rewritten_query)
            if len(sub_queries) > 1:
                queries.extend(sub_queries)
                logger.info(
                    f"查询分解: '{rewritten_query[:30]}' → {len(sub_queries)} 个子查询"
                )
            elif not queries:
                # 分解只有1条且未改写，直接用原始查询
                queries = [query]
        else:
            if not queries:
                queries = [query]

        # 保底：确保至少包含原始查询
        if query not in queries:
            queries.append(query)

        # 限制子查询数量（避免过多检索开销）
        if len(queries) > 4:
            queries = queries[:4]

        return queries

    def _rewrite_query(self, query: str) -> Optional[str]:
        """查询改写：调用LLM将口语化查询转为书面检索式

        Args:
            query: 用户原始查询

        Returns:
            改写后的查询，失败时返回None
        """
        try:
            from rag.prompts import QUERY_REWRITE_PROMPT
            from langchain_core.messages import HumanMessage
            from rag.generator import Generator

            # 复用Generator的LLM实例
            gen = Generator()
            llm = gen._get_llm()

            prompt = QUERY_REWRITE_PROMPT.format(query=query)
            response = llm.invoke([HumanMessage(content=prompt)])
            # LangChain content 类型为 str | list[dict]，实际总是 str
            content = response.content
            rewritten = content.strip() if isinstance(content, str) else str(content)

            # 防止LLM输出异常（太长或包含多余内容）
            if len(rewritten) > len(query) * 5 or len(rewritten) < 2:
                logger.warning(f"查询改写结果异常，忽略: '{rewritten[:50]}'")
                return None

            return rewritten

        except Exception as e:
            logger.warning(f"查询改写失败，使用原始查询: {e}")
            return None

    def _decompose_query(self, query: str) -> list[str]:
        """多查询分解：将复杂查询拆分为多个子查询

        Args:
            query: 查询文本（可能是改写后的）

        Returns:
            子查询列表，分解失败时返回 [query]
        """
        try:
            from rag.prompts import QUERY_DECOMPOSE_PROMPT
            from langchain_core.messages import HumanMessage
            from rag.generator import Generator

            gen = Generator()
            llm = gen._get_llm()

            prompt = QUERY_DECOMPOSE_PROMPT.format(query=query)
            response = llm.invoke([HumanMessage(content=prompt)])
            # LangChain content 类型为 str | list[dict]，实际总是 str
            raw_content = response.content
            content = raw_content.strip() if isinstance(raw_content, str) else str(raw_content)

            # 解析LLM输出：每行一个子查询
            sub_queries = [
                line.strip()
                for line in content.split("\n")
                if line.strip() and len(line.strip()) >= 2
            ]

            # 过滤异常输出
            sub_queries = [
                q for q in sub_queries
                if len(q) <= len(query) * 5
            ]

            if not sub_queries:
                return [query]

            return sub_queries

        except Exception as e:
            logger.warning(f"查询分解失败，使用原始查询: {e}")
            return [query]

    def _ensure_bm25_index(self):
        """确保BM25索引已构建"""
        if self._bm25_initialized:
            return

        # 先尝试从磁盘加载
        self.bm25_search._load_index()
        if self.bm25_search._loaded:
            self._bm25_initialized = True
            return

        # 从Chroma构建
        self.bm25_search.build_index(vector_store=self.vector_store)
        self._bm25_initialized = self.bm25_search._loaded

    def retrieve(self, query: str, top_k: Optional[int] = None,
                 filter_site: Optional[str] = None,
                 filter_type: Optional[str] = None) -> list[SearchResult]:
        """执行检索

        Args:
            query: 用户查询文本
            top_k: 返回结果数
            filter_site: 按来源站点过滤（如"教务部"）
            filter_type: 按内容类型过滤（如"pdf"）

        Returns:
            检索结果列表
        """
        top_k = top_k or self.default_top_k

        # 构建过滤条件
        filter_dict = None
        if filter_site or filter_type:
            filter_dict = {}
            if filter_site:
                filter_dict["source_site"] = filter_site
            if filter_type:
                filter_dict["content_type"] = filter_type

        # ── 查询预处理：改写 + 分解 ──
        queries = self._preprocess_query(query)

        # ── 多查询检索 + 结果合并 ──
        all_results: list[SearchResult] = []
        seen_chunk_ids: set[str] = set()

        for q in queries:
            if self.hybrid:
                partial = self._hybrid_retrieve(q, top_k, filter_dict)
            else:
                partial = self._vector_retrieve(q, top_k, filter_dict)

            # 按 chunk_id 去重合并（保留最高分）
            for r in partial:
                # 用 doc_id+chunk_index 构造唯一键
                key = f"{r.doc_id}#{r.chunk_index}"
                if key not in seen_chunk_ids:
                    seen_chunk_ids.add(key)
                    all_results.append(r)

        # 按score降序排列（多查询结果混合后需要重新排序）
        all_results.sort(key=lambda x: -x.score)

        # 按文档去重：同一文档保留 max_chunks_per_doc 个最相关的chunk
        if self.deduplicate and all_results:
            all_results = self._deduplicate(all_results)

        # 截取top_k
        all_results = all_results[:top_k]

        logger.info(
            f"检索完成: query='{query[:30]}', 子查询数={len(queries)}, "
            f"模式={'hybrid' if self.hybrid else 'vector'}, "
            f"结果数={len(all_results)}"
        )
        return all_results

    def _vector_retrieve(self, query: str, top_k: int,
                         filter_dict: Optional[dict]) -> list[SearchResult]:
        """纯向量检索（原有逻辑）"""
        # BGE推荐对查询添加指令前缀
        query_text = f"为这个句子生成表示以用于检索相关文章：{query}"
        query_embedding = self.embedder.embed_query(query_text)

        # 去重时多取候选，确保去重后数量充足
        search_k = top_k * 3 if self.deduplicate else top_k
        raw_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=search_k,
            filter_dict=filter_dict,
        )

        if not raw_results:
            logger.info(f"向量检索无结果: query='{query[:30]}'")
            return []

        results = []
        for item in raw_results:
            meta = item.get("metadata", {})
            distance = item.get("distance", 1.0)
            score = 1.0 - distance

            if score < self.score_threshold:
                continue

            result = SearchResult(
                text=item.get("text", ""),
                source_url=meta.get("source_url", ""),
                source_site=meta.get("source_site", ""),
                title=meta.get("title", ""),
                content_type=meta.get("content_type", ""),
                publish_date=meta.get("publish_date", ""),
                score=score,
                doc_id=meta.get("doc_id", ""),
                chunk_index=int(meta.get("chunk_index", 0)),
                total_chunks=int(meta.get("total_chunks", 1)),
                score_source="vector",
            )
            results.append(result)

        return results

    def _hybrid_retrieve(self, query: str, top_k: int,
                         filter_dict: Optional[dict]) -> list[SearchResult]:
        """混合检索：向量 + BM25 → RRF融合 → 重排序"""
        # ── Step 1: 向量检索 ──
        query_text = f"为这个句子生成表示以用于检索相关文章：{query}"
        query_embedding = self.embedder.embed_query(query_text)

        # 融合+去重+重排序需要更多候选
        candidate_k = top_k * 3
        vector_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=candidate_k,
            filter_dict=filter_dict,
        )

        # ── Step 2: BM25关键词检索 ──
        self._ensure_bm25_index()
        bm25_results = self.bm25_search.search(
            query=query,
            top_k=candidate_k,
            filter_dict=filter_dict,
        )

        if not vector_results and not bm25_results:
            logger.info(f"混合检索两路均无结果: query='{query[:30]}'")
            return []

        # ── Step 3: RRF融合 ──
        fused = self._rrf_fuse(vector_results, bm25_results)

        # ── Step 4: 阈值过滤 ──
        filtered = []
        for item in fused:
            if item.get("rrf_score", 0) < self.score_threshold * 0.1:
                # RRF分数范围较小，阈值需相应调整
                continue
            filtered.append(item)

        if not filtered:
            # 阈值过滤过严时，保留原始融合结果
            filtered = fused

        # ── Step 5: Cross-Encoder重排序 ──
        if self.use_reranker and filtered:
            # 构造重排序候选
            rerank_candidates = [
                {"text": item.get("text", ""), **item}
                for item in filtered
            ]
            reranked = self.reranker.rerank(
                query=query,
                candidates=rerank_candidates,
                top_k=candidate_k,
            )

            # 转换为SearchResult
            results = []
            for item in reranked:
                meta = item.get("metadata", {})
                score = item.get("rerank_score", item.get("rrf_score", 0))
                results.append(SearchResult(
                    text=item.get("text", ""),
                    source_url=meta.get("source_url", ""),
                    source_site=meta.get("source_site", ""),
                    title=meta.get("title", ""),
                    content_type=meta.get("content_type", ""),
                    publish_date=meta.get("publish_date", ""),
                    score=score,
                    doc_id=meta.get("doc_id", ""),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    total_chunks=int(meta.get("total_chunks", 1)),
                    score_source="rerank",
                ))
        else:
            # 无重排序时，直接用RRF分数
            results = []
            for item in filtered:
                meta = item.get("metadata", {})
                results.append(SearchResult(
                    text=item.get("text", ""),
                    source_url=meta.get("source_url", ""),
                    source_site=meta.get("source_site", ""),
                    title=meta.get("title", ""),
                    content_type=meta.get("content_type", ""),
                    publish_date=meta.get("publish_date", ""),
                    score=item.get("rrf_score", 0),
                    doc_id=meta.get("doc_id", ""),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    total_chunks=int(meta.get("total_chunks", 1)),
                    score_source="rrf",
                ))

        return results

    def _rrf_fuse(self, vector_results: list[dict],
                  bm25_results: list[dict]) -> list[dict]:
        """Reciprocal Rank Fusion (RRF) - 合合向量检索和BM25检索结果

        RRF公式: score = 1/(k + rank_vector) + 1/(k + rank_bm25)

        Args:
            vector_results: 向量检索结果（Chroma格式）
            bm25_results: BM25检索结果（BM25Search格式）

        Returns:
            融合后的结果列表，按rrf_score降序排列
        """
        # 按chunk_id索引，合并两路检索的排名
        rrf_scores = {}  # chunk_id → rrf_score
        item_data = {}   # chunk_id → (text, metadata, vector_score, bm25_score)

        # 向量检索排名
        for rank, item in enumerate(vector_results, 1):
            chunk_id = item.get("chunk_id", "")
            meta = item.get("metadata", {})
            distance = item.get("distance", 1.0)
            vector_score = 1.0 - distance

            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (self.rrf_k + rank)
            item_data[chunk_id] = {
                "chunk_id": chunk_id,
                "text": item.get("text", ""),
                "metadata": meta,
                "vector_score": vector_score,
                "bm25_score": 0.0,
            }

        # BM25检索排名
        for rank, item in enumerate(bm25_results, 1):
            chunk_id = item.get("chunk_id", "")
            bm25_score = item.get("bm25_score", 0)

            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (self.rrf_k + rank)

            if chunk_id in item_data:
                item_data[chunk_id]["bm25_score"] = bm25_score
            else:
                meta = item.get("metadata", {})
                item_data[chunk_id] = {
                    "chunk_id": chunk_id,
                    "text": item.get("text", ""),
                    "metadata": meta,
                    "vector_score": 0.0,
                    "bm25_score": bm25_score,
                }

        # 按RRF分数降序排列
        fused = []
        for chunk_id, rrf_score in rrf_scores.items():
            data = item_data[chunk_id]
            data["rrf_score"] = rrf_score
            fused.append(data)

        fused.sort(key=lambda x: -x["rrf_score"])

        logger.info(
            f"RRF融合: 向量={len(vector_results)}, BM25={len(bm25_results)}, "
            f"融合后={len(fused)}"
        )

        return fused

    def _deduplicate(self, results: list[SearchResult]) -> list[SearchResult]:
        """按文档去重，同一doc_id保留 max_chunks_per_doc 个最相关chunk

        比原方案改进：不再只保留1个chunk，而是保留N个最相关的，
        长文档中不同段落可能回答不同方面
        """
        doc_chunks: dict[str, list[SearchResult]] = {}

        for r in results:
            doc_id = r.doc_id
            if doc_id not in doc_chunks:
                doc_chunks[doc_id] = []
            doc_chunks[doc_id].append(r)

        # 每个文档保留 max_chunks_per_doc 个最高分的chunk
        deduped = []
        for doc_id, chunks in doc_chunks.items():
            # 按score降序
            chunks.sort(key=lambda x: -x.score)
            deduped.extend(chunks[:self.max_chunks_per_doc])

        # 整体按score降序排列
        deduped.sort(key=lambda x: -x.score)

        return deduped