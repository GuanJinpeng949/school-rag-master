"""BM25关键词检索 - 中文分词 + BM25评分

核心功能:
1. 从Chroma加载全部文档文本，构建BM25索引
2. 使用jieba中文分词，支持精确关键词匹配
3. 返回BM25评分排序的检索结果
4. 累加式索引更新（新增文档时追加，而非重建）

设计说明:
- BM25擅长精确关键词匹配（学号、文号、专有名词等），弥补向量检索的不足
- 与向量检索互补：向量捕捉语义相似，BM25捕捉词汇重叠
- 使用jieba分词处理中文文本，提升召回率
"""
import pickle
from pathlib import Path
from typing import Optional

import jieba
from loguru import logger
from rank_bm25 import BM25Okapi

from config.settings import settings, DATA_DIR


# BM25索引持久化路径
BM25_INDEX_DIR = DATA_DIR / "bm25"


def _tokenize_zh(text: str) -> list[str]:
    """中文分词：jieba分词 + 过滤停用词/空串"""
    words = jieba.lcut_for_search(text)
    # 过滤空串、纯数字单字符、纯标点
    return [w for w in words if w.strip() and len(w) > 1 or (w.isdigit() and len(w) >= 2)]


class BM25Search:
    """BM25关键词检索引擎"""

    def __init__(self, persist_dir: Optional[str] = None):
        """
        Args:
            persist_dir: BM25索引持久化目录
        """
        self.persist_dir = Path(persist_dir or str(BM25_INDEX_DIR))
        self._bm25: Optional[BM25Okapi] = None
        self._corpus: list[str] = []       # 原始文本列表
        self._tokenized: list[list[str]] = []  # 分词后的文本列表
        self._chunk_ids: list[str] = []    # 对应的chunk_id列表
        self._metadatas: list[dict] = []   # 对应的metadata列表
        self._loaded = False

    def build_index(self, vector_store=None):
        """从Chroma向量库加载全部文档，构建BM25索引

        Args:
            vector_store: VectorStore实例（如不提供则自行创建）
        """
        from indexer.vector_store import VectorStore

        if vector_store is None:
            vector_store = VectorStore()

        logger.info("开始构建BM25索引...")

        collection = vector_store._get_collection()
        count = collection.count()
        if count == 0:
            logger.warning("Chroma中无数据，BM25索引为空")
            return

        # 加载全部文档
        all_data = collection.get(include=["documents", "metadatas"])
        if not all_data or not all_data["ids"]:
            logger.warning("Chroma数据加载失败")
            return

        self._chunk_ids = all_data["ids"]
        self._corpus = all_data["documents"] or []
        self._metadatas = all_data["metadatas"] or []

        # 中文分词
        logger.info(f"分词处理: {len(self._corpus)} 条文档...")
        self._tokenized = [_tokenize_zh(text) for text in self._corpus]

        # 构建BM25索引
        self._bm25 = BM25Okapi(self._tokenized)
        self._loaded = True

        logger.info(f"BM25索引构建完成: {len(self._corpus)} 条文档")

        # 持久化
        self._save_index()

    def add_documents(self, chunk_ids: list[str], texts: list[str],
                      metadatas: list[dict]):
        """追加新文档到BM25索引（增量更新）

        Args:
            chunk_ids: 分块ID列表
            texts: 分块文本列表
            metadatas: 元数据列表
        """
        if not texts:
            return

        # 避免重复添加（按chunk_id去重）
        existing_ids = set(self._chunk_ids)
        new_ids, new_texts, new_metas = [], [], []
        for i, cid in enumerate(chunk_ids):
            if cid not in existing_ids:
                new_ids.append(cid)
                new_texts.append(texts[i])
                new_metas.append(metadatas[i])

        if not new_ids:
            logger.debug("BM25: 无新文档需要追加")
            return

        # 分词
        new_tokenized = [_tokenize_zh(t) for t in new_texts]

        # 合并到现有索引
        self._chunk_ids.extend(new_ids)
        self._corpus.extend(new_texts)
        self._metadatas.extend(new_metas)
        self._tokenized.extend(new_tokenized)

        # 重建BM25（BM25Okapi不支持增量添加，需重建）
        self._bm25 = BM25Okapi(self._tokenized)
        self._loaded = True

        logger.info(f"BM25索引追加: {len(new_ids)} 条新文档, 总计 {len(self._corpus)} 条")
        self._save_index()

    def search(self, query: str, top_k: int = 10,
               filter_dict: Optional[dict] = None) -> list[dict]:
        """BM25关键词检索

        Args:
            query: 用户查询文本
            top_k: 返回最相似的K个结果
            filter_dict: 元数据过滤条件（如 {"source_site": "教务部"}）

        Returns:
            检索结果列表，每项包含 chunk_id, text, metadata, bm25_score
        """
        if not self._loaded:
            self._load_index()
            if not self._loaded:
                logger.warning("BM25索引未构建，尝试从Chroma构建...")
                self.build_index()
                if not self._loaded:
                    return []

        # 查询分词
        query_tokens = _tokenize_zh(query)

        if not query_tokens:
            return []

        # BM25评分
        scores = self._bm25.get_scores(query_tokens)

        # 排序 + 过滤
        scored_results = []
        for i, score in enumerate(scores):
            # 元数据过滤
            if filter_dict:
                meta = self._metadatas[i] if i < len(self._metadatas) else {}
                match = all(
                    str(meta.get(k, "")) == str(v)
                    for k, v in filter_dict.items()
                )
                if not match:
                    continue

            # BM25分数可能为负数（罕见词），过滤掉
            if score <= 0:
                continue

            scored_results.append({
                "chunk_id": self._chunk_ids[i],
                "text": self._corpus[i],
                "metadata": self._metadatas[i] if i < len(self._metadatas) else {},
                "bm25_score": score,
            })

        # 按BM25分数降序排列
        scored_results.sort(key=lambda x: -x["bm25_score"])

        # 归一化BM25分数到0~1范围（用于后续融合）
        if scored_results:
            max_score = scored_results[0]["bm25_score"]
            if max_score > 0:
                for r in scored_results:
                    r["bm25_score_normalized"] = r["bm25_score"] / max_score
            else:
                for r in scored_results:
                    r["bm25_score_normalized"] = 0.0

        return scored_results[:top_k]

    def _save_index(self):
        """持久化BM25索引到磁盘"""
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "corpus": self._corpus,
            "tokenized": self._tokenized,
            "chunk_ids": self._chunk_ids,
            "metadatas": self._metadatas,
        }

        index_path = self.persist_dir / "bm25_index.pkl"
        with open(index_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(f"BM25索引已持久化: {index_path}")

    def _load_index(self):
        """从磁盘加载BM25索引"""
        index_path = self.persist_dir / "bm25_index.pkl"

        if not index_path.exists():
            logger.info("BM25索引文件不存在，需从Chroma构建")
            return

        try:
            with open(index_path, "rb") as f:
                data = pickle.load(f)

            self._corpus = data["corpus"]
            self._tokenized = data["tokenized"]
            self._chunk_ids = data["chunk_ids"]
            self._metadatas = data["metadatas"]

            # 重建BM25对象
            self._bm25 = BM25Okapi(self._tokenized)
            self._loaded = True

            logger.info(f"BM25索引已加载: {len(self._corpus)} 条文档")
        except Exception as e:
            logger.error(f"BM25索引加载失败: {e}")
            self._loaded = False