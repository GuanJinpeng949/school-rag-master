"""Chroma向量库操作 - 向量存储与检索

核心功能:
1. Chroma集合管理（初始化、持久化）
2. 批量写入向量+元数据
3. 相似度检索（支持元数据过滤）
4. 删除/更新文档

设计说明:
- Chroma零配置启动，适合开发阶段
- 持久化到 data/chroma/ 目录
- 元数据与向量存储在一起，支持过滤检索
- 向量维度与嵌入模型匹配（BGE: 1024, OpenAI: 1536）
"""
from pathlib import Path
from typing import Optional

from loguru import logger

from config.settings import settings, CHROMA_DIR
from indexer.chunker import Chunk


# Chroma元数据字段（不支持list/dict类型，需要序列化为字符串）
CHROMA_META_KEYS = [
    "doc_id", "chunk_index", "total_chunks",
    "source_url", "source_site", "title", "content_type",
    "publish_date", "author", "file_path", "file_hash",
]


class VectorStore:
    """Chroma向量库"""

    COLLECTION_NAME = "school_rag"

    def __init__(self, persist_dir: Optional[str] = None,
                 collection_name: Optional[str] = None):
        """
        Args:
            persist_dir: 持久化目录
            collection_name: 集合名称
        """
        self.persist_dir = persist_dir or settings.vector_db.chroma_persist_dir
        self.collection_name = collection_name or self.COLLECTION_NAME
        self._client = None
        self._collection = None

    def _get_client(self):
        """获取Chroma客户端"""
        if self._client is not None:
            return self._client

        import chromadb
        self._client = chromadb.PersistentClient(path=self.persist_dir)
        return self._client

    def _get_collection(self):
        """获取或创建集合"""
        if self._collection is not None:
            return self._collection

        client = self._get_client()
        self._collection = client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},  # 使用余弦相似度
        )
        return self._collection

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]],
                   batch_size: int = 500) -> int:
        """批量写入分块和向量

        Args:
            chunks: 分块列表
            embeddings: 对应的向量列表
            batch_size: 每批写入数量

        Returns:
            写入的分块数量
        """
        if not chunks or len(chunks) != len(embeddings):
            logger.error(f"分块数({len(chunks)})与向量数({len(embeddings)})不匹配")
            return 0

        collection = self._get_collection()
        added = 0

        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]

            ids = [c.chunk_id for c in batch_chunks]
            documents = [c.text for c in batch_chunks]
            metadatas = [self._chunk_to_meta(c) for c in batch_chunks]

            try:
                collection.upsert(
                    ids=ids,
                    embeddings=batch_embeddings,
                    documents=documents,
                    metadatas=metadatas,
                )
                added += len(batch_chunks)
            except Exception as e:
                logger.error(f"写入Chroma失败(批次{i//batch_size}): {e}")
                # 逐条写入失败的批次
                for j, chunk in enumerate(batch_chunks):
                    try:
                        collection.upsert(
                            ids=[chunk.chunk_id],
                            embeddings=[batch_embeddings[j]],
                            documents=[chunk.text],
                            metadatas=[self._chunk_to_meta(chunk)],
                        )
                        added += 1
                    except Exception as e2:
                        logger.warning(f"写入失败: chunk_id={chunk.chunk_id} - {e2}")

        logger.info(f"写入Chroma: {added}/{len(chunks)} 个分块")
        return added

    def search(self, query_embedding: list[float], top_k: int = 10,
               filter_dict: Optional[dict] = None) -> list[dict]:
        """向量相似度检索

        Args:
            query_embedding: 查询向量
            top_k: 返回最相似的K个结果
            filter_dict: 元数据过滤条件，如 {"source_site": "教务部"}

        Returns:
            检索结果列表，每项包含 chunk_id, text, metadata, distance
        """
        collection = self._get_collection()

        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if filter_dict:
            kwargs["where"] = filter_dict

        try:
            results = collection.query(**kwargs)
        except Exception as e:
            logger.error(f"Chroma检索失败: {e}")
            return []

        # 解析结果
        if not results or not results["ids"] or not results["ids"][0]:
            return []

        output = []
        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for i in range(len(ids)):
            item = {
                "chunk_id": ids[i],
                "text": documents[i] if documents else "",
                "metadata": metadatas[i] if metadatas else {},
                "distance": distances[i] if distances else 0.0,
            }
            output.append(item)

        return output

    def delete_by_doc_id(self, doc_id: str) -> int:
        """删除指定文档ID的所有分块"""
        collection = self._get_collection()

        try:
            # 先查询该doc_id的所有chunk
            results = collection.get(
                where={"doc_id": doc_id},
                include=[],
            )
            if results and results["ids"]:
                collection.delete(ids=results["ids"])
                return len(results["ids"])
        except Exception as e:
            logger.error(f"删除文档分块失败: doc_id={doc_id} - {e}")

        return 0

    def delete_by_source_site(self, source_site: str) -> int:
        """删除指定站点的所有分块"""
        collection = self._get_collection()

        try:
            results = collection.get(
                where={"source_site": source_site},
                include=[],
            )
            if results and results["ids"]:
                collection.delete(ids=results["ids"])
                return len(results["ids"])
        except Exception as e:
            logger.error(f"删除站点分块失败: site={source_site} - {e}")

        return 0

    def get_stats(self) -> dict:
        """获取向量库统计信息"""
        collection = self._get_collection()
        count = collection.count()

        # 获取站点分布
        site_counts = {}
        if count > 0:
            try:
                all_meta = collection.get(include=["metadatas"])
                if all_meta and all_meta["metadatas"]:
                    for meta in all_meta["metadatas"]:
                        site = meta.get("source_site", "未知")
                        site_counts[site] = site_counts.get(site, 0) + 1
            except Exception as e:
                logger.warning(f"获取站点分布失败: {e}")

        return {
            "total_chunks": count,
            "collection_name": self.collection_name,
            "persist_dir": self.persist_dir,
            "site_distribution": site_counts,
        }

    def _chunk_to_meta(self, chunk: Chunk) -> dict:
        """将Chunk元数据转为Chroma兼容格式

        Chroma元数据只支持 str/int/float/bool 类型
        """
        meta = {}
        for key in CHROMA_META_KEYS:
            val = getattr(chunk, key, "")
            if val is None:
                val = ""
            meta[key] = str(val) if not isinstance(val, (int, float, bool)) else val

        # attachments 是 list，序列化为 JSON 字符串
        if chunk.attachments:
            import json
            meta["attachments_json"] = json.dumps(chunk.attachments, ensure_ascii=False)

        # extra 是 dict，序列化为 JSON 字符串
        if chunk.extra:
            import json
            meta["extra_json"] = json.dumps(chunk.extra, ensure_ascii=False)

        return meta
