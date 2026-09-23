"""SQLite元数据存储 - 文档元数据的结构化存储

核心功能:
1. 文档元数据CRUD
2. 按站点/类型/日期过滤
3. 索引状态跟踪（已索引/未索引）
4. 增量索引支持

设计说明:
- SQLite存储结构化元数据，便于过滤和审计
- 与Chroma向量库互补：Chroma负责向量检索，SQLite负责精确查询
- 支持增量索引：只处理新增/变更的文档
"""
import json
import sqlite3
import threading
from pathlib import Path
from typing import Optional

from loguru import logger

from config.settings import settings, DATA_DIR
from indexer.chunker import Chunk


class MetadataStore:
    """SQLite元数据存储"""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or settings.metadata_db
        # 每个线程各持一个连接：sqlite3 默认禁止连接跨线程使用
        # （check_same_thread=True），而 FastAPI 的同步依赖运行在线程池线程中，
        # 连接会在 A 线程创建、B 线程使用，从而抛 RuntimeError。
        self._local = threading.local()
        self._init_db()

    # 不太理解构造
    def _get_conn(self) -> sqlite3.Connection:
        """获取当前线程的数据库连接"""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn = conn
        return conn

    def _init_db(self):
        """初始化数据库表"""
        conn = self._get_conn()

        # 建立文档与多个分块的关系
        # documents
        #   一篇文档
        #       ↓ doc_id
        # chunks
        #   chunk 0
        #   chunk 1
        #   chunk 2
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                source_url TEXT NOT NULL DEFAULT '',
                source_site TEXT NOT NULL DEFAULT '',
                title TEXT NOT NULL DEFAULT '',
                content_type TEXT NOT NULL DEFAULT '',
                publish_date TEXT NOT NULL DEFAULT '',
                author TEXT NOT NULL DEFAULT '',
                file_path TEXT NOT NULL DEFAULT '',
                file_hash TEXT NOT NULL DEFAULT '',
                text_length INTEGER NOT NULL DEFAULT 0,
                chunk_count INTEGER NOT NULL DEFAULT 0,
                attachments_json TEXT NOT NULL DEFAULT '[]',
                extra_json TEXT NOT NULL DEFAULT '{}',
                indexed_at TEXT NOT NULL DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_documents_source_site
                ON documents(source_site);
            CREATE INDEX IF NOT EXISTS idx_documents_content_type
                ON documents(content_type);
            CREATE INDEX IF NOT EXISTS idx_documents_publish_date
                ON documents(publish_date);
            CREATE INDEX IF NOT EXISTS idx_documents_file_hash
                ON documents(file_hash);

            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL DEFAULT 0,
                text_length INTEGER NOT NULL DEFAULT 0,
                indexed_at TEXT NOT NULL DEFAULT '',
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_doc_id
                ON chunks(doc_id);
        """)

        conn.commit()

    def upsert_document(self, doc_id: str, source_url: str, source_site: str,
                        title: str, content_type: str, publish_date: str = "",
                        author: str = "", file_path: str = "", file_hash: str = "",
                        text_length: int = 0, chunk_count: int = 0,
                        attachments: list = None, extra: dict = None,
                        indexed_at: str = ""):
        """插入或更新文档元数据"""
        conn = self._get_conn()
        attachments_json = json.dumps(attachments or [], ensure_ascii=False)
        extra_json = json.dumps(extra or {}, ensure_ascii=False)

        conn.execute("""
            INSERT INTO documents (doc_id, source_url, source_site, title, content_type,
                publish_date, author, file_path, file_hash, text_length, chunk_count,
                attachments_json, extra_json, indexed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(doc_id) DO UPDATE SET
                source_url=excluded.source_url,
                source_site=excluded.source_site,
                title=excluded.title,
                content_type=excluded.content_type,
                publish_date=excluded.publish_date,
                author=excluded.author,
                file_path=excluded.file_path,
                file_hash=excluded.file_hash,
                text_length=excluded.text_length,
                chunk_count=excluded.chunk_count,
                attachments_json=excluded.attachments_json,
                extra_json=excluded.extra_json,
                indexed_at=excluded.indexed_at
        """, (doc_id, source_url, source_site, title, content_type,
              publish_date, author, file_path, file_hash, text_length,
              chunk_count, attachments_json, extra_json, indexed_at))
        conn.commit()

    def upsert_chunk(self, chunk: Chunk, indexed_at: str = ""):
        """插入或更新分块记录"""
        conn = self._get_conn()
        conn.execute("""
            INSERT INTO chunks (chunk_id, doc_id, chunk_index, text_length, indexed_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(chunk_id) DO UPDATE SET
                doc_id=excluded.doc_id,
                chunk_index=excluded.chunk_index,
                text_length=excluded.text_length,
                indexed_at=excluded.indexed_at
        """, (chunk.chunk_id, chunk.doc_id, chunk.chunk_index, len(chunk.text), indexed_at))
        conn.commit()

    def upsert_chunks_batch(self, chunks: list[Chunk], indexed_at: str = ""):
        """批量插入分块记录"""
        conn = self._get_conn()
        data = [(c.chunk_id, c.doc_id, c.chunk_index, len(c.text), indexed_at) for c in chunks]
        conn.executemany("""
            INSERT INTO chunks (chunk_id, doc_id, chunk_index, text_length, indexed_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(chunk_id) DO UPDATE SET
                doc_id=excluded.doc_id,
                chunk_index=excluded.chunk_index,
                text_length=excluded.text_length,
                indexed_at=excluded.indexed_at
        """, data)
        conn.commit()

    def get_document(self, doc_id: str) -> Optional[dict]:
        """获取文档元数据"""
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        if row:
            # 转换为字典，方便修改和返回
            d = dict(row)
            # 删除(pop) JSON 字段串，替换为 Python 对象
            d["attachments"] = json.loads(d.pop("attachments_json", "[]"))
            d["extra"] = json.loads(d.pop("extra_json", "{}"))
            return d
        return None

    def get_unindexed_doc_ids(self) -> list[str]:
        """获取尚未索引的文档ID"""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT doc_id FROM documents WHERE indexed_at = '' OR indexed_at IS NULL"
        ).fetchall()
        return [row["doc_id"] for row in rows]

    def get_document_by_hash(self, file_hash: str) -> Optional[dict]:
        """根据文件哈希查询文档"""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM documents WHERE file_hash = ?", (file_hash,)
        ).fetchone()
        if row:
            return dict(row)
        return None

    def delete_document(self, doc_id: str) -> int:
        """删除文档及其分块"""
        conn = self._get_conn()
        # 先删所有分块记录，再删文档记录，避免留下孤立的 chunk 元数据
        conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        cursor = conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        conn.commit()
        # 返回文档删除是否成功
        return cursor.rowcount

    def get_stats(self) -> dict:
        """获取统计信息"""
        conn = self._get_conn()

        doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        total_text_length = conn.execute(
            "SELECT COALESCE(SUM(text_length), 0) FROM documents"
        ).fetchone()[0]
        indexed_count = conn.execute(
            "SELECT COUNT(*) FROM documents WHERE indexed_at != '' AND indexed_at IS NOT NULL"
        ).fetchone()[0]

        # 按站点统计
        site_rows = conn.execute(
            "SELECT source_site, COUNT(*) as cnt FROM documents GROUP BY source_site ORDER BY cnt DESC"
        ).fetchall()
        by_site = {row["source_site"]: row["cnt"] for row in site_rows}

        # 按类型统计
        type_rows = conn.execute(
            "SELECT content_type, COUNT(*) as cnt FROM documents GROUP BY content_type"
        ).fetchall()
        by_type = {row["content_type"]: row["cnt"] for row in type_rows}

        return {
            "total_documents": doc_count,
            "total_chunks": chunk_count,
            "total_text_length": total_text_length,
            "indexed_documents": indexed_count,
            "by_site": by_site,
            "by_type": by_type,
        }

    def close(self):
        """关闭当前线程的数据库连接"""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None


    """
        parsed_docs_current.json
            ↓
        TextChunker 分块
            ↓
        Embedder 生成向量
            ↓
        VectorStore.add_chunks()
            写入 Chroma：文本 + 向量 + 检索 metadata
            ↓
        MetadataStore.upsert_document()
            写入 SQLite：每篇文档元数据
            ↓
        MetadataStore.upsert_chunks_batch()
            写入 SQLite：每个 chunk 的归属与状态
    """
