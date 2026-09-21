"""全量索引构建脚本 - 从解析结果构建向量索引

处理流程:
1. 加载 parsed_docs.json 中的文档
2. 文本分块(chunker)
3. 向量嵌入(embedder)
4. 写入Chroma(vector_store) + SQLite(metadata_store)
"""
import os
# 注意：TRANSFORMERS_OFFLINE / HF_HUB_OFFLINE / HF_ENDPOINT 都是第三方库在导入时读取的常量，
# 必须在导入 transformers / huggingface_hub 之前设置，导入后再改环境变量不会生效。
# 这里不再强制离线，否则本地没有模型时无法回退到镜像下载。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from config.settings import ensure_dirs
from indexer.chunker import TextChunker, Chunk
from indexer.embedder import Embedder
from indexer.vector_store import VectorStore
from indexer.metadata_store import MetadataStore
from parser.base import ParsedDocument, ContentType


def load_parsed_docs(json_path: str) -> list[ParsedDocument]:
    """从JSON加载解析结果"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    for item in data:
        doc = ParsedDocument(
            doc_id=item["doc_id"],
            text=item["text"],
            source_url=item.get("source_url", ""),
            source_site=item.get("source_site", ""),
            title=item.get("title", ""),
            content_type=ContentType(item.get("content_type", "other")),
            publish_date=item.get("publish_date", ""),
            author=item.get("author", ""),
            file_path=item.get("file_path", ""),
            file_hash=item.get("file_hash", ""),
            chunk_index=item.get("chunk_index", 0),
            total_chunks=item.get("total_chunks", 1),
            attachments=item.get("attachments", []),
            extra=item.get("extra", {}),
        )
        docs.append(doc)

    return docs


def build_index(docs: list[ParsedDocument], chunk_size: int = 1024,
                chunk_overlap: int = 120, doc_keep_size: int = 2048,
                embed_batch_size: int = 32,
                provider: str = "bge-local", rebuild: bool = False):
    """构建索引

    Args:
        docs: 解析后的文档列表
        chunk_size: 分块大小（仅用于长文档分块）
        chunk_overlap: 分块重叠
        doc_keep_size: 整文档保留阈值(<=此值不分块)
        embed_batch_size: 嵌入批量大小
        provider: 嵌入提供商
        rebuild: 是否重建（清空已有索引）
    """
    start_time = time.time()

    # 初始化组件
    chunker = TextChunker(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        doc_keep_size=doc_keep_size,
    )
    embedder = Embedder(provider=provider)
    vector_store = VectorStore()
    meta_store = MetadataStore()

    # 是否重建
    if rebuild:
        logger.info("重建模式：清空已有索引...")
        try:
            import chromadb
            client = chromadb.PersistentClient(path=vector_store.persist_dir)
            try:
                client.delete_collection(vector_store.collection_name)
                logger.info("已删除旧Chroma集合")
            except Exception:
                pass
            vector_store._collection = None
        except Exception as e:
            logger.warning(f"清空Chroma失败: {e}")

        # 清空SQLite
        conn = meta_store._get_conn()
        conn.execute("DELETE FROM chunks")
        conn.execute("DELETE FROM documents")
        conn.commit()
        logger.info("已清空SQLite元数据")

    # Step 1: 分块
    logger.info(f"开始分块: {len(docs)} 个文档...")
    chunks = chunker.chunk_documents(docs)
    if not chunks:
        logger.warning("没有可索引的内容")
        return

    logger.info(f"分块完成: {len(chunks)} 个分块")

    # Step 2: 嵌入
    logger.info(f"开始嵌入: provider={provider}, 维度={embedder.dimension}...")
    texts = [c.text for c in chunks]
    all_embeddings = []

    for i in range(0, len(texts), embed_batch_size):
        batch = texts[i:i + embed_batch_size]
        batch_embeddings = embedder.embed_texts(batch, batch_size=embed_batch_size)
        all_embeddings.extend(batch_embeddings)

        if (i // embed_batch_size + 1) % 10 == 0:
            logger.info(f"  嵌入进度: {min(i + embed_batch_size, len(texts))}/{len(texts)}")

    logger.info(f"嵌入完成: {len(all_embeddings)} 个向量")

    # Step 3: 写入Chroma
    logger.info("写入Chroma向量库...")
    added = vector_store.add_chunks(chunks, all_embeddings)
    logger.info(f"Chroma写入完成: {added} 个分块")

    # Step 4: 写入SQLite
    logger.info("写入SQLite元数据...")
    indexed_at = datetime.now().isoformat()

    # 写入文档元数据
    for doc in docs:
        doc_chunks = [c for c in chunks if c.doc_id == doc.doc_id]
        meta_store.upsert_document(
            doc_id=doc.doc_id,
            source_url=doc.source_url,
            source_site=doc.source_site,
            title=doc.title,
            content_type=doc.content_type.value,
            publish_date=doc.publish_date,
            author=doc.author,
            file_path=doc.file_path,
            file_hash=doc.file_hash,
            text_length=len(doc.text),
            chunk_count=len(doc_chunks),
            attachments=doc.attachments,
            extra=doc.extra,
            indexed_at=indexed_at,
        )

    # 写入分块记录
    meta_store.upsert_chunks_batch(chunks, indexed_at=indexed_at)
    logger.info("SQLite写入完成")

    # 统计
    elapsed = time.time() - start_time
    vs_stats = vector_store.get_stats()
    db_stats = meta_store.get_stats()

    print("\n" + "=" * 60)
    print("索引构建完成")
    print("=" * 60)
    print(f"\n耗时: {elapsed:.1f}s")
    print(f"文档数: {len(docs)}")
    print(f"分块数: {len(chunks)}")
    print(f"向量维度: {embedder.dimension}")
    print(f"\nChroma统计:")
    print(f"  总分块: {vs_stats['total_chunks']}")
    print(f"\nSQLite统计:")
    print(f"  总文档: {db_stats['total_documents']}")
    print(f"  总分块: {db_stats['total_chunks']}")
    print(f"  总字符: {db_stats['total_text_length']:,}")
    print(f"\n按站点分布:")
    for site, count in sorted(db_stats["by_site"].items(), key=lambda x: -x[1]):
        print(f"  {site}: {count} 文档")
    print(f"\n按类型分布:")
    for ct, count in sorted(db_stats["by_type"].items(), key=lambda x: -x[1]):
        print(f"  {ct}: {count} 文档")

    meta_store.close()


def main():
    parser = argparse.ArgumentParser(description="全量索引构建")
    parser.add_argument("--input", type=str, default="data/parsed_docs_current.json",
                        help="解析结果JSON文件路径")
    parser.add_argument("--chunk-size", type=int, default=1024, help="分块大小（长文档）")
    parser.add_argument("--chunk-overlap", type=int, default=120, help="分块重叠")
    parser.add_argument("--doc-keep-size", type=int, default=2048,
                        help="整文档保留阈值(<=此值不分块)")
    parser.add_argument("--embed-batch-size", type=int, default=32, help="嵌入批量大小")
    parser.add_argument("--provider", type=str, default="bge-local",
                        choices=["bge-local", "openai"], help="嵌入提供商")
    parser.add_argument("--rebuild", action="store_true", help="重建索引（清空已有）")
    parser.add_argument("--verbose", action="store_true", help="详细日志")
    args = parser.parse_args()

    # 配置日志
    logger.remove()
    log_level = "DEBUG" if args.verbose else "INFO"
    logger.add(sys.stderr, level=log_level, format="{time:HH:mm:ss} | {level:<7} | {message}")

    # 确保目录
    ensure_dirs()

    # 加载解析结果
    logger.info(f"加载解析结果: {args.input}")
    docs = load_parsed_docs(args.input)
    logger.info(f"加载完成: {len(docs)} 个文档")

    if not docs:
        logger.error("没有可索引的文档")
        return

    # 构建索引
    build_index(
        docs=docs,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        doc_keep_size=args.doc_keep_size,
        embed_batch_size=args.embed_batch_size,
        provider=args.provider,
        rebuild=args.rebuild,
    )


if __name__ == "__main__":
    main()
