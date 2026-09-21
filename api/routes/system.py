"""系统接口 - 索引统计、健康检查

接口:
- GET /api/stats   - 索引统计
- GET /api/health  - 健康检查
"""
from fastapi import APIRouter, Depends

from loguru import logger

from api.schemas import StatsResponse, HealthResponse
from api.dependencies import get_vector_store, get_metadata_store
from indexer.vector_store import VectorStore
from indexer.metadata_store import MetadataStore

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/stats", response_model=StatsResponse)
async def stats(
    vs: VectorStore = Depends(get_vector_store),
    ms: MetadataStore = Depends(get_metadata_store),
):
    """获取索引统计信息"""
    try:
        db_stats = ms.get_stats()
        vs_stats = vs.get_stats()

        return StatsResponse(
            total_documents=db_stats["total_documents"],
            total_chunks=db_stats["total_chunks"],
            total_text_length=db_stats["total_text_length"],
            indexed_documents=db_stats["indexed_documents"],
            by_site=db_stats["by_site"],
            by_type=db_stats["by_type"],
            chroma_chunks=vs_stats["total_chunks"],
            site_distribution=vs_stats.get("site_distribution", {}),
        )
    except Exception as e:
        logger.error(f"获取统计失败: {e}")
        return StatsResponse()


@router.get("/health", response_model=HealthResponse)
async def health(
    vs: VectorStore = Depends(get_vector_store),
    ms: MetadataStore = Depends(get_metadata_store),
):
    """健康检查"""
    chroma_ok = False
    sqlite_ok = False

    try:
        vs_stats = vs.get_stats()
        chroma_ok = vs_stats["total_chunks"] >= 0
    except Exception:
        pass

    try:
        ms.get_stats()
        sqlite_ok = True
    except Exception:
        pass

    return HealthResponse(
        status="ok" if (chroma_ok and sqlite_ok) else "degraded",
        chroma_ok=chroma_ok,
        sqlite_ok=sqlite_ok,
    )
