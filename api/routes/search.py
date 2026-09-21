"""搜索接口 - 核心搜索功能

接口:
- POST /api/search       - 搜索（检索+LLM生成摘要）
- POST /api/search/stream - 流式搜索（SSE，先推送检索结果再流式生成）
- GET  /api/suggest      - 搜索建议（纯检索，快速）
"""
import asyncio
import hashlib
import json
import time
from collections import OrderedDict
from typing import Optional

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse

from loguru import logger

from api.schemas import (
    SearchRequest, SearchResponse, SuggestResponse,
    ResultItem, SourceItem,
)
from api.dependencies import get_rag_chain
from rag.chain import RAGChain
from rag.retriever import SearchResult

router = APIRouter(prefix="/api", tags=["search"])

# ============ 查询缓存 ============
_MAX_CACHE_SIZE = 256
_query_cache: OrderedDict[str, dict] = OrderedDict()


def _cache_key(query: str, top_k: int, filter_site: Optional[str],
               filter_type: Optional[str], use_llm: bool) -> str:
    raw = f"{query}|{top_k}|{filter_site}|{filter_type}|{use_llm}"
    return hashlib.md5(raw.encode()).hexdigest()


def _get_cached(key: str) -> Optional[dict]:
    if key in _query_cache:
        _query_cache.move_to_end(key)
        return _query_cache[key]
    return None


def _set_cached(key: str, value: dict):
    _query_cache[key] = value
    _query_cache.move_to_end(key)
    if len(_query_cache) > _MAX_CACHE_SIZE:
        _query_cache.popitem(last=False)


@router.post("/search", response_model=SearchResponse)
async def search(
    req: SearchRequest,
    chain: RAGChain = Depends(get_rag_chain),
):
    """搜索接口 - 检索 + LLM生成摘要（异步非阻塞）"""
    start_time = time.time()
    logger.info(f"搜索请求: query='{req.query[:30]}', top_k={req.top_k}, use_llm={req.use_llm}")

    # 检查缓存
    cache_k = _cache_key(req.query, req.top_k, req.filter_site, req.filter_type, req.use_llm)
    cached = _get_cached(cache_k)
    if cached:
        logger.info(f"命中缓存: query='{req.query[:30]}'")
        return SearchResponse(**cached)

    try:
        if req.use_llm:
            # 异步执行同步的RAG流程，避免阻塞事件循环
            rag_resp = await asyncio.to_thread(
                chain.ask,
                query=req.query,
                top_k=req.top_k,
                filter_site=req.filter_site,
                filter_type=req.filter_type,
            )
            total_ms = int((time.time() - start_time) * 1000)

            result = SearchResponse(
                query=rag_resp.query,
                answer=rag_resp.answer,
                results=[_to_result_item(r) for r in rag_resp.results],
                sources=[SourceItem(**s) for s in rag_resp.sources],
                has_llm=rag_resp.has_llm,
                error=rag_resp.error,
                retrieve_time_ms=rag_resp.retrieve_time_ms,
                generate_time_ms=rag_resp.generate_time_ms,
                total_time_ms=total_ms,
            )
        else:
            result = await _search_only(req, chain, start_time)

        # 写入缓存
        _set_cached(cache_k, result.model_dump())
        return result

    except Exception as e:
        logger.error(f"搜索失败: {e}")
        total_ms = int((time.time() - start_time) * 1000)
        return SearchResponse(
            query=req.query,
            error=str(e),
            total_time_ms=total_ms,
        )


@router.post("/search/stream")
async def search_stream(
    req: SearchRequest,
    chain: RAGChain = Depends(get_rag_chain),
):
    """流式搜索接口 - SSE (Server-Sent Events)

    增强协议：先推送检索元数据（来源、耗时），再流式推送生成内容
    事件格式:
      data: {"type":"meta","retrieve_time_ms":...,"sources":[...],"results":[...]}
      data: {"type":"token","content":"..."}
      data: {"type":"done","generate_time_ms":...}
    """
    logger.info(f"流式搜索请求: query='{req.query[:30]}'")

    def event_generator():
        # FastAPI 会在独立线程中运行同步生成器，无需手动创建线程
        try:
            # Step 1: 检索
            t0 = time.time()
            try:
                results = chain.retriever.retrieve(
                    query=req.query,
                    top_k=req.top_k,
                    filter_site=req.filter_site,
                    filter_type=req.filter_type,
                )
            except Exception as e:
                data = json.dumps({"type": "error", "message": str(e)}, ensure_ascii=False)
                yield f"data: {data}\n\n"
                return

            retrieve_ms = int((time.time() - t0) * 1000)

            # 收集来源
            seen_urls = set()
            sources = []
            for r in results:
                if r.source_url and r.source_url not in seen_urls:
                    seen_urls.add(r.source_url)
                    sources.append({
                        "url": r.source_url,
                        "title": r.title,
                        "site": r.source_site,
                        "type": r.content_type,
                    })

            # 推送检索元数据（前端可立即展示来源链接）
            meta_event = {
                "type": "meta",
                "retrieve_time_ms": retrieve_ms,
                "sources": sources,
                "results": [_to_result_item(r).model_dump() for r in results],
                "has_llm": req.use_llm,
            }
            yield f"data: {json.dumps(meta_event, ensure_ascii=False)}\n\n"

            # Step 2: 流式生成（如果需要LLM）
            if not req.use_llm or not results:
                done_event = {"type": "done", "generate_time_ms": 0}
                yield f"data: {json.dumps(done_event, ensure_ascii=False)}\n\n"
                return

            t0 = time.time()
            for chunk in chain.generate_stream(
                query=req.query,
                results=results,
            ):
                data = json.dumps({"type": "token", "content": chunk}, ensure_ascii=False)
                yield f"data: {data}\n\n"

            generate_ms = int((time.time() - t0) * 1000)
            done_event = {"type": "done", "generate_time_ms": generate_ms}
            yield f"data: {json.dumps(done_event, ensure_ascii=False)}\n\n"

        except Exception as e:
            logger.error(f"流式生成失败: {e}")
            data = json.dumps({"type": "error", "message": str(e)}, ensure_ascii=False)
            yield f"data: {data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/suggest", response_model=SuggestResponse)
async def suggest(
    q: str = Query(..., min_length=1, max_length=200, description="查询文本"),
    top_k: int = Query(default=5, ge=1, le=10, description="返回结果数"),
    chain: RAGChain = Depends(get_rag_chain),
):
    """搜索建议 - 纯检索，快速响应（不调用LLM）"""
    start_time = time.time()
    logger.info(f"搜索建议: q='{q[:30]}'")

    try:
        results = await asyncio.to_thread(
            chain.search_only, query=q, top_k=top_k
        )
        retrieve_ms = int((time.time() - start_time) * 1000)

        return SuggestResponse(
            query=q,
            results=[_to_result_item(r) for r in results],
            retrieve_time_ms=retrieve_ms,
        )
    except Exception as e:
        logger.error(f"搜索建议失败: {e}")
        return SuggestResponse(query=q, retrieve_time_ms=0)


async def _search_only(req: SearchRequest, chain: RAGChain, start_time: float) -> SearchResponse:
    """仅检索模式（不调用LLM，异步执行）"""
    results = await asyncio.to_thread(
        chain.search_only,
        query=req.query,
        top_k=req.top_k,
        filter_site=req.filter_site,
        filter_type=req.filter_type,
    )
    retrieve_ms = int((time.time() - start_time) * 1000)

    seen_urls = set()
    sources = []
    for r in results:
        if r.source_url and r.source_url not in seen_urls:
            seen_urls.add(r.source_url)
            sources.append(SourceItem(
                url=r.source_url, title=r.title,
                site=r.source_site, type=r.content_type,
            ))

    return SearchResponse(
        query=req.query,
        results=[_to_result_item(r) for r in results],
        sources=sources,
        has_llm=False,
        retrieve_time_ms=retrieve_ms,
        total_time_ms=retrieve_ms,
    )


def _to_result_item(r: SearchResult) -> ResultItem:
    """SearchResult → ResultItem"""
    return ResultItem(
        text=r.text,
        source_url=r.source_url,
        source_site=r.source_site,
        title=r.title,
        content_type=r.content_type,
        publish_date=r.publish_date,
        score=r.score,
        doc_id=r.doc_id,
        chunk_index=r.chunk_index,
        total_chunks=r.total_chunks,
        score_source=r.score_source,
    )
