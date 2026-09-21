"""Pydantic请求/响应模型"""

from typing import Optional

from pydantic import BaseModel, Field


# ============ 请求模型 ============

class SearchRequest(BaseModel):
    """搜索请求"""
    query: str = Field(..., min_length=1, max_length=500, description="搜索查询")
    top_k: int = Field(default=5, ge=1, le=20, description="返回结果数")
    filter_site: Optional[str] = Field(default=None, description="按站点过滤")
    filter_type: Optional[str] = Field(default=None, description="按类型过滤(html/pdf)")
    use_llm: bool = Field(default=True, description="是否使用LLM生成摘要")


class SuggestRequest(BaseModel):
    """搜索建议请求"""
    query: str = Field(..., min_length=1, max_length=200, description="查询文本")
    top_k: int = Field(default=5, ge=1, le=10, description="返回结果数")


# ============ 响应模型 ============

class SourceItem(BaseModel):
    """来源链接"""
    url: str
    title: str
    site: str
    type: str


class ResultItem(BaseModel):
    """单条检索结果"""
    text: str
    source_url: str
    source_site: str
    title: str
    content_type: str
    publish_date: str = ""
    score: float = 0.0
    doc_id: str = ""
    chunk_index: int = 0
    total_chunks: int = 1
    score_source: str = "vector"


class SearchResponse(BaseModel):
    """搜索响应"""
    query: str
    answer: str = ""
    results: list[ResultItem] = []
    sources: list[SourceItem] = []
    has_llm: bool = False
    error: Optional[str] = None
    retrieve_time_ms: int = 0
    generate_time_ms: int = 0
    total_time_ms: int = 0


class SuggestResponse(BaseModel):
    """搜索建议响应"""
    query: str
    results: list[ResultItem] = []
    retrieve_time_ms: int = 0


class StatsResponse(BaseModel):
    """索引统计响应"""
    total_documents: int = 0
    total_chunks: int = 0
    total_text_length: int = 0
    indexed_documents: int = 0
    by_site: dict[str, int] = {}
    by_type: dict[str, int] = {}
    chroma_chunks: int = 0
    site_distribution: dict[str, int] = {}


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = "ok"
    version: str = "1.0.0"
    chroma_ok: bool = False
    sqlite_ok: bool = False
