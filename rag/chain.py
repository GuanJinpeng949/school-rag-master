"""RAG Chain - 检索+生成串联，对外统一接口

核心功能:
1. 接收用户查询，串联检索和生成
2. 返回统一的RAGResponse结构
3. 支持流式输出
4. 支持元数据过滤
"""
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from rag.retriever import Retriever, SearchResult
from rag.generator import Generator


@dataclass
class RAGResponse:
    """RAG回答结果"""
    # 用户原始查询
    query: str
    # AI生成的回答
    answer: str
    # 检索到的相关文档
    results: list[SearchResult] = field(default_factory=list)
    # 来源链接汇总
    sources: list[dict] = field(default_factory=list)
    # 是否使用了LLM生成
    has_llm: bool = False
    # 错误信息
    error: Optional[str] = None
    # 检索耗时(ms)
    retrieve_time_ms: int = 0
    # 生成耗时(ms)
    generate_time_ms: int = 0

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "results": [r.to_dict() for r in self.results],
            "sources": self.sources,
            "has_llm": self.has_llm,
            "error": self.error,
            "retrieve_time_ms": self.retrieve_time_ms,
            "generate_time_ms": self.generate_time_ms,
        }


class RAGChain:
    """RAG链 - 检索增强生成的完整流程

    用法:
        chain = RAGChain()
        response = chain.ask("奖学金申请条件")
        print(response.answer)

        # 流式
        for chunk in chain.ask_stream("选课流程"):
            print(chunk, end="")
    """

    def __init__(self, retriever: Optional[Retriever] = None,
                 generator: Optional[Generator] = None,
                 llm_provider: Optional[str] = None):
        """
        Args:
            retriever: 检索器实例
            generator: 生成器实例
            llm_provider: LLM提供商("deepseek"/"openai")
        """
        self.retriever = retriever or Retriever()
        self.generator = generator or Generator(provider=llm_provider)
        logger.info("RAGChain初始化完成")

    def ask(self, query: str, top_k: int = 5,
            filter_site: Optional[str] = None,
            filter_type: Optional[str] = None) -> RAGResponse:
        """执行完整的RAG流程：检索 + 生成

        Args:
            query: 用户查询
            top_k: 返回结果数
            filter_site: 按站点过滤
            filter_type: 按类型过滤

        Returns:
            RAGResponse
        """
        import time

        # Step 1: 检索
        retrieve_start = time.time()
        results = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            filter_site=filter_site,
            filter_type=filter_type,
        )
        retrieve_ms = int((time.time() - retrieve_start) * 1000)

        # Step 2: 生成
        generate_start = time.time()
        gen_result = self.generator.generate(query=query, results=results)
        generate_ms = int((time.time() - generate_start) * 1000)

        return RAGResponse(
            query=query,
            answer=gen_result["answer"],
            results=results,
            sources=gen_result["sources"],
            has_llm=gen_result["has_llm"],
            error=gen_result.get("error"),
            retrieve_time_ms=retrieve_ms,
            generate_time_ms=generate_ms,
        )

    def ask_stream(self, query: str, top_k: int = 5,
                   filter_site: Optional[str] = None,
                   filter_type: Optional[str] = None):
        """流式RAG流程：检索 + 流式生成

        Args:
            query: 用户查询
            top_k: 返回结果数
            filter_site: 按站点过滤
            filter_type: 按类型过滤

        Yields:
            流式输出的文本片段
        """
        # Step 1: 检索（非流式）
        results = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            filter_site=filter_site,
            filter_type=filter_type,
        )

        # Step 2: 流式生成
        gen_result = self.generator.generate(query=query, results=results, stream=True)

        if "answer_stream" in gen_result:
            for chunk in gen_result["answer_stream"]:
                yield chunk
        else:
            yield gen_result["answer"]

    def search_only(self, query: str, top_k: int = 5,
                    filter_site: Optional[str] = None,
                    filter_type: Optional[str] = None) -> list[SearchResult]:
        """仅检索，不调用LLM生成

        适用于：搜索建议、快速预览、LLM不可用时
        """
        return self.retriever.retrieve(
            query=query,
            top_k=top_k,
            filter_site=filter_site,
            filter_type=filter_type,
        )

    def generate_stream(self, query: str, results: list[SearchResult]):
        """用已有检索结果做流式生成（不重复检索）

        适用于：SSE场景，先检索推送元数据，再流式生成

        Args:
            query: 用户查询
            results: 已完成的检索结果

        Yields:
            流式输出的文本片段
        """
        gen_result = self.generator.generate(query=query, results=results, stream=True)

        if "answer_stream" in gen_result:
            for chunk in gen_result["answer_stream"]:
                yield chunk
        else:
            yield gen_result["answer"]
