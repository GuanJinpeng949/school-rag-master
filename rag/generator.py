"""生成器 - Prompt模板 + LLM调用 + 来源引用

核心功能:
1. 调用LLM(DeepSeek/OpenAI)生成回答
2. 将检索结果格式化为Prompt上下文（精简版，减少token）
3. 确保回答中引用来源链接
4. 超时保护：LLM超时时返回纯检索结果
"""
import time
from typing import Optional

from loguru import logger

from config.settings import settings
from rag.prompts import QA_PROMPT_TEMPLATE, NO_RESULT_PROMPT_TEMPLATE, SYSTEM_PROMPT
from rag.retriever import SearchResult

# 上下文中每个结果的最大字符数（截断过长文本，减少输入token）
# 校园文档关键信息（截止日期、具体条件、链接）往往在文档后半段
# 500字截断太短容易丢失关键信息，提升到800字
MAX_CONTEXT_CHARS_PER_RESULT = 800


class Generator:
    """回答生成器"""

    def __init__(self, provider: Optional[str] = None, timeout: int = 60):
        """
        Args:
            provider: LLM提供商 "deepseek" 或 "openai"
            timeout: 生成超时时间(秒)，默认60s
        """
        self.provider = provider or self._detect_provider()
        self.timeout = timeout
        self._llm = None

    def _detect_provider(self) -> str:
        """自动检测可用的LLM提供商"""
        if settings.llm.deepseek_api_key:
            return "deepseek"
        elif settings.llm.openai_api_key:
            return "openai"
        return "deepseek"  # 默认

    def _get_llm(self):
        """懒加载LLM实例"""
        if self._llm is not None:
            return self._llm

        try:
            if self.provider == "deepseek":
                from langchain_openai import ChatOpenAI
                self._llm = ChatOpenAI(
                    model=settings.llm.deepseek_model,
                    openai_api_key=settings.llm.deepseek_api_key,
                    openai_api_base=settings.llm.deepseek_base_url,
                    temperature=0.3,
                    max_tokens=768,       # 从2048降到768，校园问答不需要太长
                    request_timeout=(10, self.timeout),  # 连接超时10s，读取超时60s
                )
                logger.info(f"LLM已加载: DeepSeek ({settings.llm.deepseek_model})")
            elif self.provider == "openai":
                from langchain_openai import ChatOpenAI
                self._llm = ChatOpenAI(
                    model=settings.llm.openai_model,
                    openai_api_key=settings.llm.openai_api_key,
                    openai_api_base=settings.llm.openai_base_url,
                    temperature=0.3,
                    max_tokens=768,
                    request_timeout=(10, self.timeout),
                )
                logger.info(f"LLM已加载: OpenAI ({settings.llm.openai_model})")
            else:
                raise ValueError(f"不支持的LLM提供商: {self.provider}")
        except Exception as e:
            logger.error(f"LLM加载失败: {e}")
            raise

        return self._llm

    def generate(self, query: str, results: list[SearchResult],
                 stream: bool = False) -> dict:
        """根据检索结果生成回答

        Args:
            query: 用户查询
            results: 检索结果列表
            stream: 是否流式输出

        Returns:
            {
                "answer": 生成的回答文本,
                "sources": 来源链接列表,
                "has_llm": 是否使用了LLM生成,
                "error": 错误信息(如有)
            }
        """
        # 收集来源
        sources = self._collect_sources(results)

        # 无检索结果时返回固定提示
        if not results:
            return {
                "answer": NO_RESULT_PROMPT_TEMPLATE,
                "sources": [],
                "has_llm": False,
                "error": None,
            }

        # 格式化上下文（精简版）
        context = self._format_context(results)

        # 构建Prompt
        prompt = QA_PROMPT_TEMPLATE.format(
            context=context,
            question=query,
        )

        # 调用LLM生成回答
        try:
            llm = self._get_llm()

            if stream:
                return {
                    "answer_stream": self._generate_stream(llm, prompt),
                    "sources": sources,
                    "has_llm": True,
                    "error": None,
                }

            start_time = time.time()
            from langchain_core.messages import HumanMessage, SystemMessage
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
            response = llm.invoke(messages)
            elapsed = time.time() - start_time

            answer = response.content
            logger.info(f"LLM生成完成: {len(answer)}字, 耗时{elapsed:.1f}s")

            return {
                "answer": answer,
                "sources": sources,
                "has_llm": True,
                "error": None,
            }

        except Exception as e:
            logger.error(f"LLM生成失败: {e}")
            # LLM失败时，返回纯检索结果摘要
            fallback_answer = self._fallback_answer(query, results)
            return {
                "answer": fallback_answer,
                "sources": sources,
                "has_llm": False,
                "error": str(e),
            }

    def _generate_stream(self, llm, prompt: str):
        """流式生成回答"""
        from langchain_core.messages import HumanMessage, SystemMessage
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
        for chunk in llm.stream(messages):
            if chunk.content:
                yield chunk.content

    def _format_context(self, results: list[SearchResult]) -> str:
        """将检索结果格式化为Prompt中的上下文（精简版，截断过长文本）

        优化策略：
        - 每个结果文本截断到 MAX_CONTEXT_CHARS_PER_RESULT 字符
        - 去除冗余格式，减少输入token
        """
        context_parts = []
        for i, r in enumerate(results, 1):
            # 截断过长文本，显著减少输入token
            text = r.text[:MAX_CONTEXT_CHARS_PER_RESULT]
            if len(r.text) > MAX_CONTEXT_CHARS_PER_RESULT:
                text += "..."

            source_label = f"[{r.source_site}]" if r.source_site else ""
            url_label = f"({r.source_url})" if r.source_url else ""

            part = (
                f"文档{i}{source_label}: {r.title}{url_label}\n"
                f"{text}\n"
            )
            context_parts.append(part)

        return "\n---\n".join(context_parts)

    def _collect_sources(self, results: list[SearchResult]) -> list[dict]:
        """收集来源链接"""
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
        return sources

    def _fallback_answer(self, query: str, results: list[SearchResult]) -> str:
        """LLM不可用时的降级回答（纯检索结果拼接）"""
        parts = [f"关于「{query}」，以下是检索到的相关信息：\n"]
        for i, r in enumerate(results, 1):
            url_link = f" [查看原文]({r.source_url})" if r.source_url else ""
            site_label = f"[{r.source_site}]" if r.source_site else ""
            parts.append(f"{i}. {site_label} **{r.title}**{url_link}")
            # 取文本前200字作为摘要
            preview = r.text[:200].strip()
            if len(r.text) > 200:
                preview += "..."
            parts.append(f"   {preview}\n")

        parts.append("\n> 注：AI摘要生成服务暂时不可用，以上为原始检索结果。")
        return "\n".join(parts)
