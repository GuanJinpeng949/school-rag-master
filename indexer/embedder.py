"""向量嵌入器 - 将文本转为向量

核心功能:
1. BGE-large-zh-v1.5 本地嵌入（默认，中文语义效果好）
2. OpenAI text-embedding-3-small（备选，需API Key）
3. 批量嵌入支持
4. 模型懒加载（首次使用时才加载）

设计说明:
- BGE-large-zh 本地运行免API费用，中文语义效果优于OpenAI
- 模型首次加载较慢（约10s），后续推理快（<50ms/条）
- 支持批量嵌入以减少推理开销
"""
from pathlib import Path
from typing import Optional

from loguru import logger

from config.settings import settings, resolve_inference_device, resolve_use_fp16


class Embedder:
    """向量嵌入器"""

    def __init__(self, provider: Optional[str] = None, model_name: Optional[str] = None):
        """
        Args:
            provider: 嵌入提供商 "bge-local" 或 "openai"
            model_name: 模型名称（覆盖默认配置）
        """
        self.provider = provider or settings.embedding.embedding_provider
        self.model_name = model_name or settings.embedding.bge_model_name
        self._model = None
        self._dimension: Optional[int] = None

    @property
    def dimension(self) -> int:
        """向量维度"""
        if self._dimension is None:
            # BGE-large-zh-v1.5 的维度是 1024
            # OpenAI text-embedding-3-small 的维度是 1536
            if self.provider == "bge-local":
                self._dimension = 1024
            elif self.provider == "openai":
                self._dimension = 1536
            else:
                self._dimension = 1024
        return self._dimension

    def _load_bge_model(self):
        """加载BGE本地模型"""
        if self._model is not None:
            return

        try:
            import os

            # 关键：HF_ENDPOINT / HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE 都是第三方库在导入时读取的常量，
            # 必须在 import sentence_transformers 之前确定，导入之后再改环境变量不会生效。
            local_model_path = settings.embedding.bge_local_path or os.environ.get("BGE_LOCAL_PATH", "")

            if local_model_path and Path(local_model_path).exists():
                model_source = local_model_path
                logger.info(f"从本地路径加载BGE模型: {local_model_path}")
            else:
                model_source = self.model_name
                os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
                logger.info(
                    f"本地模型未配置，从镜像加载BGE模型: {model_source} "
                    f"(HF_ENDPOINT={os.environ['HF_ENDPOINT']})"
                )

            from sentence_transformers import SentenceTransformer

            device = resolve_inference_device()
            use_fp16 = resolve_use_fp16(device)

            st_kwargs = dict(device=device, trust_remote_code=False)
            # 优先离线加载：local_files_only 是调用级参数，不会像 HF_HUB_OFFLINE 那样在
            # 导入时被固化为常量，因此本地加载失败后仍可回退到联网（镜像）。
            try:
                self._model = SentenceTransformer(
                    model_source, local_files_only=True, **st_kwargs
                )
            except Exception as e:
                logger.info(
                    f"离线加载失败，回退到联网加载"
                    f"(镜像: {os.environ.get('HF_ENDPOINT', '未设置')}): {e}"
                )
                self._model = SentenceTransformer(model_source, **st_kwargs)

            if use_fp16:
                # FP16推理：权重显存占用减半（bge-large 约1.24GB -> 约0.62GB）
                self._model.half()

            # 获取实际维度
            test_emb = self._model.encode(["测试"], normalize_embeddings=True)
            self._dimension = test_emb.shape[1]
            logger.info(
                f"BGE模型加载完成, 维度={self._dimension}, 设备={device}, FP16={use_fp16}"
            )
        except Exception as e:
            logger.error(f"BGE模型加载失败: {e}")
            logger.info("提示: 设置 BGE_LOCAL_PATH=<模型路径> 或 HF_ENDPOINT=https://hf-mirror.com")
            raise

    def embed_query(self, text: str) -> list[float]:
        """嵌入单个查询文本

        Args:
            text: 查询文本

        Returns:
            向量（list of float）
        """
        if self.provider == "bge-local":
            return self._embed_bge([text])[0]
        elif self.provider == "openai":
            return self._embed_openai([text])[0]
        else:
            raise ValueError(f"不支持的嵌入提供商: {self.provider}")

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """批量嵌入文本

        Args:
            texts: 文本列表
            batch_size: 批量大小

        Returns:
            向量列表
        """
        if not texts:
            return []

        if self.provider == "bge-local":
            return self._embed_bge(texts, batch_size)
        elif self.provider == "openai":
            return self._embed_openai(texts)
        else:
            raise ValueError(f"不支持的嵌入提供商: {self.provider}")

    def _embed_bge(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """使用BGE本地模型嵌入"""
        self._load_bge_model()

        # BGE 推荐对查询添加指令前缀以提高检索效果
        # 但对文档不需要添加前缀，查询时再添加
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
        )

        return embeddings.tolist()

    def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        """使用OpenAI API嵌入"""
        try:
            import openai
        except ImportError:
            raise ImportError("openai库未安装，请运行: pip install openai")

        api_key = settings.embedding.openai_embedding_model and settings.llm.openai_api_key
        if not api_key:
            raise ValueError("OpenAI API Key未配置")

        client = openai.OpenAI(
            api_key=settings.llm.openai_api_key,
            base_url=settings.llm.openai_base_url,
        )

        model = settings.embedding.openai_embedding_model
        response = client.embeddings.create(
            input=texts,
            model=model,
        )

        # 按 index 排序
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]
