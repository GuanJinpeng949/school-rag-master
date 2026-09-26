"""Cross-Encoder重排序 - 提升检索精度

核心功能:
1. 使用BGE-reranker交叉编码器对检索结果重排序
2. 逐对计算query-document的相关性分数（比双塔更精确）
3. 模型懒加载，首次使用时才加载
4. 失败兜底：模型不可用或推理异常时保持原始排序

设计说明:
- 双塔编码器（BGE-large-zh）速度快但精度有限
- Cross-Encoder逐对计算query-doc相关性，精度显著更高
- 仅对检索后的候选结果重排序（数量少，计算开销可控）
- 默认使用BAAI/bge-reranker-v2-m3（多语言、轻量、效果好）
"""
import os
import time
from typing import Optional

from loguru import logger

from config.settings import settings, resolve_inference_device, resolve_use_fp16


class Reranker:
    """Cross-Encoder重排序器"""

    def __init__(self, model_name: Optional[str] = None,
                 timeout: int = 30, max_candidates: Optional[int] = None):
        """
        Args:
            model_name: 重排序模型名称
            timeout: 单次重排序超时时间(秒)
            max_candidates: 最大重排序候选数（过多会慢），默认取配置 RERANK_MAX_CANDIDATES
        """
        self.model_name = model_name or settings.retrieval.reranker_model
        self.timeout = timeout
        self.max_candidates = max_candidates or settings.retrieval.rerank_max_candidates
        self._model = None

    def _load_model(self):
        """懒加载重排序模型"""
        if self._model is not None:
            return

        device = resolve_inference_device()
        use_fp16 = resolve_use_fp16(device)

        try:
            from FlagEmbedding import FlagReranker

            logger.info(f"加载重排序模型: {self.model_name}")
            self._model = FlagReranker(
                self.model_name,
                use_fp16=use_fp16,  # 仅在CUDA设备上启用FP16
            )
            logger.info(
                f"重排序模型加载完成 (设备={device}, FP16={use_fp16})"
            )
        except ImportError:
            # FlagEmbedding未安装，回退到sentence-transformers
            logger.info("FlagEmbedding未安装，使用sentence-transformers CrossEncoder")
            try:
                from sentence_transformers import CrossEncoder

                automodel_args = None
                if use_fp16:
                    # CrossEncoder 通过 automodel_args 传入半精度，权重显存占用减半
                    # （bge-reranker-v2-m3 约2.11GB -> 约1.06GB）
                    import torch
                    automodel_args = {"torch_dtype": torch.float16}

                cross_encoder_kwargs = dict(
                    max_length=512, device=device, automodel_args=automodel_args
                )
                # 优先离线加载：local_files_only 是调用级参数，不会像 HF_HUB_OFFLINE 那样在
                # 导入时被固化为常量，因此本地加载失败后仍可回退到联网（镜像）。
                try:
                    self._model = CrossEncoder(
                        self.model_name, local_files_only=True, **cross_encoder_kwargs
                    )
                except Exception as e:
                    logger.info(
                        f"离线加载失败，回退到联网加载"
                        f"(镜像: {os.environ.get('HF_ENDPOINT', '未设置')}): {e}"
                    )
                    self._model = CrossEncoder(self.model_name, **cross_encoder_kwargs)
                logger.info(
                    f"CrossEncoder加载完成: {self.model_name} "
                    f"(设备={device}, FP16={use_fp16})"
                )
            except Exception as e:
                logger.error(f"重排序模型加载失败: {e}")
                logger.info("提示: 安装FlagEmbedding: pip install FlagEmbedding")
                logger.info("或安装sentence-transformers: pip install sentence-transformers")
                raise

    def rerank(self, query: str, candidates: list[dict],
               top_k: Optional[int] = None) -> list[dict]:
        """对候选结果重排序

        Args:
            query: 用户查询文本
            candidates: 候选结果列表，每项需包含 "text" 字段
            top_k: 重排序后返回的数量（默认返回全部候选）

        Returns:
            重排序后的结果列表（按相关性降序），每项新增 "rerank_score" 字段
        """
        if not candidates:
            return []

        top_k = top_k or len(candidates)

        # 限制候选数量，避免计算开销过大
        candidates = candidates[:self.max_candidates]

        try:
            self._load_model()
        except Exception as e:
            logger.warning(f"重排序模型不可用，保持原始排序: {e}")
            return candidates[:top_k]

        start_time = time.time()

        try:
            # 构造query-doc对
            pairs = [(query, c["text"]) for c in candidates]

            # 计算相关性分数
            if hasattr(self._model, 'compute_score'):
                # FlagReranker
                scores = self._model.compute_score(pairs, normalize_score=True)
                # 单条时返回float，多条时返回list
                if isinstance(scores, (int, float)):
                    scores = [scores]
            else:
                # sentence-transformers CrossEncoder
                # CrossEncoder.predict 默认已应用 nn.Sigmoid（num_labels==1），输出
                # 本身就是0~1的相关性概率，且可跨查询比较。原实现在此之上再做
                # min-max 拉伸，会让第一名恒为1.0、末位恒为0.0，破坏区分度。
                scores = [float(s) for s in self._model.predict(pairs, batch_size=32)]

            elapsed = time.time() - start_time
            logger.info(
                f"重排序完成: {len(candidates)} 条候选, 耗时{elapsed:.2f}s"
            )

            # 赋予rerank_score并排序
            for i, score in enumerate(scores):
                candidates[i]["rerank_score"] = float(score)

            # 按rerank_score降序排列
            results = sorted(candidates, key=lambda x: -x.get("rerank_score", 0))

            return results[:top_k]

        except Exception as e:
            elapsed = time.time() - start_time
            logger.warning(f"重排序失败(耗时{elapsed:.2f}s)，保持原始排序: {e}")
            # 失败时保持原始顺序，不添加rerank_score
            return candidates[:top_k]