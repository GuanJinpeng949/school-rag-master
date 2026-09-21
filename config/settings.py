"""全局配置模块 - 读取环境变量并定义所有配置参数"""
import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings
from pydantic import Field


# 项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CHROMA_DIR = DATA_DIR / "chroma"


class CrawlerSettings(BaseSettings):
    """爬虫配置"""
    crawl_delay: float = Field(default=2.0, alias="CRAWL_DELAY")
    crawl_depth: int = Field(default=2, alias="CRAWL_DEPTH")
    crawl_concurrent: int = Field(default=4, alias="CRAWL_CONCURRENT")
    crawl_robots_txt: bool = Field(default=True, alias="CRAWL_ROBOTS_TXT")
    crawl_user_agent: str = Field(
        default="SchoolRAG-Bot/1.0",
        alias="CRAWL_USER_AGENT",
    )
    raw_data_dir: str = Field(default=str(RAW_DATA_DIR), alias="RAW_DATA_DIR")

    class Config:
        env_file = ".env"
        extra = "ignore"


class LLMSettings(BaseSettings):
    """LLM配置"""
    deepseek_api_key: Optional[str] = Field(default=None, alias="DEEPSEEK_API_KEY")
    deepseek_base_url: str = Field(default="https://api.deepseek.com", alias="DEEPSEEK_BASE_URL")
    deepseek_model: str = Field(default="deepseek-chat", alias="DEEPSEEK_MODEL")

    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    openai_base_url: str = Field(default="https://api.openai.com/v1", alias="OPENAI_BASE_URL")
    openai_model: str = Field(default="gpt-4o", alias="OPENAI_MODEL")

    class Config:
        env_file = ".env"
        extra = "ignore"


class EmbeddingSettings(BaseSettings):
    """Embedding配置"""
    embedding_provider: str = Field(default="bge-local", alias="EMBEDDING_PROVIDER")
    bge_model_name: str = Field(default="BAAI/bge-large-zh-v1.5", alias="BGE_MODEL_NAME")
    # BGE 模型本地目录：配置后会直接加载本地文件，不再联网下载
    bge_local_path: Optional[str] = Field(default=None, alias="BGE_LOCAL_PATH")
    openai_embedding_model: str = Field(default="text-embedding-3-small", alias="OPENAI_EMBEDDING_MODEL")

    class Config:
        env_file = ".env"
        extra = "ignore"


class VectorDBSettings(BaseSettings):
    """向量数据库配置"""
    chroma_persist_dir: str = Field(default=str(CHROMA_DIR), alias="CHROMA_PERSIST_DIR")

    class Config:
        env_file = ".env"
        extra = "ignore"


class APISettings(BaseSettings):
    """API服务配置"""
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    api_reload: bool = Field(default=True, alias="API_RELOAD")

    class Config:
        env_file = ".env"
        extra = "ignore"


class RetrievalSettings(BaseSettings):
    """检索策略配置"""
    # 是否启用混合检索（向量+BM25）
    hybrid_search: bool = Field(default=True, alias="HYBRID_SEARCH")
    # RRF融合常数k（默认60，越小排名靠前权重越大）
    rrf_k: int = Field(default=60, alias="RRF_K")
    # 是否启用Cross-Encoder重排序
    use_reranker: bool = Field(default=True, alias="USE_RERANKER")
    # 重排序模型名称
    reranker_model: str = Field(
        default="BAAI/bge-reranker-v2-m3",
        alias="RERANKER_MODEL",
    )
    # 每文档最多保留的chunk数（去重策略）
    max_chunks_per_doc: int = Field(default=2, alias="MAX_CHUNKS_PER_DOC")
    # 相似度阈值
    score_threshold: float = Field(default=0.3, alias="SCORE_THRESHOLD")
    # 默认返回结果数
    default_top_k: int = Field(default=5, alias="DEFAULT_TOP_K")
    # 是否启用查询改写（口语→书面语，补充上下文）
    use_query_rewrite: bool = Field(default=False, alias="USE_QUERY_REWRITE")
    # 是否启用多查询分解（复杂查询拆分为子查询）
    use_query_decompose: bool = Field(default=False, alias="USE_QUERY_DECOMPOSE")

    class Config:
        env_file = ".env"
        extra = "ignore"


class AppSettings(BaseSettings):
    """应用全局配置"""
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    metadata_db: str = Field(default=str(DATA_DIR / "metadata.db"), alias="METADATA_DB")

    crawler: CrawlerSettings = CrawlerSettings()
    llm: LLMSettings = LLMSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    vector_db: VectorDBSettings = VectorDBSettings()
    api: APISettings = APISettings()
    retrieval: RetrievalSettings = RetrievalSettings()

    class Config:
        env_file = ".env"
        extra = "ignore"


# 全局配置实例
settings = AppSettings()


def ensure_dirs():
    """确保所有必要目录存在"""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    BM25_INDEX_DIR = DATA_DIR / "bm25"
    BM25_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    for subdir in ["html", "pdf", "images", "other"]:
        (RAW_DATA_DIR / subdir).mkdir(parents=True, exist_ok=True)
