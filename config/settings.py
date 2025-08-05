"""
配置管理模块
统一管理所有配置项，支持环境变量和默认值
"""

import os
from typing import Optional
from dataclasses import dataclass


@dataclass
class Settings:
    """配置管理类"""
    
    # Azure OpenAI 配置
    azure_openai_key: str = os.getenv("AZURE_OPENAI_KEY", "your_azure_openai_key")
    azure_openai_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "https://your-resource.openai.azure.com")
    azure_openai_deployment: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    azure_openai_api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
    
    # Ollama 配置
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_embedding_model: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "dengcao/Qwen3-Embedding-8B:Q8_0")
    
    # 向量数据库配置
    chroma_db_dir: str = os.getenv("CHROMA_DB_DIR", "./data/vector_db")
    collection_name: str = os.getenv("COLLECTION_NAME", "rag_db")
    
    # 日志配置
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file: str = os.getenv("LOG_FILE", "./logs/rag_system.log")
    
    # 处理配置
    min_chunk_size: int = int(os.getenv("MIN_CHUNK_SIZE", "200"))
    max_chunk_size: int = int(os.getenv("MAX_CHUNK_SIZE", "1500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "300"))
    batch_size: int = int(os.getenv("BATCH_SIZE", "10"))
    
    # LLM 配置
    llm_temperature: float = 0.1
    llm_max_tokens: int = 5000
    llm_timeout: int = 3000
    
    # 检索配置
    default_top_k: int = 20
    keyword_weight: float = 2.0
    
    def __post_init__(self):
        """初始化后处理"""
        # 确保目录存在
        os.makedirs(self.chroma_db_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)


# 全局配置实例
settings = Settings() 