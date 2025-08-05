"""
向量嵌入模块
封装Ollama本地embedding模型
"""

from langchain_ollama import OllamaEmbeddings
from config.settings import settings
from config.logging_config import get_logger

logger = get_logger(__name__)


class EmbeddingModel:
    """向量嵌入模型封装类"""
    
    def __init__(self):
        """初始化嵌入模型"""
        self.model = OllamaEmbeddings(
            model=settings.ollama_embedding_model,
            base_url=settings.ollama_base_url
        )
        logger.info(f"初始化嵌入模型: {settings.ollama_embedding_model}")
    
    def embed_text(self, text: str) -> list:
        """
        对文本进行向量化
        
        Args:
            text: 输入文本
            
        Returns:
            向量表示
        """
        try:
            return self.model.embed_query(text)
        except Exception as e:
            logger.error(f"文本向量化失败: {e}")
            raise
    
    def embed_documents(self, texts: list) -> list:
        """
        批量对文档进行向量化
        
        Args:
            texts: 文本列表
            
        Returns:
            向量表示列表
        """
        try:
            return self.model.embed_documents(texts)
        except Exception as e:
            logger.error(f"文档批量向量化失败: {e}")
            raise
    
    def get_model(self) -> OllamaEmbeddings:
        """
        获取原始嵌入模型
        
        Returns:
            OllamaEmbeddings实例
        """
        return self.model 