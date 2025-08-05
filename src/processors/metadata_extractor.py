"""
元数据提取器
使用LLM提取文档的关键词和摘要等元数据
"""

from typing import List, Dict, Any
from tqdm import tqdm
from config.logging_config import get_logger
from ..core.llm_client import LLMClient

logger = get_logger(__name__)


class MetadataExtractor:
    """元数据提取器类"""
    
    def __init__(self):
        """初始化元数据提取器"""
        self.llm_client = LLMClient()
    
    def extract_metadata_for_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        为文本块提取元数据
        
        Args:
            chunks: 文本块列表
            
        Returns:
            包含元数据的文本块列表
        """
        logger.info(f"开始为 {len(chunks)} 个文本块提取元数据")
        
        # 先统计总数用于进度条
        total_chunks = len(chunks)
        
        for i, chunk in enumerate(tqdm(chunks, desc='提取元数据')):
            try:
                # 提取关键词
                keywords = self.llm_client.extract_keywords(chunk['content'])
                chunk['keywords'] = keywords
                
                # 生成摘要
                summary = self.llm_client.generate_summary(chunk['content'])
                chunk['summary'] = summary
                
            except Exception as e:
                logger.error(f"提取元数据失败 (chunk {i}): {e}")
                # 设置默认值
                chunk['keywords'] = []
                chunk['summary'] = ''
        
        logger.info("元数据提取完成")
        return chunks
    
    def extract_metadata_for_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        为单个文档提取元数据
        
        Args:
            document: 文档字典
            
        Returns:
            包含元数据的文档
        """
        try:
            # 提取文档级关键词
            keywords = self.llm_client.extract_keywords(document['content'], top_k=8)
            document['keywords'] = keywords
            
            # 生成文档摘要
            summary = self.llm_client.generate_summary(document['content'], max_length=300)
            document['summary'] = summary
            
        except Exception as e:
            logger.error(f"提取文档元数据失败: {e}")
            document['keywords'] = []
            document['summary'] = ''
        
        return document
    
    def extract_metadata_batch(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        批量提取文档元数据
        
        Args:
            documents: 文档列表
            
        Returns:
            包含元数据的文档列表
        """
        logger.info(f"开始批量提取 {len(documents)} 个文档的元数据")
        
        for doc in tqdm(documents, desc='提取文档元数据'):
            doc = self.extract_metadata_for_document(doc)
        
        logger.info("批量元数据提取完成")
        return documents 