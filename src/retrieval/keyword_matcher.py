"""
关键词匹配器
提供关键词提取和匹配功能
"""

import json
from typing import List, Set, Dict, Any
from config.logging_config import get_logger

logger = get_logger(__name__)


class KeywordMatcher:
    """关键词匹配器类"""
    
    def __init__(self):
        """初始化关键词匹配器"""
        pass
    
    def extract_keywords_from_text(self, text: str) -> List[str]:
        """
        从文本中提取关键词（简单实现）
        
        Args:
            text: 输入文本
            
        Returns:
            关键词列表
        """
        # 这里可以实现更复杂的关键词提取算法
        # 目前返回空列表，实际使用中会通过LLM提取
        return []
    
    def calculate_keyword_overlap(self, query_keywords: Set[str], doc_keywords: Set[str]) -> int:
        """
        计算关键词重叠度
        
        Args:
            query_keywords: 查询关键词集合
            doc_keywords: 文档关键词集合
            
        Returns:
            重叠关键词数量
        """
        return len(query_keywords & doc_keywords)
    
    def calculate_keyword_similarity(self, query_keywords: Set[str], doc_keywords: Set[str]) -> float:
        """
        计算关键词相似度（Jaccard相似度）
        
        Args:
            query_keywords: 查询关键词集合
            doc_keywords: 文档关键词集合
            
        Returns:
            相似度分数 (0-1)
        """
        if not query_keywords or not doc_keywords:
            return 0.0
        
        intersection = len(query_keywords & doc_keywords)
        union = len(query_keywords | doc_keywords)
        
        return intersection / union if union > 0 else 0.0
    
    def parse_keywords_from_metadata(self, metadata: Dict[str, Any]) -> Set[str]:
        """
        从元数据中解析关键词
        
        Args:
            metadata: 文档元数据
            
        Returns:
            关键词集合
        """
        try:
            keywords_str = metadata.get('keywords', '[]')
            if isinstance(keywords_str, str):
                keywords = json.loads(keywords_str)
            else:
                keywords = keywords_str
            
            return set(keywords) if keywords else set()
        except Exception as e:
            logger.warning(f"解析关键词失败: {e}")
            return set()
    
    def match_keywords(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        关键词匹配
        
        Args:
            query: 查询文本
            documents: 文档列表
            
        Returns:
            匹配结果列表
        """
        # 这里可以实现更复杂的关键词匹配逻辑
        # 目前返回原始文档列表
        return documents 