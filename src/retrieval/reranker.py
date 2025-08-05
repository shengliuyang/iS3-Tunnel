"""
重排序器
提供基于多维度特征的结果重排序功能
"""

from typing import List, Dict, Any, Tuple
from config.logging_config import get_logger

logger = get_logger(__name__)


class Reranker:
    """重排序器类"""
    
    def __init__(self):
        """初始化重排序器"""
        pass
    
    def rerank_by_multiple_features(self, 
                                  documents: List[Tuple[Any, float]], 
                                  query: str,
                                  weights: Dict[str, float] = None) -> List[Tuple[Any, float]]:
        """
        基于多维度特征重排序
        
        Args:
            documents: (文档, 分数)元组列表
            query: 查询文本
            weights: 特征权重字典
            
        Returns:
            重排序后的(文档, 分数)元组列表
        """
        if not documents:
            return documents
        
        # 默认权重
        default_weights = {
            'semantic_score': 0.6,
            'keyword_overlap': 0.2,
            'doc_type_relevance': 0.1,
            'freshness': 0.1
        }
        weights = weights or default_weights
        
        # 计算综合分数
        scored_docs = []
        for doc, semantic_score in documents:
            # 提取特征
            features = self._extract_features(doc, query)
            
            # 计算综合分数
            final_score = (
                weights['semantic_score'] * (1 - semantic_score) +  # 语义分数越小越好
                weights['keyword_overlap'] * features['keyword_overlap'] +
                weights['doc_type_relevance'] * features['doc_type_relevance'] +
                weights['freshness'] * features['freshness']
            )
            
            scored_docs.append((doc, final_score))
        
        # 按综合分数排序（降序）
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return scored_docs
    
    def _extract_features(self, doc: Any, query: str) -> Dict[str, float]:
        """
        提取文档特征
        
        Args:
            doc: 文档对象
            query: 查询文本
            
        Returns:
            特征字典
        """
        features = {
            'keyword_overlap': 0.0,
            'doc_type_relevance': 0.5,  # 默认中等相关性
            'freshness': 0.5  # 默认中等新鲜度
        }
        
        try:
            # 提取关键词重叠度
            if hasattr(doc, 'metadata') and doc.metadata:
                keywords = doc.metadata.get('keywords', '[]')
                if keywords:
                    # 这里可以计算关键词重叠度
                    features['keyword_overlap'] = 0.5  # 简化实现
            
            # 文档类型相关性
            if hasattr(doc, 'metadata') and doc.metadata:
                doc_type = doc.metadata.get('doc_type', 'unknown')
                # 根据查询和文档类型计算相关性
                features['doc_type_relevance'] = self._calculate_doc_type_relevance(query, doc_type)
            
            # 文档新鲜度
            if hasattr(doc, 'metadata') and doc.metadata:
                timestamp = doc.metadata.get('timestamp', '')
                features['freshness'] = self._calculate_freshness(timestamp)
                
        except Exception as e:
            logger.warning(f"提取特征失败: {e}")
        
        return features
    
    def _calculate_doc_type_relevance(self, query: str, doc_type: str) -> float:
        """
        计算文档类型相关性
        
        Args:
            query: 查询文本
            doc_type: 文档类型
            
        Returns:
            相关性分数 (0-1)
        """
        # 简化的相关性计算
        query_lower = query.lower()
        
        if doc_type == 'standard' and any(word in query_lower for word in ['标准', '规范', '规程']):
            return 0.9
        elif doc_type == 'design' and any(word in query_lower for word in ['设计', '施工']):
            return 0.8
        elif doc_type == 'report' and any(word in query_lower for word in ['报告', '分析']):
            return 0.7
        else:
            return 0.5
    
    def _calculate_freshness(self, timestamp: str) -> float:
        """
        计算文档新鲜度
        
        Args:
            timestamp: 时间戳
            
        Returns:
            新鲜度分数 (0-1)
        """
        # 简化的新鲜度计算
        if not timestamp:
            return 0.5
        
        try:
            # 这里可以实现更复杂的时间计算逻辑
            return 0.7  # 默认较新
        except:
            return 0.5
    
    def rerank_by_semantic_similarity(self, documents: List[Tuple[Any, float]]) -> List[Tuple[Any, float]]:
        """
        基于语义相似度重排序
        
        Args:
            documents: (文档, 分数)元组列表
            
        Returns:
            重排序后的结果
        """
        # 按语义分数排序（升序，因为分数越小越相似）
        return sorted(documents, key=lambda x: x[1])
    
    def rerank_by_keyword_overlap(self, documents: List[Tuple[Any, float]], query_keywords: set) -> List[Tuple[Any, float]]:
        """
        基于关键词重叠度重排序
        
        Args:
            documents: (文档, 分数)元组列表
            query_keywords: 查询关键词集合
            
        Returns:
            重排序后的结果
        """
        # 这里可以实现基于关键词重叠度的重排序
        # 目前返回原始顺序
        return documents 