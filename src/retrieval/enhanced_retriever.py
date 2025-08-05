"""
增强检索器
结合语义检索和关键词匹配的混合检索策略
"""

import json
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from config.settings import settings
from config.logging_config import get_logger
from ..core.vector_store import VectorStore
from ..core.llm_client import LLMClient
from .keyword_matcher import KeywordMatcher

logger = get_logger(__name__)


class EnhancedRetriever:
    """增强检索器类"""
    
    def __init__(self, vector_store: VectorStore):
        """
        初始化增强检索器
        
        Args:
            vector_store: 向量数据库实例
        """
        self.vector_store = vector_store
        self.llm_client = LLMClient()
        self.keyword_matcher = KeywordMatcher()
    
    def retrieve(self, query: str, top_k: int = None, keyword_weight: float = None) -> List[Document]:
        """
        执行增强检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            keyword_weight: 关键词权重
            
        Returns:
            检索结果文档列表
        """
        top_k = top_k or settings.default_top_k
        keyword_weight = keyword_weight or settings.keyword_weight
        
        logger.info(f"执行增强检索: query='{query[:50]}...', top_k={top_k}, keyword_weight={keyword_weight}")
        
        # 1. 语义召回
        docs_with_scores = self.vector_store.similarity_search_with_score(query, k=top_k * 2)
        
        # 2. 用LLM提取用户问题关键词
        query_keywords = self.llm_client.extract_keywords(query, top_k=3)
        logger.debug(f"查询关键词: {query_keywords}")
        
        # 3. 计算综合分数
        scored_results = []
        for doc, semantic_score in docs_with_scores:
            try:
                # 获取文档关键词
                doc_keywords = set(json.loads(doc.metadata.get('keywords', '[]')))
            except:
                doc_keywords = set()
            
            # 计算关键词重叠度
            keyword_overlap = len(query_keywords & doc_keywords)
            
            # 计算综合分数 (语义分数越小越好，关键词重叠越多越好)
            final_score = semantic_score - keyword_weight * keyword_overlap
            
            scored_results.append({
                'doc': doc,
                'semantic_score': semantic_score,
                'keyword_overlap': keyword_overlap,
                'final_score': final_score
            })
        
        # 4. 按综合分数排序
        scored_results.sort(key=lambda x: (x['final_score'], -x['keyword_overlap']))
        
        # 5. 返回前top_k个结果
        results = [item['doc'] for item in scored_results[:top_k]]
        
        logger.info(f"检索完成，返回 {len(results)} 个结果")
        return results
    
    def retrieve_with_metadata(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        带元数据的检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            包含元数据的检索结果
        """
        docs = self.retrieve(query, top_k)
        
        results = []
        for doc in docs:
            result = {
                'content': doc.page_content,
                'metadata': doc.metadata,
                'source': doc.metadata.get('source', ''),
                'summary': doc.metadata.get('summary', ''),
                'keywords': json.loads(doc.metadata.get('keywords', '[]')),
                'chapter_title': doc.metadata.get('chapter_title', ''),
                'doc_type': doc.metadata.get('doc_type', 'unknown')
            }
            results.append(result)
        
        return results
    
    def get_retrieval_context(self, query: str, top_k: int = 9) -> str:
        """
        获取检索上下文，用于问答
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            格式化的上下文字符串
        """
        docs = self.retrieve(query, top_k)
        
        context_parts = []
        for doc in docs:
            context_part = f"【章节】{doc.metadata.get('chapter_title','')}\n" \
                          f"【摘要】{doc.metadata.get('summary','')}\n" \
                          f"【关键词】{json.loads(doc.metadata.get('keywords','[]'))}\n" \
                          f"{doc.page_content}"
            context_parts.append(context_part)
        
        return "\n\n".join(context_parts) 