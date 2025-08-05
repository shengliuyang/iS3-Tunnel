"""
向量数据库模块
封装Chroma向量数据库操作
"""

import os
import json
import time
import shutil
from datetime import datetime
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from config.settings import settings
from config.logging_config import get_logger
from .embeddings import EmbeddingModel

logger = get_logger(__name__)


class VectorStore:
    """向量数据库操作封装类"""
    
    def __init__(self, collection_name: Optional[str] = None):
        """
        初始化向量数据库
        
        Args:
            collection_name: 集合名称
        """
        self.collection_name = collection_name or settings.collection_name
        self.persist_directory = settings.chroma_db_dir
        self.embedding_model = EmbeddingModel()
        self.vectorstore = None
        self._initialize_vectorstore()
    
    def _initialize_vectorstore(self):
        """初始化向量存储"""
        try:
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_model.get_model()
            )
            logger.info(f"初始化向量数据库: {self.collection_name}")
        except Exception as e:
            logger.error(f"向量数据库初始化失败: {e}")
            raise
    
    def clear_database(self):
        """清空向量数据库"""
        try:
            if os.path.exists(os.path.join(self.persist_directory, 'index')):
                logger.info("正在删除旧的向量数据库...")
                shutil.rmtree(os.path.join(self.persist_directory, 'index'))
            self._initialize_vectorstore()
            logger.info("向量数据库已清空")
        except Exception as e:
            logger.error(f"清空向量数据库失败: {e}")
            raise
    
    def add_documents(self, documents: List[Dict[str, Any]], batch_size: Optional[int] = None):
        """
        添加文档到向量数据库
        
        Args:
            documents: 文档列表
            batch_size: 批处理大小
        """
        batch_size = batch_size or settings.batch_size
        
        # 构造langchain文档对象
        langchain_docs = []
        for idx, doc in enumerate(tqdm(documents, desc='准备文档对象')):
            metadata = {
                'source': doc['source'],
                'doc_id': idx,
                'doc_type': doc.get('doc_type', 'unknown'),
                'filename': doc['source'],
                'filepath': doc.get('filepath', ''),
                'filesize': doc.get('filesize', 0),
                'created_time': doc.get('created_time', ''),
                'modified_time': doc.get('modified_time', ''),
                'chunk_index': doc.get('chunk_index', 0),
                'total_chunks': doc.get('total_chunks', 1),
                'chunk_size': doc.get('chunk_size', len(doc['content'])),
                'start_position': doc.get('start_position', -1),
                'end_position': doc.get('end_position', -1),
                'keywords': json.dumps(doc.get('keywords', []), ensure_ascii=False),
                'summary': doc.get('summary', ''),
                'timestamp': doc.get('timestamp', ''),
                'total_length': doc.get('total_length', 0),
                'paragraph_count': doc.get('paragraph_count', 0),
                'sentence_count': doc.get('sentence_count', 0),
                'chapter_title': doc.get('chapter_title', ''),
                'chapter_level': doc.get('chapter_level', 0)
            }
            
            langchain_docs.append(Document(
                page_content=doc['content'],
                metadata=metadata
            ))
        
        # 批量写入
        logger.info(f"开始批量写入向量库，总分块数: {len(langchain_docs)}，批量大小: {batch_size}")
        for i in tqdm(range(0, len(langchain_docs), batch_size), desc='写入向量库'):
            batch_docs = langchain_docs[i:i+batch_size]
            try:
                self.vectorstore.add_documents(batch_docs)
            except Exception as e:
                logger.error(f"批量写入出错，重试中: {e}")
                time.sleep(3)
                try:
                    self.vectorstore.add_documents(batch_docs)
                except Exception as e2:
                    logger.error(f"重试失败，跳过该批: {e2}")
            time.sleep(0.5)
        
        logger.info(f"已完成{len(langchain_docs)}个分块的向量化")
        self._save_processing_stats(documents)
    
    def _save_processing_stats(self, documents: List[Dict[str, Any]]):
        """保存处理统计信息"""
        stats = {
            'total_documents': len(set(doc['source'] for doc in documents)),
            'total_chunks': len(documents),
            'doc_types': {},
            'avg_chunk_size': sum(len(doc['content']) for doc in documents) / len(documents) if documents else 0,
            'processing_time': datetime.now().isoformat()
        }
        
        # 统计文档类型分布
        for doc in documents:
            doc_type = doc.get('doc_type', 'unknown')
            stats['doc_types'][doc_type] = stats['doc_types'].get(doc_type, 0) + 1
        
        stats_file = os.path.join(self.persist_directory, 'processing_stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"处理统计信息已保存到: {stats_file}")
    
    def similarity_search(self, query: str, k: int = 10) -> List[Document]:
        """
        相似性搜索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            相似文档列表
        """
        try:
            return self.vectorstore.similarity_search(query, k=k)
        except Exception as e:
            logger.error(f"相似性搜索失败: {e}")
            raise
    
    def similarity_search_with_score(self, query: str, k: int = 10) -> List[tuple]:
        """
        带分数的相似性搜索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            (文档, 分数)元组列表
        """
        try:
            return self.vectorstore.similarity_search_with_score(query, k=k)
        except Exception as e:
            logger.error(f"带分数的相似性搜索失败: {e}")
            raise
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        获取集合信息
        
        Returns:
            集合信息字典
        """
        try:
            collection = self.vectorstore._collection
            return {
                'name': collection.name,
                'count': collection.count(),
                'metadata': collection.metadata
            }
        except Exception as e:
            logger.error(f"获取集合信息失败: {e}")
            return {}
    
    def get_vectorstore(self) -> Chroma:
        """
        获取原始向量存储对象
        
        Returns:
            Chroma实例
        """
        return self.vectorstore 