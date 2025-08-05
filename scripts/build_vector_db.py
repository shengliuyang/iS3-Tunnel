#!/usr/bin/env python3
"""
构建向量数据库脚本
用于从文档文件夹构建向量数据库
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from config.logging_config import setup_logging
from src.processors.document_loader import DocumentLoader
from src.processors.text_processor import EnhancedTextProcessor
from src.processors.metadata_extractor import MetadataExtractor
from src.core.vector_store import VectorStore
from src.utils.file_utils import FileUtils
from src.utils.progress_tracker import get_global_progress_tracker

logger = setup_logging()


def build_vector_database(input_folder: str, 
                         collection_name: str = None,
                         clear_existing: bool = False,
                         min_chunk_size: int = None,
                         max_chunk_size: int = None,
                         chunk_overlap: int = None):
    """
    构建向量数据库
    
    Args:
        input_folder: 输入文档文件夹
        collection_name: 集合名称
        clear_existing: 是否清空现有数据库
        min_chunk_size: 最小分块大小
        max_chunk_size: 最大分块大小
        chunk_overlap: 分块重叠大小
    """
    logger.info("开始构建向量数据库")
    
    # 检查输入文件夹
    if not os.path.isdir(input_folder):
        logger.error(f"输入文件夹不存在: {input_folder}")
        return False
    
    try:
        # 初始化组件
        document_loader = DocumentLoader()
        text_processor = EnhancedTextProcessor(
            min_chunk_size=min_chunk_size or settings.min_chunk_size,
            max_chunk_size=max_chunk_size or settings.max_chunk_size,
            chunk_overlap=chunk_overlap or settings.chunk_overlap
        )
        metadata_extractor = MetadataExtractor()
        vector_store = VectorStore(collection_name)
        
        # 清空现有数据库
        if clear_existing:
            logger.info("清空现有向量数据库")
            vector_store.clear_database()
        
        # 加载文档
        logger.info(f"从文件夹加载文档: {input_folder}")
        documents = document_loader.load_documents_from_folder(input_folder)
        
        if not documents:
            logger.warning("未找到任何文档")
            return False
        
        # 处理文档
        all_chunks = []
        with get_global_progress_tracker() as tracker:
            for doc in documents:
                logger.info(f"处理文档: {doc['file_info']['filename']}")
                
                # 提取文档结构
                doc_structure = text_processor.extract_document_structure(
                    doc['content'], 
                    doc['file_info']['filename']
                )
                
                # 语义分割
                chunks = text_processor.split_text_semantically(doc['content'], doc_structure)
                
                # 添加文件信息到分块
                for chunk in chunks:
                    chunk.update({
                        'source': doc['file_info']['filename'],
                        'filepath': doc['file_info']['filepath'],
                        'filesize': doc['file_info']['filesize'],
                        'created_time': doc['file_info']['created_time'],
                        'modified_time': doc['file_info']['modified_time'],
                        'doc_type': doc_structure['doc_type'],
                        'timestamp': doc_structure['timestamp'],
                        'total_length': doc_structure['total_length'],
                        'paragraph_count': doc_structure['paragraph_count'],
                        'sentence_count': doc_structure['sentence_count']
                    })
                
                all_chunks.extend(chunks)
        
        # 提取元数据
        logger.info("提取文档元数据")
        all_chunks = metadata_extractor.extract_metadata_for_chunks(all_chunks)
        
        # 构建向量数据库
        logger.info("构建向量数据库")
        vector_store.add_documents(all_chunks)
        
        logger.info(f"向量数据库构建完成，共处理 {len(documents)} 个文档，生成 {len(all_chunks)} 个分块")
        return True
        
    except Exception as e:
        logger.error(f"构建向量数据库失败: {e}")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="iS3 Tunnel Enhanced RAG - 构建向量数据库")
    parser.add_argument("input_folder", help="输入文档文件夹路径")
    parser.add_argument("--collection", help="集合名称")
    parser.add_argument("--clear", action="store_true", help="清空现有数据库")
    parser.add_argument("--min-chunk-size", type=int, help="最小分块大小")
    parser.add_argument("--max-chunk-size", type=int, help="最大分块大小")
    parser.add_argument("--chunk-overlap", type=int, help="分块重叠大小")
    
    args = parser.parse_args()
    
    success = build_vector_database(
        input_folder=args.input_folder,
        collection_name=args.collection,
        clear_existing=args.clear,
        min_chunk_size=args.min_chunk_size,
        max_chunk_size=args.max_chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    if success:
        logger.info("向量数据库构建成功")
        sys.exit(0)
    else:
        logger.error("向量数据库构建失败")
        sys.exit(1)


if __name__ == "__main__":
    main() 