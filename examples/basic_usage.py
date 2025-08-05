#!/usr/bin/env python3
"""
基础使用示例
演示如何使用RAG系统进行基本的问答操作
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from config.logging_config import setup_logging
from src.core.vector_store import VectorStore
from src.retrieval.enhanced_retriever import EnhancedRetriever
from src.qa.qa_chain import QAChain

logger = setup_logging()


def basic_qa_example():
    """基础问答示例"""
    print("=" * 50)
    print("iS3 Tunnel Enhanced RAG基础问答示例")
    print("=" * 50)
    
    try:
        # 1. 初始化组件
        print("1. 初始化组件...")
        vector_store = VectorStore()
        retriever = EnhancedRetriever(vector_store)
        qa_chain = QAChain(retriever)
        
        # 2. 简单问答
        print("\n2. 进行问答...")
        question = "什么是RAG系统？"
        answer = qa_chain.answer_question(question)
        
        print(f"问题: {question}")
        print(f"答案: {answer}")
        
        # 3. 带元数据的问答
        print("\n3. 带元数据的问答...")
        result = qa_chain.answer_question_with_metadata(question)
        
        print(f"问题: {result['question']}")
        print(f"答案: {result['answer']}")
        print(f"检索结果数量: {len(result['retrieval_results'])}")
        print(f"模板类型: {result['template_type']}")
        
        print("\n示例完成！")
        
    except Exception as e:
        logger.error(f"示例执行失败: {e}")
        print(f"示例执行失败: {e}")


def retrieval_example():
    """检索示例"""
    print("=" * 50)
    print("iS3 Tunnel Enhanced RAG检索示例")
    print("=" * 50)
    
    try:
        # 初始化检索器
        vector_store = VectorStore()
        retriever = EnhancedRetriever(vector_store)
        
        # 执行检索
        query = "机器学习"
        results = retriever.retrieve_with_metadata(query, top_k=3)
        
        print(f"查询: {query}")
        print(f"检索到 {len(results)} 个结果:")
        
        for i, result in enumerate(results, 1):
            print(f"\n结果 {i}:")
            print(f"  来源: {result['source']}")
            print(f"  摘要: {result['summary']}")
            print(f"  关键词: {result['keywords']}")
            print(f"  内容片段: {result['content'][:100]}...")
        
        print("\n检索示例完成！")
        
    except Exception as e:
        logger.error(f"检索示例失败: {e}")
        print(f"检索示例失败: {e}")


def template_example():
    """模板使用示例"""
    print("=" * 50)
    print("iS3 Tunnel Enhanced RAG模板使用示例")
    print("=" * 50)
    
    try:
        # 初始化组件
        vector_store = VectorStore()
        retriever = EnhancedRetriever(vector_store)
        
        # 使用不同模板
        templates = ["qa", "analysis", "comparison"]
        question = "深度学习与传统机器学习的区别是什么？"
        
        for template_type in templates:
            print(f"\n使用 {template_type} 模板:")
            qa_chain = QAChain(retriever, template_type)
            answer = qa_chain.answer_question(question)
            print(f"答案: {answer}")
        
        print("\n模板示例完成！")
        
    except Exception as e:
        logger.error(f"模板示例失败: {e}")
        print(f"模板示例失败: {e}")


def main():
    """主函数"""
    print("iS3 Tunnel Enhanced RAG系统基础使用示例")
    print("请确保已构建向量数据库")
    
    # 运行示例
    basic_qa_example()
    print("\n" + "=" * 50 + "\n")
    
    retrieval_example()
    print("\n" + "=" * 50 + "\n")
    
    template_example()


if __name__ == "__main__":
    main() 