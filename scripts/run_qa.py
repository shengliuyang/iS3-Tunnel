#!/usr/bin/env python3
"""
运行问答系统脚本
用于交互式问答和批量问答
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
from src.core.vector_store import VectorStore
from src.retrieval.enhanced_retriever import EnhancedRetriever
from src.qa.qa_chain import QAChain
from src.utils.file_utils import FileUtils

logger = setup_logging()


def interactive_qa(collection_name: str = None, template_type: str = "qa"):
    """
    交互式问答
    
    Args:
        collection_name: 集合名称
        template_type: 模板类型
    """
    logger.info("启动交互式问答系统")
    
    try:
        # 初始化组件
        vector_store = VectorStore(collection_name)
        retriever = EnhancedRetriever(vector_store)
        qa_chain = QAChain(retriever, template_type)
        
        print("=" * 50)
        print("iS3 Tunnel Enhanced RAG问答系统")
        print("输入 'quit' 或 'exit' 退出")
        print("输入 'help' 查看帮助")
        print("=" * 50)
        
        while True:
            try:
                question = input("\n请输入问题: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("再见！")
                    break
                
                if question.lower() == 'help':
                    print_help()
                    continue
                
                if not question:
                    continue
                
                # 回答问题
                print("正在思考...")
                answer = qa_chain.answer_question(question)
                print(f"答案: {answer}")
                
            except KeyboardInterrupt:
                print("\n再见！")
                break
            except Exception as e:
                logger.error(f"问答过程中出错: {e}")
                print(f"抱歉，处理问题时出现错误: {e}")
                
    except Exception as e:
        logger.error(f"初始化问答系统失败: {e}")
        print(f"初始化失败: {e}")


def batch_qa(question_file: str, 
             output_file: str, 
             collection_name: str = None, 
             template_type: str = "qa"):
    """
    批量问答
    
    Args:
        question_file: 问题文件路径
        output_file: 输出文件路径
        collection_name: 集合名称
        template_type: 模板类型
    """
    logger.info("开始批量问答")
    
    try:
        # 检查问题文件
        if not os.path.exists(question_file):
            logger.error(f"问题文件不存在: {question_file}")
            return False
        
        # 读取问题
        questions = []
        with open(question_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    questions.append(line)
        
        if not questions:
            logger.warning("问题文件为空")
            return False
        
        logger.info(f"读取到 {len(questions)} 个问题")
        
        # 初始化组件
        vector_store = VectorStore(collection_name)
        retriever = EnhancedRetriever(vector_store)
        qa_chain = QAChain(retriever, template_type)
        
        # 批量问答
        results = qa_chain.batch_answer_questions(questions)
        
        # 保存结果
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(f"{result['question']} 答案：{result['answer']}\n")
        
        logger.info(f"批量问答完成，结果已保存到: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"批量问答失败: {e}")
        return False


def single_qa(question: str, 
              collection_name: str = None, 
              template_type: str = "qa",
              with_metadata: bool = False):
    """
    单个问题问答
    
    Args:
        question: 问题
        collection_name: 集合名称
        template_type: 模板类型
        with_metadata: 是否返回元数据
    """
    logger.info(f"处理单个问题: {question}")
    
    try:
        # 初始化组件
        vector_store = VectorStore(collection_name)
        retriever = EnhancedRetriever(vector_store)
        qa_chain = QAChain(retriever, template_type)
        
        if with_metadata:
            result = qa_chain.answer_question_with_metadata(question)
            print(f"问题: {result['question']}")
            print(f"答案: {result['answer']}")
            print(f"检索结果数量: {len(result['retrieval_results'])}")
            print(f"模板类型: {result['template_type']}")
        else:
            answer = qa_chain.answer_question(question)
            print(f"问题: {question}")
            print(f"答案: {answer}")
        
        return True
        
    except Exception as e:
        logger.error(f"单个问答失败: {e}")
        print(f"处理失败: {e}")
        return False


def print_help():
    """打印帮助信息"""
    help_text = """
可用命令:
- help: 显示此帮助信息
- quit/exit/q: 退出系统

支持的模板类型:
- qa: 基础问答（默认）
- analysis: 分析模式
- comparison: 对比模式
"""
    print(help_text)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="iS3 Tunnel Enhanced RAG问答系统")
    parser.add_argument("--mode", choices=["interactive", "batch", "single"], 
                       default="interactive", help="运行模式")
    parser.add_argument("--question", help="单个问题（single模式）")
    parser.add_argument("--question-file", help="问题文件路径（batch模式）")
    parser.add_argument("--output-file", help="输出文件路径（batch模式）")
    parser.add_argument("--collection", help="集合名称")
    parser.add_argument("--template", choices=["qa", "analysis", "comparison"], 
                       default="qa", help="模板类型")
    parser.add_argument("--with-metadata", action="store_true", 
                       help="返回元数据（single模式）")
    
    args = parser.parse_args()
    
    if args.mode == "interactive":
        interactive_qa(args.collection, args.template)
    elif args.mode == "batch":
        if not args.question_file or not args.output_file:
            print("batch模式需要指定 --question-file 和 --output-file")
            sys.exit(1)
        success = batch_qa(args.question_file, args.output_file, 
                          args.collection, args.template)
        sys.exit(0 if success else 1)
    elif args.mode == "single":
        if not args.question:
            print("single模式需要指定 --question")
            sys.exit(1)
        success = single_qa(args.question, args.collection, 
                           args.template, args.with_metadata)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 