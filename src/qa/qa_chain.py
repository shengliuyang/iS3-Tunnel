"""
问答链模块
构建RAG问答流程
"""

from typing import Dict, Any, Optional
from langchain.chains import RetrievalQA
from config.logging_config import get_logger
from ..core.llm_client import LLMClient
from ..retrieval.enhanced_retriever import EnhancedRetriever
from .prompt_templates import PromptTemplates

logger = get_logger(__name__)


class QAChain:
    """问答链类"""
    
    def __init__(self, retriever: EnhancedRetriever, template_type: str = "qa"):
        """
        初始化问答链
        
        Args:
            retriever: 增强检索器
            template_type: 模板类型
        """
        self.retriever = retriever
        self.llm_client = LLMClient()
        self.template_type = template_type
        self.prompt_template = self._get_prompt_template(template_type)
    
    def _get_prompt_template(self, template_type: str):
        """
        获取提示模板
        
        Args:
            template_type: 模板类型
            
        Returns:
            提示模板
        """
        if template_type == "qa":
            return PromptTemplates.get_qa_template()
        elif template_type == "analysis":
            return PromptTemplates.get_analysis_template()
        elif template_type == "comparison":
            return PromptTemplates.get_comparison_template()
        else:
            return PromptTemplates.get_qa_template()
    
    def answer_question(self, question: str, top_k: int = 9) -> str:
        """
        回答问题
        
        Args:
            question: 问题
            top_k: 检索文档数量
            
        Returns:
            答案
        """
        try:
            # 获取检索上下文
            context = self.retriever.get_retrieval_context(question, top_k)
            
            # 构建提示
            prompt_input = {
                "context": context,
                "question": question
            }
            
            # 调用LLM
            formatted_prompt = self.prompt_template.format_prompt(**prompt_input).to_string()
            response = self.llm_client.get_chat_model().invoke(formatted_prompt)
            
            # 提取答案
            if hasattr(response, 'content'):
                answer = response.content
            else:
                answer = str(response)
            
            # 清理答案
            answer = self._clean_answer(answer)
            
            logger.info(f"问题: {question[:50]}... -> 答案: {answer}")
            return answer
            
        except Exception as e:
            logger.error(f"回答问题失败: {e}")
            return f"[ERROR] {str(e)}"
    
    def answer_question_with_metadata(self, question: str, top_k: int = 9) -> Dict[str, Any]:
        """
        回答问题并返回元数据
        
        Args:
            question: 问题
            top_k: 检索文档数量
            
        Returns:
            包含答案和元数据的字典
        """
        try:
            # 获取检索结果
            retrieval_results = self.retriever.retrieve_with_metadata(question, top_k)
            
            # 构建上下文
            context_parts = []
            for result in retrieval_results:
                context_part = f"【章节】{result['chapter_title']}\n" \
                              f"【摘要】{result['summary']}\n" \
                              f"【关键词】{result['keywords']}\n" \
                              f"{result['content']}"
                context_parts.append(context_part)
            
            context = "\n\n".join(context_parts)
            
            # 构建提示
            prompt_input = {
                "context": context,
                "question": question
            }
            
            # 调用LLM
            formatted_prompt = self.prompt_template.format_prompt(**prompt_input).to_string()
            response = self.llm_client.get_chat_model().invoke(formatted_prompt)
            
            # 提取答案
            if hasattr(response, 'content'):
                answer = response.content
            else:
                answer = str(response)
            
            answer = self._clean_answer(answer)
            
            return {
                'answer': answer,
                'question': question,
                'retrieval_results': retrieval_results,
                'context': context,
                'template_type': self.template_type
            }
            
        except Exception as e:
            logger.error(f"回答问题失败: {e}")
            return {
                'answer': f"[ERROR] {str(e)}",
                'question': question,
                'retrieval_results': [],
                'context': '',
                'template_type': self.template_type
            }
    
    def _clean_answer(self, answer: str) -> str:
        """
        清理答案
        
        Args:
            answer: 原始答案
            
        Returns:
            清理后的答案
        """
        # 移除多余的空格和换行
        answer = answer.strip()
        
        # 如果是选择题答案，只保留选项部分
        if self.template_type == "qa":
            # 提取第一个非空字符作为选项
            words = answer.split()
            if words:
                answer = words[0]
        
        return answer
    
    def batch_answer_questions(self, questions: list, top_k: int = 9) -> list:
        """
        批量回答问题
        
        Args:
            questions: 问题列表
            top_k: 检索文档数量
            
        Returns:
            答案列表
        """
        results = []
        for question in questions:
            answer = self.answer_question(question, top_k)
            results.append({
                'question': question,
                'answer': answer
            })
        
        return results
    
    def set_template_type(self, template_type: str):
        """
        设置模板类型
        
        Args:
            template_type: 模板类型
        """
        self.template_type = template_type
        self.prompt_template = self._get_prompt_template(template_type)
        logger.info(f"设置模板类型: {template_type}") 