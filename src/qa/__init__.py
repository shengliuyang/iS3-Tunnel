"""
问答模块
提供RAG问答链和提示模板功能
"""

from .qa_chain import QAChain
from .prompt_templates import PromptTemplates

__all__ = ['QAChain', 'PromptTemplates'] 