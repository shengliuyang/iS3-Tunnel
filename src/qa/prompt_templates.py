"""
提示模板模块
提供各种问答场景的提示模板
"""

from langchain.prompts import PromptTemplate
from config.logging_config import get_logger

logger = get_logger(__name__)


class PromptTemplates:
    """提示模板类"""
    
    @staticmethod
    def get_qa_template() -> PromptTemplate:
        """
        获取基础问答模板
        
        Returns:
            问答提示模板
        """
        template = """
你是一个专业的知识问答助手。请结合以下知识库内容回答问题，所有题目均为单选题，只有一个选项正确。如果知识库无法回答，可以用你自己的知识补充，但必须给出一个明确的选项答案。

知识库内容：{context}
问题：{question}

请直接给出选择题的标准答案（只要选项字母或数字，不要题干、解释、标点、空格）。
"""
        return PromptTemplate(
            input_variables=["context", "question"],
            template=template
        )
    
    @staticmethod
    def get_summary_template() -> PromptTemplate:
        """
        获取摘要生成模板
        
        Returns:
            摘要提示模板
        """
        template = """
请用中文对下面的文本进行高度凝练的摘要，要求：
1. 摘要应覆盖文本核心要点，长度不超过{max_length}字。
2. 只输出摘要内容，不要编号，不要解释，不要多余内容。

原始文本如下：
{text}
"""
        return PromptTemplate(
            input_variables=["text", "max_length"],
            template=template
        )
    
    @staticmethod
    def get_keyword_extraction_template() -> PromptTemplate:
        """
        获取关键词提取模板
        
        Returns:
            关键词提取提示模板
        """
        template = """
请从下面的文本中提取最能代表主题的{top_k}个关键词，要求：
1. 只返回关键词本身，不要编号，不要解释，不要分行，每个关键词用逗号分隔。
2. 关键词可以是词组或短语，尽量覆盖文本核心内容。
3. 只输出关键词列表，不要多余内容。

原始文本如下：
{text}
"""
        return PromptTemplate(
            input_variables=["text", "top_k"],
            template=template
        )
    
    @staticmethod
    def get_analysis_template() -> PromptTemplate:
        """
        获取分析模板
        
        Returns:
            分析提示模板
        """
        template = """
基于以下知识库内容，请对问题进行深入分析并给出详细答案：

知识库内容：{context}
问题：{question}

请提供：
1. 直接答案
2. 相关依据和解释
3. 可能的注意事项
"""
        return PromptTemplate(
            input_variables=["context", "question"],
            template=template
        )
    
    @staticmethod
    def get_comparison_template() -> PromptTemplate:
        """
        获取对比分析模板
        
        Returns:
            对比分析提示模板
        """
        template = """
基于知识库内容，请对以下问题进行对比分析：

知识库内容：{context}
问题：{question}

请从以下方面进行对比：
1. 相似点
2. 差异点
3. 适用场景
4. 建议选择
"""
        return PromptTemplate(
            input_variables=["context", "question"],
            template=template
        )
    
    @staticmethod
    def get_custom_template(template_text: str, variables: list) -> PromptTemplate:
        """
        创建自定义模板
        
        Args:
            template_text: 模板文本
            variables: 变量列表
            
        Returns:
            自定义提示模板
        """
        return PromptTemplate(
            input_variables=variables,
            template=template_text
        ) 