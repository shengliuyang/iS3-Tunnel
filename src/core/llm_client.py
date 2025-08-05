"""
LLM客户端模块
封装Azure OpenAI调用，提供重试机制和错误处理
"""

import time
import logging
from typing import List, Dict, Any, Optional
from openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, reraise
from config.settings import settings
from config.logging_config import get_logger

logger = get_logger(__name__)


class LLMClient:
    """LLM客户端封装类"""
    
    def __init__(self):
        """初始化LLM客户端"""
        self.client = AzureOpenAI(
            api_key=settings.azure_openai_key,
            api_version=settings.azure_openai_api_version,
            azure_endpoint=settings.azure_openai_endpoint
        )
        
        self.chat_model = AzureChatOpenAI(
            openai_api_key=settings.azure_openai_key,
            azure_endpoint=settings.azure_openai_endpoint,
            deployment_name=settings.azure_openai_deployment,
            openai_api_version=settings.azure_openai_api_version,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
        
        self.deployment_name = settings.azure_openai_deployment
    
    @retry(
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        reraise=True
    )
    def extract_keywords(self, text: str, top_k: int = 6) -> List[str]:
        """
        使用LLM提取关键词
        
        Args:
            text: 输入文本
            top_k: 关键词数量
            
        Returns:
            关键词列表
        """
        text = text[:1000]  # 截断输入，减少token消耗
        prompt = f"""
请从下面的文本中提取最能代表主题的{top_k}个关键词，要求：
1. 只返回关键词本身，不要编号，不要解释，不要分行，每个关键词用逗号分隔。
2. 关键词可以是词组或短语，尽量覆盖文本核心内容。
3. 只输出关键词列表，不要多余内容。

原始文本如下：
{text}
"""
        
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[
                        {"role": "system", "content": "你是一个专业的中文关键词提取助手。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    top_p=0.9,
                    max_tokens=800,
                    timeout=settings.llm_timeout
                )
                result = response.choices[0].message.content.strip()
                keywords = [kw.strip() for kw in result.replace('\n', '').replace('，', ',').split(',') if kw.strip()]
                time.sleep(5)  # 避免频率限制
                return keywords[:top_k]
                
            except Exception as e:
                if '429' in str(e):
                    logger.warning('检测到429限流，自动暂停65秒...')
                    time.sleep(65)
                    continue
                else:
                    logger.error(f"关键词提取失败: {e}")
                    raise
    
    @retry(
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        reraise=True
    )
    def generate_summary(self, text: str, max_length: int = 200) -> str:
        """
        使用LLM生成摘要
        
        Args:
            text: 输入文本
            max_length: 摘要最大长度
            
        Returns:
            生成的摘要
        """
        text = text[:1000]  # 截断输入，减少token消耗
        prompt = f"""
请用中文对下面的文本进行高度凝练的摘要，要求：
1. 摘要应覆盖文本核心要点，长度不超过{max_length}字。
2. 只输出摘要内容，不要编号，不要解释，不要多余内容。

原始文本如下：
{text}
"""
        
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[
                        {"role": "system", "content": "你是一个专业的中文文本摘要助手。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    top_p=0.9,
                    max_tokens=800,
                    timeout=settings.llm_timeout
                )
                summary = response.choices[0].message.content.strip()
                time.sleep(5)  # 避免频率限制
                return summary
                
            except Exception as e:
                if '429' in str(e):
                    logger.warning('检测到429限流，自动暂停65秒...')
                    time.sleep(65)
                    continue
                else:
                    logger.error(f"摘要生成失败: {e}")
                    raise
    
    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        聊天补全
        
        Args:
            messages: 消息列表
            **kwargs: 其他参数
            
        Returns:
            回复内容
        """
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"聊天补全失败: {e}")
            raise
    
    def get_chat_model(self) -> AzureChatOpenAI:
        """
        获取LangChain聊天模型
        
        Returns:
            LangChain聊天模型实例
        """
        return self.chat_model 