"""
日志配置模块
提供统一的日志配置和格式化
"""

import logging
import os
from typing import Optional
from .settings import settings


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    disable_existing_loggers: bool = True
) -> logging.Logger:
    """
    设置日志配置
    
    Args:
        log_level: 日志级别
        log_file: 日志文件路径
        disable_existing_loggers: 是否禁用现有日志器
    
    Returns:
        配置好的根日志器
    """
    # 使用配置或参数
    level = log_level or settings.log_level
    file_path = log_file or settings.log_file
    
    # 确保日志目录存在
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # 配置根日志器
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(file_path, encoding='utf-8'),
            logging.StreamHandler()
        ],
        disable_existing_loggers=disable_existing_loggers
    )
    
    # 设置特定模块的日志级别
    logging.getLogger("chromadb").setLevel(logging.ERROR)
    logging.getLogger("langchain").setLevel(logging.ERROR)
    logging.getLogger("langchain_community").setLevel(logging.ERROR)
    logging.getLogger("langchain_ollama").setLevel(logging.ERROR)
    
    return logging.getLogger()


def get_logger(name: str) -> logging.Logger:
    """
    获取指定名称的日志器
    
    Args:
        name: 日志器名称
    
    Returns:
        日志器实例
    """
    return logging.getLogger(name) 