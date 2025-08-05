"""
错误处理模块
提供统一的异常处理和错误恢复功能
"""

import traceback
import sys
from typing import Callable, Any, Optional
from functools import wraps
from config.logging_config import get_logger

logger = get_logger(__name__)


class ErrorHandler:
    """错误处理类"""
    
    @staticmethod
    def handle_exception(func: Callable) -> Callable:
        """
        异常处理装饰器
        
        Args:
            func: 要装饰的函数
            
        Returns:
            装饰后的函数
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"函数 {func.__name__} 执行失败: {e}")
                logger.error(f"错误详情: {traceback.format_exc()}")
                raise
        return wrapper
    
    @staticmethod
    def safe_execute(func: Callable, *args, default_return: Any = None, **kwargs) -> Any:
        """
        安全执行函数
        
        Args:
            func: 要执行的函数
            *args: 位置参数
            default_return: 默认返回值
            **kwargs: 关键字参数
            
        Returns:
            函数返回值或默认值
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"安全执行函数 {func.__name__} 失败: {e}")
            logger.error(f"错误详情: {traceback.format_exc()}")
            return default_return
    
    @staticmethod
    def retry_on_failure(max_attempts: int = 3, delay: float = 1.0):
        """
        失败重试装饰器
        
        Args:
            max_attempts: 最大尝试次数
            delay: 重试延迟（秒）
            
        Returns:
            装饰器函数
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                last_exception = None
                
                for attempt in range(max_attempts):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        logger.warning(f"函数 {func.__name__} 第 {attempt + 1} 次尝试失败: {e}")
                        
                        if attempt < max_attempts - 1:
                            import time
                            time.sleep(delay)
                
                logger.error(f"函数 {func.__name__} 在 {max_attempts} 次尝试后仍然失败")
                raise last_exception
                
            return wrapper
        return decorator
    
    @staticmethod
    def log_error(error: Exception, context: str = ""):
        """
        记录错误信息
        
        Args:
            error: 异常对象
            context: 错误上下文
        """
        error_msg = f"错误: {error}"
        if context:
            error_msg = f"{context} - {error_msg}"
        
        logger.error(error_msg)
        logger.error(f"错误类型: {type(error).__name__}")
        logger.error(f"错误详情: {traceback.format_exc()}")
    
    @staticmethod
    def format_error_message(error: Exception, include_traceback: bool = True) -> str:
        """
        格式化错误信息
        
        Args:
            error: 异常对象
            include_traceback: 是否包含堆栈跟踪
            
        Returns:
            格式化的错误信息
        """
        error_info = f"错误类型: {type(error).__name__}\n"
        error_info += f"错误信息: {str(error)}\n"
        
        if include_traceback:
            error_info += f"堆栈跟踪:\n{traceback.format_exc()}"
        
        return error_info
    
    @staticmethod
    def is_critical_error(error: Exception) -> bool:
        """
        判断是否为严重错误
        
        Args:
            error: 异常对象
            
        Returns:
            是否为严重错误
        """
        critical_error_types = [
            KeyboardInterrupt,
            SystemExit,
            MemoryError,
            OSError,
            ImportError
        ]
        
        return any(isinstance(error, error_type) for error_type in critical_error_types)
    
    @staticmethod
    def handle_critical_error(error: Exception):
        """
        处理严重错误
        
        Args:
            error: 异常对象
        """
        logger.critical(f"检测到严重错误: {error}")
        logger.critical(f"错误详情: {traceback.format_exc()}")
        
        # 可以在这里添加清理代码
        # 例如：关闭数据库连接、保存临时文件等
        
        logger.critical("系统即将退出...")
        sys.exit(1)


def safe_function_call(func: Callable, *args, **kwargs) -> Optional[Any]:
    """
    安全函数调用
    
    Args:
        func: 要调用的函数
        *args: 位置参数
        **kwargs: 关键字参数
        
    Returns:
        函数返回值或None
    """
    return ErrorHandler.safe_execute(func, *args, default_return=None, **kwargs)


def retry_function(func: Callable, max_attempts: int = 3, delay: float = 1.0):
    """
    重试函数装饰器
    
    Args:
        func: 要装饰的函数
        max_attempts: 最大尝试次数
        delay: 重试延迟（秒）
        
    Returns:
        装饰后的函数
    """
    return ErrorHandler.retry_on_failure(max_attempts, delay)(func) 