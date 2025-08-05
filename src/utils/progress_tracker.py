"""
进度跟踪器模块
提供多级进度条管理功能
"""

from tqdm import tqdm
from typing import Optional
from config.logging_config import get_logger

logger = get_logger(__name__)


class ProgressTracker:
    """进度跟踪器类"""
    
    def __init__(self):
        """初始化进度跟踪器"""
        self.progress_bars = {}
        self.current_level = 0
    
    def create_progress_bar(self, 
                          total: int, 
                          desc: str, 
                          level: int = 0,
                          position: Optional[int] = None) -> tqdm:
        """
        创建进度条
        
        Args:
            total: 总数
            desc: 描述
            level: 层级
            position: 位置
            
        Returns:
            进度条对象
        """
        try:
            if position is None:
                position = level
            
            pbar = tqdm(
                total=total,
                desc=desc,
                position=position,
                leave=True
            )
            
            self.progress_bars[level] = pbar
            self.current_level = max(self.current_level, level)
            
            logger.info(f"创建进度条: {desc} (总数: {total}, 层级: {level})")
            return pbar
            
        except Exception as e:
            logger.error(f"创建进度条失败: {e}")
            return None
    
    def update_progress(self, level: int, increment: int = 1):
        """
        更新进度
        
        Args:
            level: 层级
            increment: 增量
        """
        try:
            if level in self.progress_bars:
                self.progress_bars[level].update(increment)
        except Exception as e:
            logger.error(f"更新进度失败 (层级 {level}): {e}")
    
    def set_progress(self, level: int, value: int):
        """
        设置进度值
        
        Args:
            level: 层级
            value: 进度值
        """
        try:
            if level in self.progress_bars:
                self.progress_bars[level].n = value
                self.progress_bars[level].refresh()
        except Exception as e:
            logger.error(f"设置进度失败 (层级 {level}): {e}")
    
    def close_progress_bar(self, level: int):
        """
        关闭进度条
        
        Args:
            level: 层级
        """
        try:
            if level in self.progress_bars:
                self.progress_bars[level].close()
                del self.progress_bars[level]
                logger.info(f"关闭进度条 (层级: {level})")
        except Exception as e:
            logger.error(f"关闭进度条失败 (层级 {level}): {e}")
    
    def close_all_progress_bars(self):
        """关闭所有进度条"""
        try:
            for level in list(self.progress_bars.keys()):
                self.close_progress_bar(level)
            logger.info("关闭所有进度条")
        except Exception as e:
            logger.error(f"关闭所有进度条失败: {e}")
    
    def get_progress_info(self, level: int) -> Optional[dict]:
        """
        获取进度信息
        
        Args:
            level: 层级
            
        Returns:
            进度信息字典
        """
        try:
            if level in self.progress_bars:
                pbar = self.progress_bars[level]
                return {
                    'current': pbar.n,
                    'total': pbar.total,
                    'percentage': (pbar.n / pbar.total * 100) if pbar.total > 0 else 0,
                    'desc': pbar.desc
                }
            return None
        except Exception as e:
            logger.error(f"获取进度信息失败 (层级 {level}): {e}")
            return None
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close_all_progress_bars()


# 全局进度跟踪器实例
global_progress_tracker = ProgressTracker()


def get_global_progress_tracker() -> ProgressTracker:
    """
    获取全局进度跟踪器
    
    Returns:
        全局进度跟踪器实例
    """
    return global_progress_tracker 