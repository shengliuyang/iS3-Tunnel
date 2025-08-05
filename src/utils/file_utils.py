"""
文件工具模块
提供文件操作和格式转换功能
"""

import os
import json
import shutil
from datetime import datetime
from typing import List, Dict, Any, Optional
from config.logging_config import get_logger

logger = get_logger(__name__)


class FileUtils:
    """文件工具类"""
    
    @staticmethod
    def ensure_directory(directory_path: str) -> bool:
        """
        确保目录存在
        
        Args:
            directory_path: 目录路径
            
        Returns:
            是否成功创建
        """
        try:
            os.makedirs(directory_path, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"创建目录失败 {directory_path}: {e}")
            return False
    
    @staticmethod
    def save_json(data: Any, file_path: str, ensure_ascii: bool = False) -> bool:
        """
        保存JSON文件
        
        Args:
            data: 要保存的数据
            file_path: 文件路径
            ensure_ascii: 是否确保ASCII编码
            
        Returns:
            是否保存成功
        """
        try:
            # 确保目录存在
            directory = os.path.dirname(file_path)
            if directory:
                FileUtils.ensure_directory(directory)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=ensure_ascii, indent=2)
            
            logger.info(f"JSON文件保存成功: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"保存JSON文件失败 {file_path}: {e}")
            return False
    
    @staticmethod
    def load_json(file_path: str) -> Optional[Any]:
        """
        加载JSON文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            加载的数据，失败返回None
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"JSON文件加载成功: {file_path}")
            return data
            
        except Exception as e:
            logger.error(f"加载JSON文件失败 {file_path}: {e}")
            return None
    
    @staticmethod
    def save_text(text: str, file_path: str) -> bool:
        """
        保存文本文件
        
        Args:
            text: 文本内容
            file_path: 文件路径
            
        Returns:
            是否保存成功
        """
        try:
            # 确保目录存在
            directory = os.path.dirname(file_path)
            if directory:
                FileUtils.ensure_directory(directory)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(text)
            
            logger.info(f"文本文件保存成功: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"保存文本文件失败 {file_path}: {e}")
            return False
    
    @staticmethod
    def load_text(file_path: str) -> Optional[str]:
        """
        加载文本文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            文本内容，失败返回None
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            logger.info(f"文本文件加载成功: {file_path}")
            return text
            
        except Exception as e:
            logger.error(f"加载文本文件失败 {file_path}: {e}")
            return None
    
    @staticmethod
    def get_file_info(file_path: str) -> Optional[Dict[str, Any]]:
        """
        获取文件信息
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件信息字典
        """
        try:
            if not os.path.exists(file_path):
                return None
            
            stat = os.stat(file_path)
            return {
                'filename': os.path.basename(file_path),
                'filepath': file_path,
                'filesize': stat.st_size,
                'created_time': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                'modified_time': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'extension': os.path.splitext(file_path)[1].lower()
            }
            
        except Exception as e:
            logger.error(f"获取文件信息失败 {file_path}: {e}")
            return None
    
    @staticmethod
    def list_files(directory: str, extensions: Optional[List[str]] = None) -> List[str]:
        """
        列出目录中的文件
        
        Args:
            directory: 目录路径
            extensions: 文件扩展名过滤列表
            
        Returns:
            文件路径列表
        """
        try:
            if not os.path.exists(directory):
                return []
            
            files = []
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path):
                    if extensions is None:
                        files.append(file_path)
                    else:
                        ext = os.path.splitext(filename)[1].lower()
                        if ext in extensions:
                            files.append(file_path)
            
            return files
            
        except Exception as e:
            logger.error(f"列出文件失败 {directory}: {e}")
            return []
    
    @staticmethod
    def backup_file(file_path: str, backup_suffix: str = ".backup") -> bool:
        """
        备份文件
        
        Args:
            file_path: 文件路径
            backup_suffix: 备份后缀
            
        Returns:
            是否备份成功
        """
        try:
            if not os.path.exists(file_path):
                return False
            
            backup_path = file_path + backup_suffix
            shutil.copy2(file_path, backup_path)
            
            logger.info(f"文件备份成功: {file_path} -> {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"文件备份失败 {file_path}: {e}")
            return False
    
    @staticmethod
    def clean_directory(directory: str, keep_extensions: Optional[List[str]] = None) -> bool:
        """
        清理目录
        
        Args:
            directory: 目录路径
            keep_extensions: 要保留的文件扩展名列表
            
        Returns:
            是否清理成功
        """
        try:
            if not os.path.exists(directory):
                return True
            
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path):
                    if keep_extensions is None:
                        os.remove(file_path)
                    else:
                        ext = os.path.splitext(filename)[1].lower()
                        if ext not in keep_extensions:
                            os.remove(file_path)
            
            logger.info(f"目录清理完成: {directory}")
            return True
            
        except Exception as e:
            logger.error(f"目录清理失败 {directory}: {e}")
            return False 