"""
文档加载器
支持多种格式文档的加载和预处理
"""

import os
import glob
from datetime import datetime
from typing import List, Dict, Any
from tqdm import tqdm
import pdfplumber
from config.logging_config import get_logger

logger = get_logger(__name__)


class DocumentLoader:
    """文档加载器类"""
    
    def __init__(self):
        """初始化文档加载器"""
        self.supported_formats = ['.txt', '.pdf', '.md']
    
    def load_documents_from_folder(self, folder_path: str) -> List[Dict[str, Any]]:
        """
        从文件夹加载所有支持的文档
        
        Args:
            folder_path: 文件夹路径
            
        Returns:
            文档列表
        """
        logger.info(f"开始从文件夹加载文档: {folder_path}")
        
        if not os.path.isdir(folder_path):
            raise ValueError(f"文件夹不存在: {folder_path}")
        
        documents = []
        
        # 加载TXT文件
        txt_docs = self._load_txt_files(folder_path)
        documents.extend(txt_docs)
        
        # 加载PDF文件
        pdf_docs = self._load_pdf_files(folder_path)
        documents.extend(pdf_docs)
        
        # 加载Markdown文件
        md_docs = self._load_markdown_files(folder_path)
        documents.extend(md_docs)
        
        logger.info(f"总共加载了 {len(documents)} 个文档")
        return documents
    
    def _load_txt_files(self, folder_path: str) -> List[Dict[str, Any]]:
        """
        加载TXT文件
        
        Args:
            folder_path: 文件夹路径
            
        Returns:
            TXT文档列表
        """
        txt_files = glob.glob(os.path.join(folder_path, '*.txt'))
        logger.info(f"检测到 {len(txt_files)} 个TXT文件")
        
        documents = []
        for file_path in tqdm(txt_files, desc='加载TXT文件'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                file_info = self._get_file_info(file_path)
                documents.append({
                    'content': text,
                    'file_info': file_info,
                    'format': 'txt'
                })
                
            except Exception as e:
                logger.error(f"加载TXT文件失败 {file_path}: {e}")
                continue
        
        return documents
    
    def _load_pdf_files(self, folder_path: str) -> List[Dict[str, Any]]:
        """
        加载PDF文件
        
        Args:
            folder_path: 文件夹路径
            
        Returns:
            PDF文档列表
        """
        pdf_files = glob.glob(os.path.join(folder_path, '*.pdf'))
        logger.info(f"检测到 {len(pdf_files)} 个PDF文件")
        
        documents = []
        for file_path in tqdm(pdf_files, desc='加载PDF文件'):
            try:
                with pdfplumber.open(file_path) as pdf:
                    text = "\n".join(page.extract_text() or '' for page in pdf.pages)
                
                file_info = self._get_file_info(file_path)
                documents.append({
                    'content': text,
                    'file_info': file_info,
                    'format': 'pdf'
                })
                
            except Exception as e:
                logger.error(f"加载PDF文件失败 {file_path}: {e}")
                continue
        
        return documents
    
    def _load_markdown_files(self, folder_path: str) -> List[Dict[str, Any]]:
        """
        加载Markdown文件
        
        Args:
            folder_path: 文件夹路径
            
        Returns:
            Markdown文档列表
        """
        md_files = glob.glob(os.path.join(folder_path, '*.md'))
        logger.info(f"检测到 {len(md_files)} 个Markdown文件")
        
        documents = []
        for file_path in tqdm(md_files, desc='加载Markdown文件'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                file_info = self._get_file_info(file_path)
                documents.append({
                    'content': text,
                    'file_info': file_info,
                    'format': 'markdown'
                })
                
            except Exception as e:
                logger.error(f"加载Markdown文件失败 {file_path}: {e}")
                continue
        
        return documents
    
    def _get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        获取文件信息
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件信息字典
        """
        stat = os.stat(file_path)
        return {
            'filename': os.path.basename(file_path),
            'filepath': file_path,
            'filesize': stat.st_size,
            'created_time': datetime.fromtimestamp(stat.st_ctime).isoformat(),
            'modified_time': datetime.fromtimestamp(stat.st_mtime).isoformat()
        }
    
    def get_supported_formats(self) -> List[str]:
        """
        获取支持的文档格式
        
        Returns:
            支持的格式列表
        """
        return self.supported_formats.copy() 