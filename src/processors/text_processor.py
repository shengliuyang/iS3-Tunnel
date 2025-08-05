"""
增强文本处理器
实现语义分割、文档结构提取和智能分块
"""

import re
from datetime import datetime
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownTextSplitter
from config.settings import settings
from config.logging_config import get_logger

logger = get_logger(__name__)


class EnhancedTextProcessor:
    """增强的文本处理器，实现语义分割和元数据提取"""
    
    def __init__(self, min_chunk_size: int = None, max_chunk_size: int = None, chunk_overlap: int = None):
        """
        初始化文本处理器
        
        Args:
            min_chunk_size: 最小分块大小
            max_chunk_size: 最大分块大小
            chunk_overlap: 分块重叠大小
        """
        self.min_chunk_size = min_chunk_size or settings.min_chunk_size
        self.max_chunk_size = max_chunk_size or settings.max_chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        
        logger.info(f"初始化文本处理器: min_chunk_size={self.min_chunk_size}, "
                   f"max_chunk_size={self.max_chunk_size}, chunk_overlap={self.chunk_overlap}")
        
        # 初始化分割器
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )
        
        self.markdown_splitter = MarkdownTextSplitter(
            chunk_size=self.max_chunk_size,
            chunk_overlap=self.chunk_overlap
        )
    
    def extract_document_structure(self, text: str, filename: str) -> Dict[str, Any]:
        """
        提取文档结构和元数据
        
        Args:
            text: 文档文本
            filename: 文件名
            
        Returns:
            文档结构信息
        """
        logger.info(f"正在提取文档结构和元数据: {filename}")
        
        # 检测文档类型
        doc_type = self._detect_document_type(filename, text)
        
        # 提取章节信息
        chapters = self._extract_chapters(text)
        
        # 提取时间信息
        timestamp = self._extract_timestamp(text)
        
        return {
            'doc_type': doc_type,
            'chapters': chapters,
            'timestamp': timestamp,
            'total_length': len(text),
            'paragraph_count': len(text.split('\n\n')),
            'sentence_count': len(re.split(r'[。！？]', text))
        }
    
    def _detect_document_type(self, filename: str, text: str) -> str:
        """
        检测文档类型
        
        Args:
            filename: 文件名
            text: 文档文本
            
        Returns:
            文档类型
        """
        filename_lower = filename.lower()
        text_lower = text[:1000].lower()  # 只检查前1000个字符
        
        if any(keyword in filename_lower for keyword in ['标准', '规范', '规程']):
            return 'standard'
        elif any(keyword in filename_lower for keyword in ['设计', '施工']):
            return 'design'
        elif any(keyword in filename_lower for keyword in ['报告', '分析']):
            return 'report'
        elif any(keyword in text_lower for keyword in ['第.*章', '第.*节']):
            return 'chapter'
        else:
            return 'general'
    
    def _extract_chapters(self, text: str) -> List[Dict[str, Any]]:
        """
        提取章节信息
        
        Args:
            text: 文档文本
            
        Returns:
            章节信息列表
        """
        chapters = []
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            # 匹配章节标题模式
            chapter_match = re.match(r'^第[一二三四五六七八九十\d]+[章节]', line.strip())
            if chapter_match:
                chapters.append({
                    'title': line.strip(),
                    'start_line': i,
                    'level': 1 if '章' in line else 2
                })
        
        return chapters
    
    def _extract_timestamp(self, text: str) -> str:
        """
        提取时间戳信息
        
        Args:
            text: 文档文本
            
        Returns:
            时间戳字符串
        """
        # 查找日期模式
        date_patterns = [
            r'\d{4}年\d{1,2}月\d{1,2}日',
            r'\d{4}-\d{1,2}-\d{1,2}',
            r'\d{4}/\d{1,2}/\d{1,2}',
            r'\d{4}\.\d{1,2}\.\d{1,2}'
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            if matches:
                return matches[0]
        
        return datetime.now().strftime('%Y-%m-%d')
    
    def split_text_semantically(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        语义分割文本
        
        Args:
            text: 文档文本
            metadata: 文档元数据
            
        Returns:
            分块列表
        """
        chunks = []
        
        # 根据文档类型选择分割策略
        if metadata['doc_type'] == 'chapter' and metadata['chapters']:
            # 按章节分割
            chunks = self._split_by_chapters(text, metadata['chapters'])
        else:
            # 使用递归分割器
            text_chunks = self.recursive_splitter.split_text(text)
            chunks = self._process_chunks(text_chunks, text, metadata)
        
        # 过滤太小的块
        chunks = [chunk for chunk in chunks if len(chunk['content']) >= self.min_chunk_size]
        
        return chunks
    
    def _split_by_chapters(self, text: str, chapters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        按章节分割文本
        
        Args:
            text: 文档文本
            chapters: 章节信息
            
        Returns:
            分块列表
        """
        chunks = []
        lines = text.split('\n')
        
        for i, chapter in enumerate(chapters):
            start_line = chapter['start_line']
            end_line = chapters[i + 1]['start_line'] if i + 1 < len(chapters) else len(lines)
            
            chapter_text = '\n'.join(lines[start_line:end_line])
            
            # 对章节内容进行进一步分割
            if len(chapter_text) > self.max_chunk_size:
                sub_chunks = self.recursive_splitter.split_text(chapter_text)
                for j, sub_chunk in enumerate(sub_chunks):
                    chunks.append({
                        'content': sub_chunk,
                        'chapter_title': chapter['title'],
                        'chapter_level': chapter['level'],
                        'chunk_index': j,
                        'total_chunks': len(sub_chunks),
                        'start_line': start_line,
                        'end_line': end_line
                    })
            else:
                chunks.append({
                    'content': chapter_text,
                    'chapter_title': chapter['title'],
                    'chapter_level': chapter['level'],
                    'chunk_index': 0,
                    'total_chunks': 1,
                    'start_line': start_line,
                    'end_line': end_line
                })
        
        return chunks
    
    def _process_chunks(self, text_chunks: List[str], original_text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        处理分割后的文本块
        
        Args:
            text_chunks: 文本块列表
            original_text: 原始文本
            metadata: 文档元数据
            
        Returns:
            处理后的分块列表
        """
        chunks = []
        
        for i, chunk in enumerate(text_chunks):
            # 计算在原文中的位置
            start_pos = original_text.find(chunk)
            end_pos = start_pos + len(chunk) if start_pos != -1 else -1
            
            chunks.append({
                'content': chunk,
                'chunk_index': i,
                'total_chunks': len(text_chunks),
                'start_position': start_pos,
                'end_position': end_pos,
                'chunk_size': len(chunk)
            })
        
        return chunks 