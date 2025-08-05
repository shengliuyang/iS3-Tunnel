"""
基础测试文件
测试项目的基本功能
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from config.logging_config import setup_logging

logger = setup_logging()


def test_settings():
    """测试配置设置"""
    assert settings.azure_openai_key is not None
    assert settings.azure_openai_endpoint is not None
    assert settings.ollama_base_url is not None
    assert settings.chroma_db_dir is not None


def test_imports():
    """测试模块导入"""
    try:
        from src.core.llm_client import LLMClient
        from src.core.embeddings import EmbeddingModel
        from src.core.vector_store import VectorStore
        from src.processors.text_processor import EnhancedTextProcessor
        from src.processors.document_loader import DocumentLoader
        from src.processors.metadata_extractor import MetadataExtractor
        from src.retrieval.enhanced_retriever import EnhancedRetriever
        from src.retrieval.keyword_matcher import KeywordMatcher
        from src.retrieval.reranker import Reranker
        from src.qa.qa_chain import QAChain
        from src.qa.prompt_templates import PromptTemplates
        from src.utils.file_utils import FileUtils
        from src.utils.progress_tracker import ProgressTracker
        from src.utils.error_handler import ErrorHandler
        assert True
    except ImportError as e:
        assert False, f"导入失败: {e}"


def test_directory_structure():
    """测试目录结构"""
    project_root = Path(__file__).parent.parent
    
    # 检查必要的目录是否存在
    required_dirs = [
        'config',
        'src',
        'src/core',
        'src/processors',
        'src/retrieval',
        'src/qa',
        'src/utils',
        'scripts',
        'data',
        'tests',
        'examples'
    ]
    
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        assert full_path.exists(), f"目录不存在: {dir_path}"
        assert full_path.is_dir(), f"不是目录: {dir_path}"


def test_required_files():
    """测试必要文件"""
    project_root = Path(__file__).parent.parent
    
    # 检查必要的文件是否存在
    required_files = [
        'README.md',
        'requirements.txt',
        'env_example.txt',
        '.gitignore',
        'config/settings.py',
        'config/logging_config.py',
        'src/__init__.py',
        'src/core/__init__.py',
        'src/processors/__init__.py',
        'src/retrieval/__init__.py',
        'src/qa/__init__.py',
        'src/utils/__init__.py',
        'scripts/build_vector_db.py',
        'scripts/run_qa.py',
        'examples/basic_usage.py'
    ]
    
    for file_path in required_files:
        full_path = project_root / file_path
        assert full_path.exists(), f"文件不存在: {file_path}"
        assert full_path.is_file(), f"不是文件: {file_path}"


if __name__ == "__main__":
    # 运行测试
    print("运行基础测试...")
    
    test_settings()
    print("✓ 配置测试通过")
    
    test_imports()
    print("✓ 导入测试通过")
    
    test_directory_structure()
    print("✓ 目录结构测试通过")
    
    test_required_files()
    print("✓ 必要文件测试通过")
    
    print("所有基础测试通过！") 