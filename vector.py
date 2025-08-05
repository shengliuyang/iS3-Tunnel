import os
import glob
import time
import re
import json
from datetime import datetime
from typing import List, Dict, Any
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
# LLM配置（与process.py一致）
from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import pdfplumber

# 全局进度条对象
llm_pbar = None
llm_total = 0
llm_count = 0

def set_llm_progress_bar(total):
    global llm_pbar, llm_total, llm_count
    llm_total = total
    llm_count = 0
    if llm_pbar is not None:
        llm_pbar.close()
    llm_pbar = tqdm(total=llm_total, desc='LLM请求进度(分块级)', position=1)

def update_llm_progress_bar():
    global llm_pbar, llm_count
    if llm_pbar is not None:
        llm_pbar.update(1)
        llm_count += 1
        if llm_count >= llm_total:
            llm_pbar.close()
            llm_pbar = None

# 向量数据库保存路径
CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), 'chroma_db')

# 使用 Ollama 本地中文 embedding 模型
EMBEDDING_MODEL = OllamaEmbeddings(
    model="dengcao/Qwen3-Embedding-8B:Q8_0",
    base_url="http://localhost:11434"
)

# Azure OpenAI配置
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY", "your_azure_openai_key"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://your-resource.openai.azure.com")
)
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, min=2, max=20), reraise=True)
def llm_extract_keywords(text, top_k=6):
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
            response = client.chat.completions.create(
                model=DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": "你是一个专业的中文关键词提取助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                top_p=0.9,
                max_tokens=800,
                timeout=3000
            )
            result = response.choices[0].message.content.strip()
            keywords = [kw.strip() for kw in result.replace('\n', '').replace('，', ',').split(',') if kw.strip()]
            time.sleep(5)
            update_llm_progress_bar()
            return keywords[:top_k]
        except Exception as e:
            if '429' in str(e):
                print('[WARN] 检测到429限流，自动暂停65秒...')
                time.sleep(65)
                continue
            else:
                raise

@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, min=2, max=20), reraise=True)
def llm_generate_summary(text, max_length=200):
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
            response = client.chat.completions.create(
                model=DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": "你是一个专业的中文文本摘要助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                top_p=0.9,
                max_tokens=800,
                timeout=3000
            )
            summary = response.choices[0].message.content.strip()
            time.sleep(5)
            update_llm_progress_bar()
            return summary
        except Exception as e:
            if '429' in str(e):
                print('[WARN] 检测到429限流，自动暂停65秒...')
                time.sleep(65)
                continue
            else:
                raise

class EnhancedTextProcessor:
    """增强的文本处理器，实现语义分割和元数据提取"""
    
    def __init__(self, min_chunk_size=200, max_chunk_size=1500, chunk_overlap=300):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        print(f"[INFO] 初始化文本处理器: min_chunk_size={min_chunk_size}, max_chunk_size={max_chunk_size}, chunk_overlap={chunk_overlap}")
        # 初始化分割器
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )
        
        self.markdown_splitter = MarkdownTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def extract_document_structure(self, text: str, filename: str) -> Dict[str, Any]:
        print(f"[INFO] 正在提取文档结构和元数据: {filename}")
        # 检测文档类型
        doc_type = self._detect_document_type(filename, text)
        
        # 提取章节信息
        chapters = self._extract_chapters(text)
        
        # 用LLM提取关键词
        keywords = llm_extract_keywords(text)
        print(f"[INFO] 调用LLM提取关键词: {filename}")
        
        # 用LLM生成摘要
        summary = llm_generate_summary(text)
        print(f"[INFO] 调用LLM生成摘要: {filename}")
        
        # 提取时间信息
        timestamp = self._extract_timestamp(text)
        
        return {
            'doc_type': doc_type,
            'chapters': chapters,
            'keywords': keywords,
            'summary': summary,
            'timestamp': timestamp,
            'total_length': len(text),
            'paragraph_count': len(text.split('\n\n')),
            'sentence_count': len(re.split(r'[。！？]', text))
        }
    
    def _detect_document_type(self, filename: str, text: str) -> str:
        """检测文档类型"""
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
        """提取章节信息"""
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
    
    # 已删除 _extract_keywords 和 _generate_summary 函数，彻底去除jieba相关代码
    
    def _extract_timestamp(self, text: str) -> str:
        """提取时间戳信息"""
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
        """语义分割文本"""
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
        """按章节分割文本"""
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
        """处理分割后的文本块"""
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

def load_and_split_txts(folder, min_chunk_size=200, max_chunk_size=1500, chunk_overlap=300):
    print(f"[INFO] 开始加载并分割文本文件，目录: {folder}")
    txt_files = glob.glob(os.path.join(folder, '*.txt'))
    print(f"[INFO] 共检测到 {len(txt_files)} 个txt文件")
    processor = EnhancedTextProcessor(min_chunk_size, max_chunk_size, chunk_overlap)
    all_docs = []
    # 先统计所有分块总数
    total_chunks = 0
    chunk_counts = []
    for file in txt_files:
        with open(file, 'r', encoding='utf-8') as f:
            text = f.read()
            doc_structure = processor.extract_document_structure(text, os.path.basename(file))
            chunks = processor.split_text_semantically(text, doc_structure)
            chunk_counts.append(len(chunks))
            total_chunks += len(chunks)
    set_llm_progress_bar(total_chunks * 2)  # 每个分块关键词+摘要
    # 重新处理，正式生成分块和元数据
    for idx, file in enumerate(tqdm(txt_files, desc='读取txt文件')):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                text = f.read()
                file_info = {
                    'filename': os.path.basename(file),
                    'filepath': file,
                    'filesize': os.path.getsize(file),
                    'created_time': datetime.fromtimestamp(os.path.getctime(file)).isoformat(),
                    'modified_time': datetime.fromtimestamp(os.path.getmtime(file)).isoformat()
                }
                doc_structure = processor.extract_document_structure(text, file_info['filename'])
                print(f"[INFO] 正在对 {file_info['filename']} 进行语义分割...")
                chunks = processor.split_text_semantically(text, doc_structure)
                for chunk in chunks:
                    # 对每个分块调用LLM提取关键词和摘要
                    chunk['keywords'] = llm_extract_keywords(chunk['content'])
                    chunk['summary'] = llm_generate_summary(chunk['content'])
                    update_llm_progress_bar()
                    update_llm_progress_bar()
                    chunk['source'] = file_info['filename']
                    chunk['filepath'] = file_info['filepath']
                    chunk['filesize'] = file_info['filesize']
                    chunk['created_time'] = file_info['created_time']
                    chunk['modified_time'] = file_info['modified_time']
                    chunk.update({
                        'doc_type': doc_structure['doc_type'],
                        'timestamp': doc_structure['timestamp'],
                        'total_length': doc_structure['total_length'],
                        'paragraph_count': doc_structure['paragraph_count'],
                        'sentence_count': doc_structure['sentence_count']
                    })
                    all_docs.append(chunk)
        except Exception as e:
            print(f"[ERROR] 处理文件 {file} 时出错: {e}")
            continue
    print(f"[INFO] 总共处理了 {len(txt_files)} 个文件，生成了 {len(all_docs)} 个文本块")
    return all_docs

def load_and_split_pdfs(folder, min_chunk_size=200, max_chunk_size=1500, chunk_overlap=300):
    print(f"[INFO] 开始加载并分割PDF文件，目录: {folder}")
    pdf_files = glob.glob(os.path.join(folder, '*.pdf'))
    print(f"[INFO] 共检测到 {len(pdf_files)} 个pdf文件")
    processor = EnhancedTextProcessor(min_chunk_size, max_chunk_size, chunk_overlap)
    all_docs = []
    # 先统计所有分块总数
    total_chunks = 0
    chunk_counts = []
    for file in pdf_files:
        with pdfplumber.open(file) as pdf:
            text = "\n".join(page.extract_text() or '' for page in pdf.pages)
            doc_structure = processor.extract_document_structure(text, os.path.basename(file))
            chunks = processor.split_text_semantically(text, doc_structure)
            chunk_counts.append(len(chunks))
            total_chunks += len(chunks)
    set_llm_progress_bar(total_chunks * 2)  # 每个分块关键词+摘要
    # 重新处理，正式生成分块和元数据
    for idx, file in enumerate(tqdm(pdf_files, desc='读取pdf文件')):
        try:
            with pdfplumber.open(file) as pdf:
                text = "\n".join(page.extract_text() or '' for page in pdf.pages)
                file_info = {
                    'filename': os.path.basename(file),
                    'filepath': file,
                    'filesize': os.path.getsize(file),
                    'created_time': datetime.fromtimestamp(os.path.getctime(file)).isoformat(),
                    'modified_time': datetime.fromtimestamp(os.path.getmtime(file)).isoformat()
                }
                doc_structure = processor.extract_document_structure(text, file_info['filename'])
                print(f"[INFO] 正在对 {file_info['filename']} 进行语义分割...")
                chunks = processor.split_text_semantically(text, doc_structure)
                for chunk in chunks:
                    chunk['keywords'] = llm_extract_keywords(chunk['content'])
                    chunk['summary'] = llm_generate_summary(chunk['content'])
                    update_llm_progress_bar()
                    update_llm_progress_bar()
                    chunk['source'] = file_info['filename']
                    chunk['filepath'] = file_info['filepath']
                    chunk['filesize'] = file_info['filesize']
                    chunk['created_time'] = file_info['created_time']
                    chunk['modified_time'] = file_info['modified_time']
                    chunk.update({
                        'doc_type': doc_structure['doc_type'],
                        'timestamp': doc_structure['timestamp'],
                        'total_length': doc_structure['total_length'],
                        'paragraph_count': doc_structure['paragraph_count'],
                        'sentence_count': doc_structure['sentence_count']
                    })
                    all_docs.append(chunk)
        except Exception as e:
            print(f"[ERROR] 处理文件 {file} 时出错: {e}")
            continue
    print(f"[INFO] 总共处理了 {len(pdf_files)} 个pdf文件，生成了 {len(all_docs)} 个文本块")
    return all_docs

def build_chroma_db(docs, persist_dir=CHROMA_DB_DIR, batch_size=10):
    print(f"[INFO] 开始构建Chroma向量数据库，目标目录: {persist_dir}")
    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir)
    
    # 先删除原有向量库
    if os.path.exists(os.path.join(persist_dir, 'index')):
        import shutil
        print(f"[INFO] 检测到已有向量库，正在删除旧的index...")
        shutil.rmtree(os.path.join(persist_dir, 'index'))
    
    # 初始化 Chroma vectorstore
    vectorstore = Chroma(
        collection_name="rag_db",
        persist_directory=persist_dir,
        embedding_function=EMBEDDING_MODEL
    )
    
    # 构造langchain文档对象
    all_docs = []
    for idx, doc in enumerate(tqdm(docs, desc='准备文档对象')):
        # 创建丰富的元数据
        metadata = {
            'source': doc['source'],
            'doc_id': idx,
            'doc_type': doc.get('doc_type', 'unknown'),
            'filename': doc['source'],
            'filepath': doc.get('filepath', ''),
            'filesize': doc.get('filesize', 0),
            'created_time': doc.get('created_time', ''),
            'modified_time': doc.get('modified_time', ''),
            'chunk_index': doc.get('chunk_index', 0),
            'total_chunks': doc.get('total_chunks', 1),
            'chunk_size': doc.get('chunk_size', len(doc['content'])),
            'start_position': doc.get('start_position', -1),
            'end_position': doc.get('end_position', -1),
            'keywords': json.dumps(doc.get('keywords', []), ensure_ascii=False),
            'summary': doc.get('summary', ''),
            'timestamp': doc.get('timestamp', ''),
            'total_length': doc.get('total_length', 0),
            'paragraph_count': doc.get('paragraph_count', 0),
            'sentence_count': doc.get('sentence_count', 0),
            'chapter_title': doc.get('chapter_title', ''),
            'chapter_level': doc.get('chapter_level', 0)
        }
        
        all_docs.append(Document(
            page_content=doc['content'],
            metadata=metadata
        ))
    
    # 批量写入
    print(f"[INFO] 开始批量写入向量库，总分块数: {len(all_docs)}，批量大小: {batch_size}")
    for i in tqdm(range(0, len(all_docs), batch_size), desc='写入向量库'):
        batch_docs = all_docs[i:i+batch_size]
        try:
            vectorstore.add_documents(batch_docs)
        except Exception as e:
            print(f"[ERROR] 批量写入出错，重试中: {e}")
            time.sleep(3)
            try:
                vectorstore.add_documents(batch_docs)
            except Exception as e2:
                print(f"[ERROR] 重试失败，跳过该批: {e2}")
        time.sleep(0.5)
    
    print(f'[INFO] 已完成{len(all_docs)}个分块的向量化，向量库保存在: {persist_dir}')
    
    # 保存处理统计信息
    stats = {
        'total_documents': len(set(doc['source'] for doc in docs)),
        'total_chunks': len(all_docs),
        'doc_types': {},
        'avg_chunk_size': sum(len(doc['content']) for doc in docs) / len(docs) if docs else 0,
        'processing_time': datetime.now().isoformat()
    }
    
    # 统计文档类型分布
    for doc in docs:
        doc_type = doc.get('doc_type', 'unknown')
        stats['doc_types'][doc_type] = stats['doc_types'].get(doc_type, 0) + 1
    
    stats_file = os.path.join(persist_dir, 'processing_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"[INFO] 处理统计信息已保存到: {stats_file}")

def main():
    folder = r"./data/raw"
    print(f"[INFO] 主流程开始，输入目录: {folder}")
    if not os.path.isdir(folder):
        print('[ERROR] 文件夹不存在')
        return
    # 使用优化的参数
    docs_txt = load_and_split_txts(
        folder, 
        min_chunk_size=200,
        max_chunk_size=1500,
        chunk_overlap=100
    )
    docs_pdf = load_and_split_pdfs(
        folder,
        min_chunk_size=200,
        max_chunk_size=1500,
        chunk_overlap=100
    )
    docs = docs_txt + docs_pdf
    build_chroma_db(docs)
    print("[INFO] 全部流程已完成！")

if __name__ == '__main__':
    main()
