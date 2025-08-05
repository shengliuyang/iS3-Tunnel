import os
import warnings
import logging
import json
from tqdm import tqdm
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
import chromadb
from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

# 禁用所有警告
warnings.filterwarnings("ignore")
# 禁用日志输出
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("langchain").setLevel(logging.ERROR)
logging.getLogger("langchain_community").setLevel(logging.ERROR)
logging.getLogger("langchain_ollama").setLevel(logging.ERROR)

# LLM配置
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY", "your_azure_openai_key")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://your-resource.openai.azure.com")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), 'chroma_db')

# 使用Ollama本地中文embedding模型
embeddings = OllamaEmbeddings(
    model="dengcao/Qwen3-Embedding-8B:Q8_0",
    base_url="http://localhost:11434"
)
llm = AzureChatOpenAI(
    openai_api_key=AZURE_OPENAI_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    deployment_name=AZURE_OPENAI_DEPLOYMENT,
    openai_api_version=AZURE_OPENAI_API_VERSION,
    temperature=0.1,
    max_tokens=5000,
)

client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)
DEPLOYMENT_NAME = AZURE_OPENAI_DEPLOYMENT

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=20), reraise=True)
def llm_extract_keywords(text, top_k=3):
    prompt = f"""
请从下面的文本中提取最能代表主题的{top_k}个关键词，要求：
1. 只返回关键词本身，不要编号，不要解释，不要分行，每个关键词用逗号分隔。
2. 关键词可以是词组或短语，尽量覆盖文本核心内容。
3. 只输出关键词列表，不要多余内容。

原始文本如下：
{text}
"""
    response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": "你是一个专业的中文关键词提取助手。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        top_p=0.9,
        max_tokens=1200,
        timeout=3000
    )
    result = response.choices[0].message.content.strip()
    keywords = [kw.strip() for kw in result.replace('\n', '').replace('，', ',').split(',') if kw.strip()]
    return set(keywords[:top_k])

def enhanced_retrieve(vectordb, query, top_k=20, keyword_weight=2):
    """
    高级检索：embedding召回+LLM关键词加权+元数据排序
    返回最相关的分块内容和元数据
    """
    # 1. 语义召回
    docs = vectordb.similarity_search_with_score(query, k=top_k*2)
    # 2. 用LLM提取用户问题关键词
    query_keywords = llm_extract_keywords(query, top_k=3)
    scored = []
    for doc, score in docs:
        meta = doc.metadata
        try:
            doc_keywords = set(json.loads(meta.get('keywords', '[]')))
        except:
            doc_keywords = set()
        keyword_overlap = len(query_keywords & doc_keywords)
        final_score = score - keyword_weight * keyword_overlap
        scored.append((doc, final_score, keyword_overlap))
    scored.sort(key=lambda x: (x[1], -x[2]))
    return [doc for doc, _, _ in scored[:top_k]]

def build_qa_chain():
    vectordb = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings,
        collection_name='rag_db',
    )
    def custom_retriever(query):
        docs = enhanced_retrieve(vectordb, query, top_k=9)
        # 拼接内容+元数据摘要
        context = "\n\n".join([
            f"【章节】{d.metadata.get('chapter_title','')}\n【摘要】{d.metadata.get('summary','')}\n【关键词】{json.loads(d.metadata.get('keywords','[]'))}\n{d.page_content}"
            for d in docs
        ])
        return context
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
你是一个专业的知识问答助手。请结合以下知识库内容回答问题，所有题目均为单选题，只有一个选项正确。如果知识库无法回答，可以用你自己的知识补充，但必须给出一个明确的选项答案。
知识库内容：{context}
问题：{question}
请直接给出选择题的标准答案（只要选项字母或数字，不要题干、解释、标点、空格）。
"""
    )
    def qa_func(inputs):
        context = custom_retriever(inputs["query"])
        _input = {"context": context, "question": inputs["query"]}
        return llm.invoke(prompt.format_prompt(**_input).to_string())
    return qa_func

def main():
    question_file = os.path.join('data', 'questions.txt')
    if not os.path.exists(question_file):
        print('未找到question.txt')
        return
    with open(question_file, 'r', encoding='utf-8') as f:
        questions = [line.strip() for line in f if line.strip()]
    qa_func = build_qa_chain()
    results = []
    print("[INFO] 正在RAG答题...")
    for q in tqdm(questions, desc='RAG答题进度'):
        try:
            ans_obj = qa_func({"query": q})
            ans_str = ans_obj.content if hasattr(ans_obj, "content") else str(ans_obj)
            ans = ans_str.split()[0] if ans_str else ''
            results.append(f"{q} 答案：{ans}")
        except Exception as e:
            print(f"[ERROR] 答题失败: {e}")
            results.append(f"{q} 答案：[ERROR]")
    with open('data/answers.txt', 'w', encoding='utf-8') as f:
        for item in results:
            f.write(item + '\n')
    print(f"[INFO] 已完成{len(results)}道题目的RAG作答，结果已写入rag_question.txt")

if __name__ == '__main__':
    main()
