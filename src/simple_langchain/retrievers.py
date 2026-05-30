"""
Simple LangChain — Retriever（检索 + RAG）

核心概念：
- Document：文档对象（page_content + metadata）
- CharacterTextSplitter：按字符数分割文档
- SimpleVectorStore：内存向量存储（TF-IDF + 余弦相似度）
- RetrievalChain：检索 + 生成完整流程
"""

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from simple_langchain.llms import BaseLLM
from simple_langchain.parsers import BaseOutputParser, StrOutputParser
from simple_langchain.prompts import PromptTemplate


@dataclass
class Document:
    """文档对象"""
    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class CharacterTextSplitter:
    """
    按字符数分割文本。

    chunk_size：每块最大字符数
    chunk_overlap：相邻块重叠字符数
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 0):
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> list[str]:
        if len(text) <= self._chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + self._chunk_size
            chunks.append(text[start:end])
            start = end - self._chunk_overlap

        return chunks

    def split_documents(self, documents: list[Document]) -> list[Document]:
        result = []
        for doc in documents:
            chunks = self.split_text(doc.page_content)
            for chunk in chunks:
                result.append(Document(
                    page_content=chunk,
                    metadata=dict(doc.metadata),
                ))
        return result


def _tokenize(text: str) -> list[str]:
    """简单的中英文分词"""
    # 英文单词 + 中文字符
    tokens = re.findall(r'[a-zA-Z]+|[\u4e00-\u9fff]', text.lower())
    return tokens


def _cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """计算两个稀疏向量的余弦相似度"""
    # 找共同 key
    common = set(vec_a.keys()) & set(vec_b.keys())
    if not common:
        return 0.0

    dot = sum(vec_a[k] * vec_b[k] for k in common)
    norm_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    norm_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


class SimpleVectorStore:
    """
    内存向量存储。

    使用简单的 TF-IDF + 余弦相似度进行检索。
    不依赖任何外部向量数据库或 embedding 模型。
    """

    def __init__(self):
        self._documents: list[Document] = []
        self._tf: list[dict[str, float]] = []  # 每个文档的 TF 向量

    def add_documents(self, documents: list[Document]) -> None:
        for doc in documents:
            self._documents.append(doc)
            tokens = _tokenize(doc.page_content)
            tf = Counter(tokens)
            total = len(tokens) if tokens else 1
            self._tf.append({k: v / total for k, v in tf.items()})

    def add_texts(self, texts: list[str]) -> None:
        self.add_documents([Document(page_content=t) for t in texts])

    def search(
        self, query: str, k: int = 4, with_score: bool = False
    ) -> list[Document] | list[tuple[Document, float]]:
        if not self._documents:
            return []

        query_tokens = _tokenize(query)
        query_tf = Counter(query_tokens)
        total = len(query_tokens) if query_tokens else 1
        query_vec = {k: v / total for k, v in query_tf.items()}

        # 计算每个文档的相似度
        scored = []
        for i, doc_vec in enumerate(self._tf):
            score = _cosine_similarity(query_vec, doc_vec)
            scored.append((self._documents[i], score))

        # 按分数降序排列
        scored.sort(key=lambda x: x[1], reverse=True)
        top_k = scored[:k]

        if with_score:
            return top_k
        return [doc for doc, _ in top_k]


class RetrievalChain:
    """
    检索 + 生成链。

    流程：query → 检索相关文档 → 拼入 prompt → LLM 生成回答
    """

    def __init__(
        self,
        retriever: SimpleVectorStore,
        prompt: PromptTemplate,
        llm: BaseLLM,
        output_parser: BaseOutputParser | None = None,
        k: int = 4,
    ):
        self._retriever = retriever
        self._prompt = prompt
        self._llm = llm
        self._output_parser = output_parser or StrOutputParser()
        self._k = k

    def invoke(self, inputs: dict[str, Any]) -> Any:
        query = inputs.get("question", "")

        # 1. 检索相关文档
        docs = self._retriever.search(query, k=self._k)
        context = "\n".join(doc.page_content for doc in docs)

        # 2. 填入 prompt
        full_inputs = dict(inputs)
        full_inputs["context"] = context

        formatted = self._prompt.format(**full_inputs)

        # 3. LLM 生成
        llm_output = self._llm.invoke(formatted)

        # 4. 解析输出
        return self._output_parser.parse(llm_output)
