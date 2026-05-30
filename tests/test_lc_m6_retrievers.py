"""
Milestone 6 测试：Retriever（检索 + RAG）

TDD RED 阶段：测试文档、分割、向量存储、检索链。
"""

import pytest
import math


# ============================================================
# Document
# ============================================================

class TestDocument:

    def test_create_document(self):
        from simple_langchain.retrievers import Document

        doc = Document(page_content="Hello world", metadata={"source": "test"})
        assert doc.page_content == "Hello world"
        assert doc.metadata["source"] == "test"

    def test_document_default_metadata(self):
        from simple_langchain.retrievers import Document

        doc = Document(page_content="content")
        assert doc.metadata == {}


# ============================================================
# TextSplitter
# ============================================================

class TestTextSplitter:

    def test_split_by_characters(self):
        from simple_langchain.retrievers import CharacterTextSplitter

        splitter = CharacterTextSplitter(chunk_size=10, chunk_overlap=0)
        text = "0123456789abcdefghij"
        chunks = splitter.split_text(text)
        assert len(chunks) == 2
        assert chunks[0] == "0123456789"
        assert chunks[1] == "abcdefghij"

    def test_split_with_overlap(self):
        from simple_langchain.retrievers import CharacterTextSplitter

        splitter = CharacterTextSplitter(chunk_size=10, chunk_overlap=3)
        text = "0123456789abcdefghij"
        chunks = splitter.split_text(text)
        # chunk1: 0123456789, chunk2: 789abcdefg (overlap=3), chunk3: efghij
        assert len(chunks) >= 2
        # 验证 overlap
        if len(chunks) >= 2:
            assert chunks[1][:3] == "789"

    def test_split_short_text(self):
        """文本比分块大小短时不分割"""
        from simple_langchain.retrievers import CharacterTextSplitter

        splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
        chunks = splitter.split_text("short")
        assert chunks == ["short"]

    def test_split_documents(self):
        from simple_langchain.retrievers import Document, CharacterTextSplitter

        splitter = CharacterTextSplitter(chunk_size=10, chunk_overlap=0)
        doc = Document(page_content="0123456789abcdefghij", metadata={"src": "t"})
        docs = splitter.split_documents([doc])
        assert len(docs) == 2
        assert docs[0].metadata["src"] == "t"
        assert docs[1].metadata["src"] == "t"


# ============================================================
# SimpleVectorStore
# ============================================================

class TestSimpleVectorStore:

    def test_add_and_search(self):
        from simple_langchain.retrievers import Document, SimpleVectorStore

        store = SimpleVectorStore()
        store.add_documents([
            Document(page_content="猫是一种宠物", metadata={"id": 1}),
            Document(page_content="狗是一种宠物", metadata={"id": 2}),
            Document(page_content="Python是编程语言", metadata={"id": 3}),
        ])

        results = store.search("动物宠物", k=2)
        assert len(results) == 2
        # 猫和狗应该排在 Python 前面
        ids = [r.metadata["id"] for r in results]
        assert 1 in ids or 2 in ids

    def test_search_with_scores(self):
        from simple_langchain.retrievers import SimpleVectorStore

        store = SimpleVectorStore()
        store.add_texts(["苹果是水果", "香蕉是水果", "汽车是交通工具"])

        results = store.search("水果", k=2, with_score=True)
        assert len(results) == 2
        for doc, score in results:
            assert isinstance(score, float)
            assert 0 <= score <= 1

    def test_empty_store_search(self):
        from simple_langchain.retrievers import SimpleVectorStore

        store = SimpleVectorStore()
        results = store.search("anything", k=3)
        assert results == []

    def test_k_larger_than_store(self):
        from simple_langchain.retrievers import SimpleVectorStore

        store = SimpleVectorStore()
        store.add_texts(["doc1", "doc2"])
        results = store.search("doc", k=10)
        assert len(results) == 2  # 最多返回所有文档


# ============================================================
# RetrievalChain
# ============================================================

class TestRetrievalChain:

    def test_retrieval_chain_invoke(self):
        """检索 + 生成完整流程"""
        from simple_langchain.retrievers import Document, SimpleVectorStore
        from simple_langchain.prompts import PromptTemplate
        from simple_langchain.llms import FakeListLLM
        from simple_langchain.chains import RetrievalChain

        store = SimpleVectorStore()
        store.add_documents([
            Document(page_content="LangGraph 是一个状态图框架"),
            Document(page_content="LangChain 是一个 LLM 应用框架"),
        ])

        prompt = PromptTemplate.from_template(
            "根据以下内容回答问题：\n{context}\n\n问题：{question}\n回答："
        )
        llm = FakeListLLM(responses=["LangChain 是一个 LLM 应用框架"])

        chain = RetrievalChain(
            retriever=store,
            prompt=prompt,
            llm=llm,
        )
        result = chain.invoke({"question": "LangChain是什么？"})
        assert "LLM" in result

    def test_retrieval_chain_uses_relevant_docs(self):
        """检索链只使用最相关的文档"""
        from simple_langchain.retrievers import Document, SimpleVectorStore
        from simple_langchain.prompts import PromptTemplate
        from simple_langchain.llms import FakeListLLM
        from simple_langchain.chains import RetrievalChain

        store = SimpleVectorStore()
        store.add_documents([
            Document(page_content="地球是圆的"),
            Document(page_content="太阳是恒星"),
            Document(page_content="月亮是卫星"),
        ])

        prompt = PromptTemplate.from_template("上下文：{context}\n问题：{question}")
        llm = FakeListLLM(responses=["太阳是恒星"])

        chain = RetrievalChain(retriever=store, prompt=prompt, llm=llm, k=2)
        result = chain.invoke({"question": "太阳是什么"})
        assert result == "太阳是恒星"
