"""
Simple LangChain — Chain（链式调用）

核心概念：
- LLMChain：prompt → llm → parser，一步到位
- SequentialChain：多个 chain 串行，前一个输出是后一个输入
- batch()：批量调用
"""

from typing import Any

from simple_langchain.llms import BaseLLM
from simple_langchain.parsers import BaseOutputParser, StrOutputParser
from simple_langchain.prompts import PromptTemplate


class LLMChain:
    """
    最核心的链：prompt → llm → parser。

    invoke() 接收一个 dict（匹配 prompt 变量），
    返回解析后的结果（默认 str）。
    """

    def __init__(
        self,
        prompt: PromptTemplate,
        llm: BaseLLM,
        output_parser: BaseOutputParser | None = None,
        output_key: str = "text",
        memory: Any | None = None,
    ):
        self.prompt = prompt
        self.llm = llm
        self.output_parser = output_parser or StrOutputParser()
        self.output_key = output_key
        self.memory = memory

    @property
    def input_keys(self) -> list[str]:
        """这个 chain 需要哪些输入变量"""
        return list(self.prompt.input_variables)

    def invoke(self, inputs: dict[str, Any]) -> Any:
        """
        执行链：
        1. 注入记忆到 inputs（如有 memory）
        2. 自动注入 parser 的 format_instructions（如 prompt 有此变量）
        3. 用 inputs 填充 prompt 模板
        4. 调用 LLM
        5. 解析输出
        6. 保存到 memory（如有 memory）
        """
        inputs = dict(inputs)

        # 注入记忆
        if self.memory:
            history = self.memory.load_memory()
            history_str = "\n".join(
                f"{m['role']}: {m['content']}" for m in history
            )
            inputs["history"] = history_str

        # 自动注入 format_instructions（仅当 prompt 模板有此变量时）
        if "format_instructions" in self.prompt.input_variables:
            inputs.setdefault(
                "format_instructions",
                self.output_parser.get_format_instructions(),
            )

        # 1. 格式化 prompt
        formatted = self.prompt.format(**inputs)

        # 2. 调用 LLM
        llm_output = self.llm.invoke(formatted)

        # 3. 解析输出
        result = self.output_parser.parse(llm_output)

        # 4. 保存到 memory
        if self.memory:
            human_input = inputs.get("input", "")
            self.memory.save_context(human_input, str(result))

        return result

    def batch(self, inputs_list: list[dict[str, Any]]) -> list[Any]:
        """批量调用：对每个输入执行 invoke"""
        return [self.invoke(inputs) for inputs in inputs_list]


class SequentialChain:
    """
    串行链：多个 chain 按顺序执行。

    前一个 chain 的输出（通过 output_key）会合并到上下文 dict 中，
    供后续 chain 使用。最终返回包含所有 output_key 的完整 dict。
    """

    def __init__(
        self,
        chains: list[LLMChain],
        input_variables: list[str],
    ):
        self.chains = chains
        self._input_variables = input_variables

    @property
    def input_keys(self) -> list[str]:
        return list(self._input_variables)

    def invoke(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        按顺序执行所有 chain。

        上下文从 inputs 开始，每个 chain 的结果写入 context[output_key]。
        """
        context = dict(inputs)

        for chain in self.chains:
            # 从 context 中提取 chain 需要的变量
            chain_inputs = {k: context[k] for k in chain.input_keys if k in context}
            result = chain.invoke(chain_inputs)
            context[chain.output_key] = result

        return context


class RetrievalChain:
    """
    检索 + 生成链：query → 检索 → 拼入 prompt → LLM 生成。
    """

    def __init__(
        self,
        retriever: Any,
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
