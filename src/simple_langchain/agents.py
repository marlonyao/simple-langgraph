"""
Simple LangChain — Agent + Tool（代理 + 工具调用）

核心概念：
- Tool：工具定义（name + description + func + args_schema）
- @tool 装饰器：从函数签名自动推导 name / description / args_schema
- AgentExecutor：驱动 Agent 运行，解析 ReAct 格式输出
- ToolCallingAgent：基于 LLM 结构化 tool_calls 的现代 Agent
- ReAct 循环：Thought → Action → Observation → Thought → ... → Final Answer
"""

import inspect
import re
from typing import Any, Callable, get_type_hints

from simple_langchain.llms import BaseLLM


class Tool:
    """
    工具定义。

    name：工具名（Agent 在 Action 中使用）
    description：工具描述（告诉 LLM 这个工具能做什么）
    func：工具的实际执行函数
    args_schema：可选，Pydantic BaseModel 类，定义多参数输入的 schema
    """

    def __init__(
        self,
        name: str,
        description: str,
        func: Callable,
        args_schema: type | None = None,
    ):
        self.name = name
        self.description = description
        self.func = func
        self.args_schema = args_schema

    def run(self, tool_input: str | dict) -> str:
        """
        执行工具。

        - 如果 args_schema 存在且 tool_input 是 dict：
          用 schema 验证/转换后，以关键字参数调用 func
        - 否则：直接以 str 传给 func
        """
        if self.args_schema is not None and isinstance(tool_input, dict):
            validated = self.args_schema.model_validate(tool_input)
            return str(self.func(**validated.model_dump()))
        return str(self.func(tool_input))


class AgentExecutor:
    """
    Agent 执行器。

    驱动 ReAct 循环：
    1. 把用户问题 + 工具列表 + 历史思考过程拼成 prompt
    2. LLM 生成 Thought / Action / Final Answer
    3. 如果是 Action：执行工具，结果作为 Observation 加入历史
    4. 如果是 Final Answer：返回结果
    5. 循环直到 Final Answer 或超过最大迭代次数
    """

    def __init__(
        self,
        llm: BaseLLM,
        tools: list[Tool],
        max_iterations: int = 10,
    ):
        self._llm = llm
        self._tools = {tool.name: tool for tool in tools}
        self._max_iterations = max_iterations

    def _build_prompt(self, question: str, scratchpad: str) -> str:
        """构建 Agent prompt"""
        tools_desc = "\n".join(
            f"- {name}: {tool.description}"
            for name, tool in self._tools.items()
        )

        prompt = f"""你可以使用以下工具：
{tools_desc}

请用以下格式回答：
Thought: 你的思考
Action: 工具名
Action Input: 工具输入
（或者直接回答）
Thought: 你的思考
Final Answer: 最终回答

用户问题：{question}

{scratchpad}
"""
        return prompt

    def _parse_action(self, text: str) -> tuple[str | None, str | None, str | None]:
        """
        解析 LLM 输出。

        返回 (action_name, action_input, final_answer)
        如果是 Final Answer，action_name 为 None
        """
        # 检查 Final Answer
        final_match = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL)
        if final_match:
            return None, None, final_match.group(1).strip()

        # 检查 Action
        action_match = re.search(r"Action:\s*(.+)", text)
        input_match = re.search(r"Action Input:\s*(.+)", text)

        if action_match and input_match:
            action_name = action_match.group(1).strip()
            action_input = input_match.group(1).strip()
            return action_name, action_input, None

        # 无法解析，当作 Final Answer
        return None, None, text.strip()

    def invoke(self, question: str) -> str:
        """执行 Agent"""
        scratchpad = ""
        iterations = 0

        while iterations < self._max_iterations:
            prompt = self._build_prompt(question, scratchpad)
            llm_output = self._llm.invoke(prompt)

            action_name, action_input, final_answer = self._parse_action(llm_output)

            if final_answer:
                return final_answer

            if action_name:
                # 执行工具
                if action_name in self._tools:
                    observation = self._tools[action_name].run(action_input)
                else:
                    observation = f"错误：工具 '{action_name}' 不存在。可用工具：{list(self._tools.keys())}"

                # 记录到 scratchpad
                scratchpad += f"\nThought: {llm_output}\nObservation: {observation}\n"
            else:
                # 无法解析，当作最终回答
                return llm_output.strip()

            iterations += 1

        raise RuntimeError(
            f"Agent exceeded maximum iterations ({self._max_iterations}). "
            f"Last scratchpad:\n{scratchpad}"
        )


def tool(
    func: Callable | None = None,
    *,
    name: str | None = None,
) -> Tool | Callable:
    """
    装饰器：把函数变成 Tool。

    自动从函数签名推导：
    - name：函数名（可被 name 参数覆盖）
    - description：函数 docstring
    - args_schema：多参数时自动用 Pydantic BaseModel

    用法：
        @tool
        def search(query: str) -> str:
            \"\"\"搜索信息\"\"\"
            return f"结果: {query}"

        @tool(name="web_search")
        def search(query: str) -> str:
            \"\"\"搜索\"\"\"
            ...
    """

    def _make_tool(fn: Callable) -> Tool:
        tool_name = name or fn.__name__
        description = (fn.__doc__ or "").strip()

        # 分析参数签名
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())
        hints = get_type_hints(fn) if hasattr(fn, '__annotations__') else {}

        # 单个 str 参数 → 不需要 args_schema（走老路径）
        if len(params) == 1 and hints.get(params[0].name) is str:
            return Tool(
                name=tool_name,
                description=description,
                func=fn,
                args_schema=None,
            )

        # 多参数或非 str → 自动生成 Pydantic schema
        if params:
            from pydantic import create_model

            field_definitions = {}
            for p in params:
                ptype = hints.get(p.name, str)
                default = ... if p.default is inspect.Parameter.empty else p.default
                field_definitions[p.name] = (ptype, default)

            args_schema = create_model(f"{tool_name}Input", **field_definitions)
        else:
            args_schema = None

        return Tool(
            name=tool_name,
            description=description,
            func=fn,
            args_schema=args_schema,
        )

    # 支持 @tool 和 @tool(name="xxx") 两种写法
    if func is not None:
        return _make_tool(func)
    return _make_tool


class ToolCallingAgent:
    """
    现代 Agent：基于 LLM 结构化 tool_calls。

    对比 AgentExecutor（ReAct 文本解析）：
    - LLM 返回 JSON 格式的 tool_calls（不是 Action/Action Input 文本）
    - Agent 直接解析 JSON 调用对应工具
    - 更可靠、更标准（对应真实 LangChain 的 create_tool_calling_agent）

    循环：
    1. LLM 返回 {"tool_calls": [...]} 或 {"content": "..."}
    2. 如果是 tool_calls → 执行工具 → 结果加入 messages → 继续循环
    3. 如果是 content → 返回文本回答
    """

    def __init__(
        self,
        llm: Any,  # FakeToolCallLLM 或支持 tool_call 的 LLM
        tools: list[Tool],
        max_iterations: int = 10,
    ):
        self._llm = llm
        self._tools = {tool.name: tool for tool in tools}
        self._max_iterations = max_iterations

    def invoke(self, question: str) -> str:
        """执行 Agent"""
        messages = [{"role": "user", "content": question}]
        iterations = 0

        while iterations < self._max_iterations:
            # 构建 prompt：把消息历史拼成字符串
            prompt_parts = []
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if isinstance(content, dict):
                    import json
                    content = json.dumps(content, ensure_ascii=False)
                prompt_parts.append(f"{role}: {content}")
            prompt = "\n".join(prompt_parts)

            response = self._llm.invoke(prompt)

            # 检查是否有 tool_calls
            if "tool_calls" in response:
                for tc in response["tool_calls"]:
                    tool_name = tc["name"]
                    tool_args = tc["arguments"]

                    if tool_name in self._tools:
                        observation = self._tools[tool_name].run(tool_args)
                    else:
                        observation = (
                            f"错误：工具 '{tool_name}' 不存在。"
                            f"可用工具：{list(self._tools.keys())}"
                        )

                    # 记录到消息历史
                    messages.append({
                        "role": "assistant",
                        "content": response,
                    })
                    messages.append({
                        "role": "tool",
                        "content": observation,
                    })
            elif "content" in response:
                return response["content"]
            else:
                # 无法识别的格式
                return str(response)

            iterations += 1

        raise RuntimeError(
            f"ToolCallingAgent exceeded maximum iterations ({self._max_iterations})."
        )
