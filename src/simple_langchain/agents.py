"""
Simple LangChain — Agent + Tool（代理 + 工具调用）

核心概念：
- Tool：工具定义（name + description + func）
- AgentExecutor：驱动 Agent 运行，解析 ReAct 格式输出
- ReAct 循环：Thought → Action → Observation → Thought → ... → Final Answer
"""

import re
from typing import Any, Callable

from simple_langchain.llms import BaseLLM


class Tool:
    """
    工具定义。

    name：工具名（Agent 在 Action 中使用）
    description：工具描述（告诉 LLM 这个工具能做什么）
    func：工具的实际执行函数
    """

    def __init__(self, name: str, description: str, func: Callable):
        self.name = name
        self.description = description
        self.func = func

    def run(self, tool_input: str) -> str:
        """执行工具"""
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
