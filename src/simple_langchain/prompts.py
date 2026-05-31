"""
Simple LangChain — Prompt 模板

核心概念：
- PromptTemplate：纯文本模板，支持 {variable} 变量插值（继承 Runnable）
- ChatPromptTemplate：消息角色模板（system / human / ai）
- partial()：偏应用，预填充部分变量
"""

import re
from typing import Any

from simple_langchain.runnable import Runnable


class PromptTemplate(Runnable):
    """
    提示词模板。

    用 {variable} 语法在字符串中标记变量位置，
    调用 format() 时用实际值替换。

    支持双花括号转义：{{ }} → { }

    作为 Runnable：invoke(dict) → format(dict) → str
    """

    def __init__(
        self,
        template: str,
        input_variables: list[str],
    ):
        self.template = template
        self.input_variables = list(input_variables)

    @classmethod
    def from_template(cls, template: str) -> "PromptTemplate":
        """
        工厂方法：从模板字符串自动检测变量。

        检测规则：
        - {name} → 变量
        - {{ }} → 转义的花括号，不算变量
        """
        # 先把 {{ }} 替换为临时占位符，避免干扰变量检测
        temp = template.replace("{{", "<<DQ>>").replace("}}", "<<DQ>>")
        # 找所有 {variable}
        variables = re.findall(r"\{(\w+)\}", temp)
        # 去重并保持顺序
        seen = set()
        unique = []
        for v in variables:
            if v not in seen:
                seen.add(v)
                unique.append(v)
        return cls(template=template, input_variables=unique)

    def format(self, **kwargs: Any) -> str:
        """
        用实际值替换模板中的变量。

        规则：
        - {{ }} → { }（转义）
        - {name} → 用 kwargs[name] 替换
        - 缺少变量 → KeyError
        - 多余变量 → 忽略
        """
        # 检查必需变量
        missing = set(self.input_variables) - set(kwargs.keys())
        if missing:
            raise KeyError(f"Missing variables: {missing}")

        # 先处理转义：{{ → 临时标记
        result = self.template.replace("{{", "\x00").replace("}}", "\x01")

        # 替换变量
        for key, value in kwargs.items():
            result = result.replace("{" + key + "}", str(value))

        # 还原转义
        result = result.replace("\x00", "{").replace("\x01", "}")

        return result

    def partial(self, **kwargs: Any) -> "PromptTemplate":
        """
        偏应用：预填充部分变量，返回一个新的 PromptTemplate。

        新模板的变量列表 = 原变量 - 已填充的变量
        """
        # 先把已填充的变量替换进模板
        new_template = self.template
        for key, value in kwargs.items():
            new_template = new_template.replace("{" + key + "}", str(value))

        # 新的变量列表
        new_variables = [v for v in self.input_variables if v not in kwargs]

        return PromptTemplate(template=new_template, input_variables=new_variables)

    def invoke(self, input: dict[str, Any]) -> str:
        """Runnable 接口：input 是 dict，输出格式化后的字符串"""
        return self.format(**input)


class ChatPromptTemplate:
    """
    聊天消息模板。

    由多条 (role, content_template) 组成，
    format_messages() 返回消息列表。

    每条消息是 {"role": ..., "content": ...} 格式的 dict。
    """

    def __init__(
        self,
        messages: list[tuple[str, str]],
        input_variables: list[str],
    ):
        self.messages = messages
        self.input_variables = list(input_variables)

    @classmethod
    def from_messages(
        cls, message_templates: list[tuple[str, str]]
    ) -> "ChatPromptTemplate":
        """
        工厂方法：从消息模板列表创建。

        参数格式：[("role", "content template"), ...]
        自动收集所有模板中的变量。
        """
        variables = []
        seen = set()
        for role, content in message_templates:
            temp = content.replace("{{", "<<DQ>>").replace("}}", "<<DQ>>")
            for var in re.findall(r"\{(\w+)\}", temp):
                if var not in seen:
                    seen.add(var)
                    variables.append(var)

        return cls(messages=message_templates, input_variables=variables)

    def format_messages(self, **kwargs: Any) -> list[dict[str, str]]:
        """
        用实际值填充所有消息模板，返回消息列表。

        每条消息：{"role": "system", "content": "填充后的内容"}
        """
        # 检查必需变量
        missing = set(self.input_variables) - set(kwargs.keys())
        if missing:
            raise KeyError(f"Missing variables: {missing}")

        result = []
        for role, content in self.messages:
            # 复用 PromptTemplate 的格式化逻辑
            if self.input_variables or "{{" in content or "}}" in content:
                tmpl = PromptTemplate.from_template(content)
                filled = tmpl.format(**kwargs)
            else:
                filled = content
            result.append({"role": role, "content": filled})

        return result
