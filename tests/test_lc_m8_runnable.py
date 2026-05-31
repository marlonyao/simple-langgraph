"""
Milestone 8 测试：Runnable 协议（LCEL 基础）

TDD RED 阶段：Runnable 基类、RunnableSequence、RunnableLambda、管道操作。
"""

import pytest


# ============================================================
# Runnable 基类
# ============================================================

class TestRunnableBase:

    def test_runnable_has_invoke(self):
        """Runnable 必须实现 invoke"""
        from simple_langchain.runnable import Runnable

        class DoubleRunnable(Runnable):
            def invoke(self, input):
                return input * 2

        r = DoubleRunnable()
        assert r.invoke(3) == 6

    def test_runnable_batch(self):
        """batch 批量调用 invoke"""
        from simple_langchain.runnable import Runnable

        class DoubleRunnable(Runnable):
            def invoke(self, input):
                return input * 2

        r = DoubleRunnable()
        results = r.batch([1, 2, 3])
        assert results == [2, 4, 6]

    def test_runnable_batch_empty(self):
        """batch 空列表"""
        from simple_langchain.runnable import Runnable

        class EchoRunnable(Runnable):
            def invoke(self, input):
                return input

        r = EchoRunnable()
        assert r.batch([]) == []


# ============================================================
# RunnableLambda
# ============================================================

class TestRunnableLambda:

    def test_lambda_wraps_function(self):
        """RunnableLambda 把普通函数包成 Runnable"""
        from simple_langchain.runnable import RunnableLambda

        r = RunnableLambda(lambda x: x + 1)
        assert r.invoke(5) == 6

    def test_lambda_batch(self):
        """RunnableLambda 支持 batch"""
        from simple_langchain.runnable import RunnableLambda

        r = RunnableLambda(lambda x: x ** 2)
        assert r.batch([2, 3, 4]) == [4, 9, 16]


# ============================================================
# RunnableSequence（管道）
# ============================================================

class TestRunnableSequence:

    def test_pipe_operator(self):
        """a | b 创建 RunnableSequence"""
        from simple_langchain.runnable import Runnable, RunnableLambda

        add1 = RunnableLambda(lambda x: x + 1)
        double = RunnableLambda(lambda x: x * 2)

        seq = add1 | double
        # 3 + 1 = 4, 4 * 2 = 8
        assert seq.invoke(3) == 8

    def test_three_step_pipe(self):
        """多步管道"""
        from simple_langchain.runnable import RunnableLambda

        add1 = RunnableLambda(lambda x: x + 1)
        double = RunnableLambda(lambda x: x * 2)
        to_str = RunnableLambda(lambda x: f"result={x}")

        seq = add1 | double | to_str
        # 3 + 1 = 4, 4 * 2 = 8, "result=8"
        assert seq.invoke(3) == "result=8"

    def test_sequence_batch(self):
        """RunnableSequence 支持 batch"""
        from simple_langchain.runnable import RunnableLambda

        add1 = RunnableLambda(lambda x: x + 1)
        double = RunnableLambda(lambda x: x * 2)

        seq = add1 | double
        assert seq.batch([1, 2, 3]) == [4, 6, 8]

    def test_sequence_is_runnable(self):
        """RunnableSequence 也是 Runnable，可以继续管道"""
        from simple_langchain.runnable import RunnableLambda

        a = RunnableLambda(lambda x: x + 1)
        b = RunnableLambda(lambda x: x * 2)
        c = RunnableLambda(lambda x: str(x))

        # (a | b) | c
        seq = (a | b) | c
        assert seq.invoke(3) == "8"

    def test_pipe_with_dict_input(self):
        """管道中间传递 dict"""
        from simple_langchain.runnable import RunnableLambda

        extract = RunnableLambda(lambda d: d["name"])
        upper = RunnableLambda(lambda s: s.upper())

        seq = extract | upper
        assert seq.invoke({"name": "hello"}) == "HELLO"

    def test_pipe_operator_type_check(self):
        """| 右边必须是 Runnable"""
        from simple_langchain.runnable import RunnableLambda

        r = RunnableLambda(lambda x: x)
        with pytest.raises(TypeError):
            r | "not a runnable"
