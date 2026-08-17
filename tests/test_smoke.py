"""冒烟测试：验证核心模块可运行且收益核算有区分度。"""
import unittest

from dispute_agent.case_generator import CaseGenerator
from dispute_agent.data_generation import (
    case_from_dict,
    case_to_dict,
    generate_rl_data,
    generate_sft_data,
    load_cases_from_jsonl,
    make_sft_tool_example,
)
from dispute_agent.environment import DisputeEnvironment
from dispute_agent.evaluate import evaluate_agents
from dispute_agent.oracle import OracleAgent
from dispute_agent.platform_agent import (
    ProConsumerAgent,
    ProMerchantAgent,
    RuleBasedAgent,
    ToolLoopAgent,
)
from dispute_agent.reward import RewardEngine, compute_reward
from dispute_agent.tools import execute_tool


class SmokeTest(unittest.TestCase):
    def test_generate_case(self):
        gen = CaseGenerator(seed=0)
        case = gen.generate()
        self.assertTrue(case.case_id)
        self.assertGreater(case.order_amount, 0)
        self.assertGreaterEqual(case.buyer_requested_amount, 0)
        self.assertTrue(case.buyer_claim)
        self.assertTrue(case.merchant_response)
        self.assertTrue(case.chat_log)

    def test_step_and_report(self):
        env = DisputeEnvironment(seed=0)
        agent = RuleBasedAgent()
        record = env.step(agent)
        self.assertIsNotNone(record.decision)
        self.assertGreaterEqual(record.outcome.buyer_satisfaction, 0.0)
        self.assertLessEqual(record.outcome.buyer_satisfaction, 1.0)

        report = env.run(agent, n_cases=50)
        self.assertEqual(report.n_cases, 50)
        self.assertGreater(report.avg_long_term_value, -10_000)

    def test_strategy_comparison_runs(self):
        reports = evaluate_agents(
            {
                "规则基线": RuleBasedAgent(),
                "偏买家": ProConsumerAgent(),
                "偏商家": ProMerchantAgent(),
            },
            n_cases=100,
            seed=1,
        )
        self.assertEqual(len(reports), 3)
        values = [r.avg_long_term_value for r in reports]
        self.assertGreater(max(values), min(values))  # 策略之间有差异

    def test_oracle_and_reward(self):
        gen = CaseGenerator(seed=0)
        case = gen.generate()
        oracle = OracleAgent()
        decision = oracle.decide(case)
        reward = compute_reward(case, decision)
        self.assertGreater(reward, 0.0)

        engine = RewardEngine.from_cases([case])
        rl_example = {"prompt": f"案例编号：{case.case_id}\n", "case_id": case.case_id}
        single = engine.single_reward(rl_example["prompt"], '{"liability": "商家责任", "compensation": 0, "escalate": false, "reason": "test"}')
        self.assertGreaterEqual(single, -1.0)

    def test_data_generation_roundtrip(self):
        # 直接测试序列化 roundtrip
        gen = CaseGenerator(seed=0)
        case = gen.generate()
        restored = case_from_dict(case_to_dict(case))
        self.assertEqual(restored.case_id, case.case_id)
        self.assertEqual(restored.true_liability, case.true_liability)
        self.assertEqual(restored.order_amount, case.order_amount)

    def test_generate_rl_data_to_tmp(self):
        import shutil
        import tempfile
        import os
        from pathlib import Path

        tmp = Path(tempfile.gettempdir()) / f"pytest-rl-{os.getpid()}"
        tmp.mkdir(parents=True, exist_ok=False)
        try:
            cases = generate_rl_data(5, seed=0, output_dir=str(tmp))
            self.assertEqual(len(cases), 5)
            loaded = load_cases_from_jsonl(str(tmp / "rl_cases.jsonl"))
            self.assertEqual(len(loaded), 5)
            self.assertEqual(loaded[0].case_id, cases[0].case_id)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_tools_and_multi_step_sft(self):
        gen = CaseGenerator(seed=0)
        case = gen.generate()
        # 工具执行可复现
        r1 = execute_tool(case, "check_buyer_history", {"buyer_id": case.buyer_id})
        r2 = execute_tool(case, "check_buyer_history", {"buyer_id": case.buyer_id})
        self.assertEqual(r1.content, r2.content)
        self.assertGreaterEqual(r1.cost, 0.0)

        # 多步 SFT 样本结构正确
        example = make_sft_tool_example(case, OracleAgent())
        self.assertGreaterEqual(len(example["messages"]), 3)
        roles = [m["role"] for m in example["messages"]]
        self.assertEqual(roles[0], "system")
        self.assertEqual(roles[1], "user")
        self.assertEqual(roles[-1], "assistant")
        self.assertIn("final", example["messages"][-1]["content"])

    def test_tool_loop_action_parser(self):
        action = ToolLoopAgent._action_to_decision({
            "action": "final",
            "liability": "商家责任",
            "compensation": 30.0,
            "escalate": False,
            "reason": "测试",
        })
        self.assertEqual(action.liability.value, "商家责任")
        self.assertEqual(action.compensation, 30.0)

    def test_generate_sft_data_to_tmp(self):
        import shutil
        import tempfile
        import os
        from pathlib import Path

        tmp = Path(tempfile.gettempdir()) / f"pytest-sft-{os.getpid()}"
        tmp.mkdir(parents=True, exist_ok=False)
        try:
            path = str(tmp / "sft.jsonl")
            cases = generate_sft_data(3, seed=0, output_path=path, style="multi_tool")
            self.assertEqual(len(cases), 3)
            with open(path, "r", encoding="utf-8") as f:
                lines = [line for line in f if line.strip()]
            self.assertEqual(len(lines), 3)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
