"""平台判责 Agent。

提供多种决策策略，便于在相同仿真环境下对比长期收益：
- RuleBasedAgent：基于证据强度与金额的规则基线
- ProConsumerAgent：偏买家策略
- ProMerchantAgent：偏商家策略
- RandomAgent：随机策略（用于对照）
- LLMAgent：调用 OpenAI 兼容接口，JSON 输出（可插拔）

注意：所有 Agent 只读取 `DisputeCase` 中的平台可见字段
（订单、投诉、商家回应、聊天记录、证据），不应读取 `true_*` 与
`*_strategy` 等 ground truth 字段。
"""
from __future__ import annotations

import os
import random
from abc import ABC, abstractmethod

from .models import AgentDecision, DisputeCase, Liability
from .prompting import (
    SYSTEM_PROMPT,
    TOOL_LOOP_SYSTEM_PROMPT,
    build_case_prompt,
    build_full_prompt,
    build_tool_loop_user_prompt,
    parse_action,
    parse_agent_response,
)
from .tools import execute_tool, format_tool_definitions, format_tool_result_message


class BasePlatformAgent(ABC):
    """平台判责 Agent 基类。"""

    name: str = "base"

    @abstractmethod
    def decide(self, case: DisputeCase) -> AgentDecision:
        """根据平台可见信息输出判责、赔付与是否人工升级。"""

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}({self.name})"


def _avg_evidence_strength(case: DisputeCase, party: str) -> float:
    """计算某一方证据的平均证明力（规则 Agent 的简化证据审核能力）。"""
    if party == "buyer":
        items = case.buyer_evidence
    else:
        items = case.merchant_evidence
    if not items:
        return 0.5
    return sum(item.strength for item in items) / len(items)


def _chat_hostility(chat_log) -> float:
    """聊天记录中的情绪激烈程度 0~1（用于判断是否人工升级）。"""
    hot_words = ["投诉", "差评", "曝光", "报警", "起诉", "欺骗", "恶劣", "推卸", "拒", "不退"]
    if not chat_log:
        return 0.0
    hits = sum(1 for msg in chat_log for w in hot_words if w in msg.content)
    return min(1.0, hits / (len(chat_log) * 1.5))


class RuleBasedAgent(BasePlatformAgent):
    """规则基线 Agent。

    思路：比较双方证据强度 → 判定责任方向；按责任方向与订单金额计算赔付；
    证据冲突、金额大、情绪激烈时人工升级。
    """

    name = "规则基线"

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def decide(self, case: DisputeCase) -> AgentDecision:
        buyer_ev = _avg_evidence_strength(case, "buyer")
        merchant_ev = _avg_evidence_strength(case, "merchant")
        diff = buyer_ev - merchant_ev
        amount = case.order_amount
        request = case.buyer_requested_amount
        hostility = _chat_hostility(case.chat_log)

        # ---- 责任判定 ----
        if diff > 0.25:
            liability = Liability.MERCHANT
        elif diff < -0.25:
            liability = Liability.BUYER
        elif abs(diff) <= 0.12:
            liability = Liability.NONE
        else:
            liability = Liability.SPLIT

        # ---- 赔付方案 ----
        if liability == Liability.MERCHANT:
            if case.claim_type.value == "未收到货":
                comp = min(request, amount)
            else:
                comp = min(request, amount) * 0.8
        elif liability == Liability.SPLIT:
            comp = min(request, amount) * 0.4
        else:
            comp = 0.0

        # ---- 人工升级 ----
        evidence_conflict = abs(diff) <= 0.12
        escalate = (
            evidence_conflict
            or request > max(amount * 1.2, 500.0)
            or hostility > 0.6
            or (liability == Liability.NONE and request > 200.0)
        )

        reasons = {
            Liability.MERCHANT: f"买家证据证明力({buyer_ev:.2f})显著强于商家({merchant_ev:.2f})",
            Liability.BUYER: f"商家证据证明力({merchant_ev:.2f})显著强于买家({buyer_ev:.2f})",
            Liability.SPLIT: f"双方证据证明力接近(买家{buyer_ev:.2f}/商家{merchant_ev:.2f})，认定共担",
            Liability.NONE: "双方证据均不足以单独认定责任",
        }
        reason = reasons[liability]
        if escalate:
            reason += "；证据冲突/金额较大/情绪激烈，建议人工升级"

        return AgentDecision(
            liability=liability,
            compensation=round(comp, 2),
            escalate=escalate,
            reason=reason,
        )


class ProConsumerAgent(BasePlatformAgent):
    """偏买家策略：优先支持买家，控制单笔赔付上限为订单金额。"""

    name = "偏买家"

    def decide(self, case: DisputeCase) -> AgentDecision:
        comp = min(case.buyer_requested_amount, case.order_amount)
        escalate = case.buyer_requested_amount > case.order_amount * 1.2
        return AgentDecision(
            liability=Liability.MERCHANT,
            compensation=round(comp, 2),
            escalate=escalate,
            reason="用户满意度优先，倾向支持买家合理诉求",
        )


class ProMerchantAgent(BasePlatformAgent):
    """偏商家策略：优先保护商家，通常不赔付。"""

    name = "偏商家"

    def decide(self, case: DisputeCase) -> AgentDecision:
        return AgentDecision(
            liability=Liability.BUYER,
            compensation=0.0,
            escalate=False,
            reason="商家经营成本优先，证据不足不支持赔付",
        )


class RandomAgent(BasePlatformAgent):
    """随机策略（对照实验用）。"""

    name = "随机"

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def decide(self, case: DisputeCase) -> AgentDecision:
        liability = self.rng.choice(list(Liability))
        if liability in (Liability.MERCHANT, Liability.SPLIT):
            comp = case.buyer_requested_amount * self.rng.uniform(0.2, 1.0)
        else:
            comp = 0.0
        return AgentDecision(
            liability=liability,
            compensation=round(min(comp, case.order_amount), 2),
            escalate=self.rng.random() < 0.2,
            reason="随机基线",
        )


class LocalModelAgent(BasePlatformAgent):
    """加载本地训练产物（SFT/RL 后的 checkpoint）做判责。

    依赖 transformers + torch，延迟导入。生成失败时回退 RuleBasedAgent。
    """

    name = "本地模型"

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        max_new_tokens: int = 256,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("LocalModelAgent 需要安装 torch 和 transformers") from exc

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype="auto", trust_remote_code=True
        )
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(device).eval()
        self.max_new_tokens = max_new_tokens
        self.fallback = RuleBasedAgent()

    def decide(self, case: DisputeCase) -> AgentDecision:
        try:
            return self._generate(case)
        except Exception as exc:  # noqa: BLE001
            decision = self.fallback.decide(case)
            decision.reason = f"本地模型生成失败({exc})，回退规则基线。{decision.reason}"
            return decision

    def _generate(self, case: DisputeCase) -> AgentDecision:
        import torch

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_case_prompt(case)},
        ]
        if getattr(self.tokenizer, "chat_template", None) is not None:
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self.device)
        else:
            input_ids = self.tokenizer(
                build_full_prompt(case), return_tensors="pt"
            ).input_ids.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        generated = outputs[0][input_ids.shape[1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        data = parse_agent_response(text)

        liability_map = {
            "商家责任": Liability.MERCHANT,
            "买家责任": Liability.BUYER,
            "双方共担": Liability.SPLIT,
            "无法认定": Liability.NONE,
        }
        if not data:
            raise ValueError(f"无法解析模型输出：{text[:80]}")
        liability = liability_map.get(data.get("liability", ""), Liability.NONE)
        comp = float(data.get("compensation", 0.0))
        escalate = bool(data.get("escalate", False))
        reason = str(data.get("reason", "本地模型判定"))[:200]
        return AgentDecision(
            liability=liability,
            compensation=round(max(0.0, comp), 2),
            escalate=escalate,
            reason=reason,
        )


class ToolLoopAgent(BasePlatformAgent):
    """轻量自研多步工具循环 Agent。

    通过 OpenAI 兼容接口（vLLM）驱动模型：
    1. 模型每步输出一个 JSON：tool_call 或 final；
    2. tool_call 由仿真工具层执行，结果回填；
    3. final 解析为 AgentDecision。

    与 SFT 训练时的消息格式、工具定义、工具结果格式完全一致。
    """

    name = "多步工具Agent"

    def __init__(
        self,
        model: str = "dispute-7b",
        api_key: str = "EMPTY",
        base_url: str = "http://localhost:8000/v1",
        max_steps: int = 6,
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or "EMPTY"
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.max_steps = max_steps
        self.temperature = temperature
        self.fallback = RuleBasedAgent()

    def decide(self, case: DisputeCase) -> AgentDecision:
        try:
            return self._run_loop(case)
        except Exception as exc:  # noqa: BLE001
            decision = self.fallback.decide(case)
            decision.reason = f"多步工具循环失败({exc})，回退规则基线。{decision.reason}"
            return decision

    def _run_loop(self, case: DisputeCase) -> AgentDecision:
        import openai

        client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
        messages: list[dict] = [
            {"role": "system", "content": TOOL_LOOP_SYSTEM_PROMPT},
            {"role": "user", "content": build_tool_loop_user_prompt(case, format_tool_definitions())},
        ]
        total_tool_cost = 0.0

        for _ in range(self.max_steps):
            resp = client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=messages,
                response_format={"type": "json_object"},
            )
            text = resp.choices[0].message.content or ""
            action = parse_action(text)
            if not action:
                raise ValueError(f"无法解析模型输出：{text[:100]}")

            if action.get("action") == "final":
                decision = self._action_to_decision(action)
                decision.tool_cost = round(total_tool_cost, 2)
                return decision

            if action.get("action") == "tool_call":
                tool_name = str(action.get("tool", ""))
                args = action.get("arguments", {})
                if not isinstance(args, dict):
                    args = {}
                result = execute_tool(case, tool_name, args)
                if not result.success:
                    raise ValueError(f"未知工具：{tool_name}")
                total_tool_cost += result.cost
                messages.append({"role": "assistant", "content": text})
                # 用 user 角色回填工具结果，兼容 OpenAI/vLLM 与 ms-swift
                messages.append({"role": "user", "content": format_tool_result_message(result)})
                continue

            raise ValueError(f"未知动作类型：{action.get('action')}")

        raise RuntimeError(f"超过最大步数 {self.max_steps}")

    @staticmethod
    def _action_to_decision(action: dict) -> AgentDecision:
        liability_map = {
            "商家责任": Liability.MERCHANT,
            "买家责任": Liability.BUYER,
            "双方共担": Liability.SPLIT,
            "无法认定": Liability.NONE,
        }
        liability = liability_map.get(action.get("liability", ""), Liability.NONE)
        comp = float(action.get("compensation", 0.0))
        escalate = bool(action.get("escalate", False))
        reason = str(action.get("reason", "多步工具判定"))[:200]
        return AgentDecision(
            liability=liability,
            compensation=round(max(0.0, comp), 2),
            escalate=escalate,
            reason=reason,
        )


class LLMAgent(BasePlatformAgent):
    """调用 OpenAI 兼容 LLM 的判责 Agent。

    通过环境变量 `OPENAI_API_KEY` 提供密钥；模型可注入。
    若 openai 未安装、未配置或调用失败，自动回退到 RuleBasedAgent。
    """

    name = "LLM"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.temperature = temperature
        self.fallback = RuleBasedAgent()

    def decide(self, case: DisputeCase) -> AgentDecision:
        try:
            return self._call_llm(case)
        except Exception as exc:  # noqa: BLE001
            decision = self.fallback.decide(case)
            decision.reason = f"LLM 调用失败({exc})，回退规则基线。{decision.reason}"
            return decision

    # ---------- LLM 调用 ----------
    def _call_llm(self, case: DisputeCase) -> AgentDecision:
        import openai

        if not self.api_key:
            raise RuntimeError("未配置 OPENAI_API_KEY")

        client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url or None)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_case_prompt(case)},
        ]
        resp = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=messages,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content or ""
        data = parse_agent_response(content)

        liability_map = {
            "商家责任": Liability.MERCHANT,
            "买家责任": Liability.BUYER,
            "双方共担": Liability.SPLIT,
            "无法认定": Liability.NONE,
        }
        liability = liability_map.get(data.get("liability", ""), Liability.NONE)
        comp = float(data.get("compensation", 0.0))
        escalate = bool(data.get("escalate", False))
        reason = str(data.get("reason", "LLM 判定"))[:200]

        return AgentDecision(
            liability=liability,
            compensation=round(comp, 2),
            escalate=escalate,
            reason=reason,
        )


#: 方便外部直接使用的默认 Agent 集合
DEFAULT_AGENTS: dict[str, BasePlatformAgent] = {
    "规则基线": RuleBasedAgent(),
    "偏买家": ProConsumerAgent(),
    "偏商家": ProMerchantAgent(),
    "随机": RandomAgent(seed=42),
}
