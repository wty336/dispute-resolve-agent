"""纠纷案例生成器。

按概率采样 ground truth（真实责任、真实损失、买卖双方博弈策略），
再据此生成订单、投诉、举证、聊天记录等“平台可见”信息。

这样既能仿真“买家夸大 / 商家推卸”的博弈，又能在评估时计算判责准确率。
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from .models import (
    BuyerStrategy,
    ChatMessage,
    ClaimType,
    DisputeCase,
    Evidence,
    Liability,
    MerchantStrategy,
    Party,
)

ITEMS = [
    ("蓝牙耳机", 199.0, True),
    ("电热水壶", 89.0, False),
    ("连衣裙", 259.0, False),
    ("机械键盘", 349.0, True),
    ("移动电源", 129.0, True),
    ("运动鞋", 399.0, False),
    ("智能手环", 249.0, True),
    ("保温杯", 79.0, False),
    ("台灯", 119.0, True),
    ("双肩包", 159.0, False),
]

CLAIM_TEXTS = {
    ClaimType.QUALITY_ISSUE: "商品存在质量问题，无法正常使用",
    ClaimType.NOT_AS_DESCRIBED: "实物与商品描述严重不符",
    ClaimType.DAMAGED: "收到商品时已破损/少件",
    ClaimType.NOT_RECEIVED: "物流显示签收但本人未收到货",
    ClaimType.AFTERSALES: "商家拒绝履行售后承诺",
}

BUYER_HONEST_TEMPLATES = [
    "我在{date}收到{goods}，{claim_text}，申请退款/赔付 {amount:.2f} 元。",
    "订单{order_id}的{goods}有问题：{claim_text}。实际损失约 {amount:.2f} 元，请平台处理。",
]

BUYER_EXAGGERATE_TEMPLATES = [
    "我在{date}收到{goods}，{claim_text}，严重影响了我的使用，还耽误了我好几天工作，要求赔偿 {amount:.2f} 元！",
    "订单{order_id}的{goods}太差了：{claim_text}。这已经不只是商品问题了，我的时间成本和心情损失也很大，索赔 {amount:.2f} 元。",
]

BUYER_FRAUD_TEMPLATES = [
    "我在{date}收到{goods}，{claim_text}，要求全额退款并赔偿 {amount:.2f} 元，否则差评加投诉。",
    "订单{order_id}的{goods}：{claim_text}。商家态度还很差，我要求平台严肃处理，赔付 {amount:.2f} 元。",
]

MERCHANT_HONEST_TEMPLATES = [
    "经核实，买家反馈属实，我们愿意承担相应责任，但希望平台按实际损失核定赔付金额。",
    "该单确实存在{claim_text}的情况，我们愿意配合处理，同意退款。",
]

MERCHANT_EVASIVE_TEMPLATES = [
    "商品发出前已检查完好，可能是物流途中或买家使用不当导致，我方不应承担全部责任。",
    "买家反馈的问题没有提供有效凭证，无法证明是我方责任；且买家已签收多日才反馈，不符合售后时效。",
]

MERCHANT_DENY_TEMPLATES = [
    "该买家是恶意投诉，我们发货记录和质检凭证齐全，商品没有任何问题，拒绝赔付。",
    "我们怀疑买家调包/故意损坏商品，已保留证据，不接受任何赔偿要求。",
]


@dataclass
class GeneratorConfig:
    """案例生成的概率参数。"""

    # 真实责任分布
    p_merchant: float = 0.45
    p_split: float = 0.25
    p_buyer: float = 0.15
    p_none: float = 0.15

    # 买家策略：按真实责任分别配置 [诚实, 夸大, 虚假]
    buyer_strategy_by_liability: dict[Liability, tuple[float, float, float]] | None = None
    # 商家策略：按真实责任分别配置 [诚实, 推卸, 虚假否认]
    merchant_strategy_by_liability: dict[Liability, tuple[float, float, float]] | None = None

    def __post_init__(self) -> None:
        if self.buyer_strategy_by_liability is None:
            self.buyer_strategy_by_liability = {
                Liability.MERCHANT: (0.45, 0.45, 0.10),
                Liability.SPLIT: (0.40, 0.50, 0.10),
                Liability.BUYER: (0.20, 0.25, 0.55),
                Liability.NONE: (0.15, 0.25, 0.60),
            }
        if self.merchant_strategy_by_liability is None:
            self.merchant_strategy_by_liability = {
                Liability.MERCHANT: (0.25, 0.50, 0.25),
                Liability.SPLIT: (0.35, 0.50, 0.15),
                Liability.BUYER: (0.75, 0.25, 0.00),
                Liability.NONE: (0.70, 0.30, 0.00),
            }


class CaseGenerator:
    """生成带 ground truth 的纠纷案例。"""

    def __init__(self, seed: int | None = None, config: GeneratorConfig | None = None) -> None:
        self.rng = random.Random(seed)
        self.config = config or GeneratorConfig()
        self._counter = 0

    # ---------- 公开接口 ----------
    def generate(self) -> DisputeCase:
        """生成一单纠纷案例。"""
        self._counter += 1
        item_name, order_amount, _fragile = self.rng.choice(ITEMS)
        claim_type = self.rng.choice(list(ClaimType))
        order_id = f"O{self.rng.randint(100000, 999999)}"
        buyer_id = f"B{self.rng.randint(100000, 999999)}"
        merchant_id = f"M{self.rng.randint(100000, 999999)}"

        true_liability = self._sample_liability()
        true_loss = self._sample_true_loss(true_liability, order_amount)
        buyer_strategy = self._sample_strategy(
            self.config.buyer_strategy_by_liability[true_liability], BuyerStrategy
        )
        merchant_strategy = self._sample_strategy(
            self.config.merchant_strategy_by_liability[true_liability], MerchantStrategy
        )

        requested = self._requested_amount(true_loss, order_amount, buyer_strategy, claim_type)
        buyer_claim = self._make_buyer_claim(
            item_name, claim_type, requested, buyer_strategy, true_loss, order_id
        )
        merchant_response = self._make_merchant_response(
            claim_type, merchant_strategy, true_liability
        )

        chat_log = self._make_chat_log(
            buyer_strategy, merchant_strategy, claim_type, requested
        )
        buyer_evidence, merchant_evidence = self._make_evidence(
            true_liability, buyer_strategy, merchant_strategy, claim_type
        )

        return DisputeCase(
            case_id=f"C{self._counter:06d}",
            order_id=order_id,
            buyer_id=buyer_id,
            merchant_id=merchant_id,
            item_name=item_name,
            order_amount=order_amount,
            claim_type=claim_type,
            buyer_claim=buyer_claim,
            buyer_requested_amount=requested,
            merchant_response=merchant_response,
            chat_log=chat_log,
            buyer_evidence=buyer_evidence,
            merchant_evidence=merchant_evidence,
            true_liability=true_liability,
            true_buyer_loss=true_loss,
            buyer_strategy=buyer_strategy,
            merchant_strategy=merchant_strategy,
        )

    def generate_batch(self, n: int) -> list[DisputeCase]:
        return [self.generate() for _ in range(n)]

    # ---------- 内部采样 ----------
    def _sample_liability(self) -> Liability:
        r = self.rng.random()
        cfg = self.config
        if r < cfg.p_merchant:
            return Liability.MERCHANT
        if r < cfg.p_merchant + cfg.p_split:
            return Liability.SPLIT
        if r < cfg.p_merchant + cfg.p_split + cfg.p_buyer:
            return Liability.BUYER
        return Liability.NONE

    def _sample_strategy(self, weights: tuple[float, float, float], enum_cls):
        """按权重采样博弈策略枚举。weights 顺序与枚举定义顺序一致。"""
        return self.rng.choices(list(enum_cls), weights=list(weights), k=1)[0]

    def _sample_true_loss(self, liability: Liability, order_amount: float) -> float:
        if liability == Liability.MERCHANT:
            return round(order_amount * self.rng.uniform(0.25, 1.0), 2)
        if liability == Liability.SPLIT:
            return round(order_amount * self.rng.uniform(0.10, 0.50), 2)
        return 0.0

    def _requested_amount(
        self,
        true_loss: float,
        order_amount: float,
        strategy: BuyerStrategy,
        claim_type: ClaimType,
    ) -> float:
        if strategy == BuyerStrategy.HONEST:
            amount = true_loss if true_loss > 0 else order_amount * 0.3
        elif strategy == BuyerStrategy.EXAGGERATE:
            base = true_loss if true_loss > 0 else order_amount * 0.3
            amount = base * self.rng.uniform(1.6, 3.0)
        else:  # FRAUD
            amount = order_amount * self.rng.uniform(0.4, 1.0)
        # 未收到货类型通常诉求接近订单金额
        if claim_type == ClaimType.NOT_RECEIVED:
            amount = max(amount, order_amount * 0.9)
        return round(min(amount, order_amount * 1.2), 2)

    # ---------- 文本生成 ----------
    def _make_buyer_claim(
        self,
        item_name: str,
        claim_type: ClaimType,
        requested: float,
        strategy: BuyerStrategy,
        true_loss: float,
        order_id: str,
    ) -> str:
        claim_text = CLAIM_TEXTS[claim_type]
        date = f"{self.rng.randint(1, 28)}日"
        if strategy == BuyerStrategy.HONEST:
            template = self.rng.choice(BUYER_HONEST_TEMPLATES)
        elif strategy == BuyerStrategy.EXAGGERATE:
            template = self.rng.choice(BUYER_EXAGGERATE_TEMPLATES)
        else:
            template = self.rng.choice(BUYER_FRAUD_TEMPLATES)
        return template.format(
            date=date, goods=item_name, claim_text=claim_text,
            amount=requested, order_id=order_id,
        )

    def _make_merchant_response(
        self,
        claim_type: ClaimType,
        strategy: MerchantStrategy,
        true_liability: Liability,
    ) -> str:
        claim_text = CLAIM_TEXTS[claim_type]
        if strategy == MerchantStrategy.HONEST:
            if true_liability in (Liability.BUYER, Liability.NONE):
                return "经核实，商品/服务不存在买家所述问题；但我方愿意配合平台提供完整凭证。"
            template = self.rng.choice(MERCHANT_HONEST_TEMPLATES)
        elif strategy == MerchantStrategy.EVASIVE:
            template = self.rng.choice(MERCHANT_EVASIVE_TEMPLATES)
        else:
            template = self.rng.choice(MERCHANT_DENY_TEMPLATES)
        return template.format(claim_text=claim_text)

    def _make_chat_log(
        self,
        buyer_strategy: BuyerStrategy,
        merchant_strategy: MerchantStrategy,
        claim_type: ClaimType,
        requested: float,
    ) -> list[ChatMessage]:
        claim_text = CLAIM_TEXTS[claim_type]
        buyer_open = f"亲，{claim_text}，我要申请赔付 {requested:.2f} 元，请尽快处理。"
        if buyer_strategy == BuyerStrategy.EXAGGERATE:
            buyer_open += " 这件事已经严重影响我的生活和心情了！"
        if buyer_strategy == BuyerStrategy.FRAUD:
            buyer_open += " 不处理我就差评、投诉到底！"

        if merchant_strategy == MerchantStrategy.HONEST:
            merchant_reply = "亲，非常抱歉给您带来不便。我们核实后会给您一个满意答复，请您稍等。"
        elif merchant_strategy == MerchantStrategy.EVASIVE:
            merchant_reply = "亲，商品发出前是完好的哦，可能是物流/使用环境导致，建议您先提供凭证。"
        else:
            merchant_reply = "亲，经核实商品无问题，您反馈的情况我方无法认可。"

        buyer_follow = "凭证我已经提交平台了，你们这是在推卸责任！"
        if buyer_strategy == BuyerStrategy.HONEST:
            buyer_follow = "好的，我已经补充了相关凭证，麻烦尽快核实。"
        merchant_close = "我们会配合平台核实，谢谢。" if merchant_strategy != MerchantStrategy.DENY else "请平台介入，我们拒绝私下协商。"

        return [
            ChatMessage(role="buyer", content=buyer_open),
            ChatMessage(role="merchant", content=merchant_reply),
            ChatMessage(role="buyer", content=buyer_follow),
            ChatMessage(role="merchant", content=merchant_close),
        ]

    def _make_evidence(
        self,
        true_liability: Liability,
        buyer_strategy: BuyerStrategy,
        merchant_strategy: MerchantStrategy,
        claim_type: ClaimType,
    ) -> tuple[list[Evidence], list[Evidence]]:
        buyer_evidence: list[Evidence] = []
        merchant_evidence: list[Evidence] = []

        # 买家证据强度：真实责任在商家/共担时更强；虚假投诉时弱
        if true_liability in (Liability.MERCHANT, Liability.SPLIT):
            buyer_strength = self.rng.uniform(0.6, 0.95)
        elif buyer_strategy == BuyerStrategy.FRAUD:
            buyer_strength = self.rng.uniform(0.05, 0.25)
        else:
            buyer_strength = self.rng.uniform(0.3, 0.55)

        # 商家证据强度：真实责任在买家/无责时更强；商家推卸/否认时通常有形式证据但证明力不足
        if true_liability in (Liability.BUYER, Liability.NONE):
            merchant_strength = self.rng.uniform(0.65, 0.95)
        elif merchant_strategy == MerchantStrategy.DENY:
            merchant_strength = self.rng.uniform(0.15, 0.40)
        elif merchant_strategy == MerchantStrategy.EVASIVE:
            merchant_strength = self.rng.uniform(0.35, 0.60)
        else:
            merchant_strength = self.rng.uniform(0.50, 0.75)

        if claim_type in (ClaimType.DAMAGED, ClaimType.NOT_RECEIVED):
            buyer_evidence.append(Evidence(
                type="开箱照片/视频" if claim_type == ClaimType.DAMAGED else "物流签收凭证",
                description="买家提供的商品破损照片" if claim_type == ClaimType.DAMAGED else "买家称未收到货的物流截图",
                party=Party.BUYER, strength=buyer_strength,
            ))
            merchant_evidence.append(Evidence(
                type="发货打包照片/物流签收底单",
                description="商家提供的打包完好照片与签收记录",
                party=Party.MERCHANT, strength=merchant_strength,
            ))
        else:
            buyer_evidence.append(Evidence(
                type="聊天记录/检测报告/实拍图",
                description="买家提供的聊天记录与商品实拍",
                party=Party.BUYER, strength=buyer_strength,
            ))
            merchant_evidence.append(Evidence(
                type="质检凭证/授权证书/聊天记录",
                description="商家提供的质检凭证与沟通记录",
                party=Party.MERCHANT, strength=merchant_strength,
            ))
        return buyer_evidence, merchant_evidence
