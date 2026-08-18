from dataclasses import dataclass

from dispute_agent.data.language_enrichment import (
    EnrichmentResult,
    LanguageRewrite,
    public_payload_hash,
    public_payload,
)
from dispute_agent.data.splits import DEFAULT_COUNTS
from scripts.generate_data import _build_rows


@dataclass
class FakeRunner:
    calls: int = 0

    def enrich(self, instance, *, seed, ratio):
        self.calls += 1
        source = public_payload(instance)
        rewrite = LanguageRewrite(
            buyer_claim=source["buyer_claim"] + "（风格增强）",
            merchant_response=source["merchant_response"] + "（风格增强）",
            chat_log=[f"{line}（风格增强）" for line in source["chat_log"]],
            evidence_descriptions={
                item["evidence_id"]: item["description"] + "（风格增强）"
                for item in source["evidence"]
            },
        )
        return EnrichmentResult(
            status="enriched",
            style="platform_formal",
            rewrite=rewrite,
            source_public_hash=public_payload_hash(source),
        )


def _small_counts():
    return dict(DEFAULT_COUNTS)


def test_enrichment_is_applied_before_sft_and_grpo_rendering():
    runner = FakeRunner()
    rows = _build_rows(
        seed=20260817,
        counts=_small_counts(),
        fixture_size=24,
        enricher=runner,
        language_ratio=1.0,
    )

    sft_row = rows["sft_train"][0]
    grpo_row = rows["grpo_train"][0]
    assert "（风格增强）" in sft_row["messages"][1]["content"]
    assert "（风格增强）" in grpo_row["observation"]["buyer_claim"]
    assert "messages" not in grpo_row
    assert runner.calls == 24


def test_disabled_enrichment_does_not_call_a_runner():
    rows = _build_rows(seed=20260817, counts=_small_counts(), fixture_size=24)

    assert sum(len(items) for items in rows.values()) == 24
    assert all(
        "（风格增强）" not in str(row)
        for items in rows.values()
        for row in items
    )
