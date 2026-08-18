import pytest
from types import SimpleNamespace
import json

from dispute_agent.data.generator import generate_fact_instances
from dispute_agent.data.language_enrichment import (
    LanguageRewrite,
    LanguageEnrichmentRunner,
    JsonlRewriteCache,
    RewriteValidationError,
    STYLE_PROFILES,
    apply_rewrite,
    public_payload,
    style_for_case,
    validate_rewrite,
)


@pytest.fixture
def instance():
    return generate_fact_instances(seed=20260817, n=1)[0]


def _candidate(source: dict) -> dict:
    return {
        "buyer_claim": source["buyer_claim"] + " 麻烦平台核实。",
        "merchant_response": source["merchant_response"] + " 我们愿意配合处理。",
        "chat_log": [f"{line} 请核实。" for line in source["chat_log"]],
        "evidence_descriptions": {
            item["evidence_id"]: item["description"] + " 记录待进一步核对。"
            for item in source["evidence"]
        },
    }


def test_public_payload_excludes_hidden_truth_and_style_is_deterministic(instance):
    payload = public_payload(instance)

    assert "ground_truth" not in payload
    assert "true_liability" not in str(payload)
    assert style_for_case(instance.case_id, 20260817) == style_for_case(instance.case_id, 20260817)
    assert style_for_case(instance.case_id, 20260817) in STYLE_PROFILES


def test_valid_rewrite_preserves_public_shape_and_hidden_truth(instance):
    source = public_payload(instance)
    candidate = validate_rewrite(source, _candidate(source))

    assert isinstance(candidate, LanguageRewrite)
    hidden_before = instance.ground_truth.model_dump(mode="json")
    apply_rewrite(instance, candidate)

    assert instance.observation.buyer_claim == candidate.buyer_claim
    assert instance.observation.evidence[0].description == candidate.evidence_descriptions[
        instance.observation.evidence[0].evidence_id
    ]
    assert instance.ground_truth.model_dump(mode="json") == hidden_before


def test_rewrite_rejects_extra_keys_and_hidden_decision_language(instance):
    source = public_payload(instance)
    candidate = _candidate(source)
    candidate["unexpected"] = "must be rejected"
    with pytest.raises(RewriteValidationError, match="schema"):
        validate_rewrite(source, candidate)

    candidate = _candidate(source)
    candidate["buyer_claim"] = "责任由商家承担。"
    with pytest.raises(RewriteValidationError, match="decision"):
        validate_rewrite(source, candidate)


def test_rewrite_rejects_dropped_numeric_anchor(instance):
    source = public_payload(instance)
    source["buyer_claim"] = "商品在 2026-08-01 收到后出现破损。"
    candidate = _candidate(source)
    candidate["buyer_claim"] = "商品收到后出现破损。"

    with pytest.raises(RewriteValidationError, match="anchor"):
        validate_rewrite(source, candidate)


class FakeDeepSeekClient:
    def __init__(self, content=None, error=None):
        self.content = content
        self.error = error
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
        )


def test_deepseek_runner_uses_public_json_and_returns_valid_rewrite(instance):
    source = public_payload(instance)
    client = FakeDeepSeekClient(content=json.dumps(_candidate(source), ensure_ascii=False))
    runner = LanguageEnrichmentRunner(client=client, model="deepseek-v4-flash")

    result = runner.enrich(instance, seed=20260817, force=True)

    assert result.status == "enriched"
    assert result.rewrite is not None
    assert len(client.calls) == 1
    request_text = json.dumps(client.calls[0], ensure_ascii=False)
    assert "true_liability" not in request_text
    assert "reasonable_compensation_range" not in request_text
    assert client.calls[0]["response_format"] == {"type": "json_object"}
    assert client.calls[0]["extra_body"] == {"thinking": {"type": "disabled"}}


def test_deepseek_runner_cache_hit_avoids_second_api_call(instance, tmp_path):
    source = public_payload(instance)
    content = json.dumps(_candidate(source), ensure_ascii=False)
    first_client = FakeDeepSeekClient(content=content)
    cache_path = tmp_path / "language.jsonl"
    first = LanguageEnrichmentRunner(
        client=first_client,
        model="deepseek-v4-flash",
        cache=JsonlRewriteCache(cache_path),
    )
    assert first.enrich(instance, seed=20260817, force=True).status == "enriched"
    first.flush_cache()

    second_client = FakeDeepSeekClient(error=AssertionError("cache miss"))
    second = LanguageEnrichmentRunner(
        client=second_client,
        model="deepseek-v4-flash",
        cache=JsonlRewriteCache(cache_path),
    )
    result = second.enrich(instance, seed=20260817, force=True)

    assert result.status == "cache_hit"
    assert second_client.calls == []


def test_cached_fallback_is_retried_when_api_becomes_available(instance, tmp_path):
    cache_path = tmp_path / "language.jsonl"
    unavailable = LanguageEnrichmentRunner(
        client=None,
        model="deepseek-v4-flash",
        cache=JsonlRewriteCache(cache_path),
    )
    assert unavailable.enrich(instance, seed=20260817, force=True).status == "fallback"
    unavailable.flush_cache()

    source = public_payload(instance)
    available_client = FakeDeepSeekClient(
        content=json.dumps(_candidate(source), ensure_ascii=False)
    )
    available = LanguageEnrichmentRunner(
        client=available_client,
        model="deepseek-v4-flash",
        cache=JsonlRewriteCache(cache_path),
    )

    assert available.enrich(instance, seed=20260817, force=True).status == "enriched"
    assert len(available_client.calls) == 1


@pytest.mark.parametrize(
    "client, expected_error",
    [
        (None, "missing_api_key"),
        (FakeDeepSeekClient(content="not-json"), "invalid_json"),
        (FakeDeepSeekClient(error=RuntimeError("network")), "RuntimeError"),
    ],
)
def test_deepseek_runner_falls_back_without_losing_instance(client, expected_error, instance):
    runner = LanguageEnrichmentRunner(client=client, model="deepseek-v4-flash")

    result = runner.enrich(instance, seed=20260817, force=True)

    assert result.status == "fallback"
    assert result.rewrite is None
    assert expected_error in (result.error or "")
