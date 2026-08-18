"""Optional public-language enrichment for deterministic dispute instances.

The module keeps all hidden truth outside the rewrite payload. The API client and
cache are defined here as well so callers can keep the default data path entirely
offline and deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from dispute_agent.data.generator import FactInstance
from dispute_agent.data.validators import HIDDEN_KEYWORDS


PROMPT_VERSION = "deepseek-public-rewrite-v1"
STYLE_PROFILES = (
    "platform_formal",
    "buyer_concise",
    "buyer_timeline",
    "buyer_emotional_factual",
    "merchant_cooperative",
    "merchant_defensive",
    "merchant_procedural",
    "chat_fragmented",
)
STYLE_INSTRUCTIONS = {
    "platform_formal": "使用平台工单的正式、客观、简洁表达。",
    "buyer_concise": "使用买家简短直接的口语表达，但保留事实。",
    "buyer_timeline": "使用按时间顺序叙述的买家表达，突出事件先后。",
    "buyer_emotional_factual": "允许买家有明确情绪，但每句话仍必须基于已有事实。",
    "merchant_cooperative": "使用商家配合核查、愿意提供材料的表达。",
    "merchant_defensive": "使用商家辩解或强调自身操作的表达，但不要新增事实。",
    "merchant_procedural": "使用商家客服的流程化、规范化表达。",
    "chat_fragmented": "使用短句、口语、省略主语的聊天记录风格。",
}
DECISION_PATTERNS = (
    "true_liability",
    "true_loss",
    "reasonable_compensation_range",
    "should_escalate",
    "buyer_strategy",
    "merchant_strategy",
    "tool_information_value",
    "evidence_authenticity",
    "商家责任",
    "买家责任",
    "平台责任",
    "责任由商家承担",
    "责任由买家承担",
    "商家应承担",
    "买家应承担",
    "责任归属",
    "应赔偿",
    "应赔付",
    "赔偿金额",
    "赔付金额",
    "升级人工",
    "人工复核",
    "责任判定",
    "责任认定",
    "建议升级人工",
    "建议人工复核",
    "最终判定",
)
NUMBER_PATTERN = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?:%|元|天)?")


class RewriteValidationError(ValueError):
    """Raised when a public rewrite changes facts or exposes a decision."""


class LanguageRewrite(BaseModel):
    model_config = ConfigDict(extra="forbid")

    buyer_claim: str = Field(min_length=1)
    merchant_response: str = Field(min_length=1)
    chat_log: list[str] = Field(min_length=1)
    evidence_descriptions: dict[str, str]


@dataclass(frozen=True)
class EnrichmentResult:
    status: str
    style: str
    rewrite: LanguageRewrite | None
    source_public_hash: str
    error: str | None = None


class JsonlRewriteCache:
    """Small cache whose records contain public text only."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._records: dict[str, dict[str, Any]] = {}
        if self.path is not None and self.path.is_file():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict) and isinstance(record.get("key"), str):
                    self._records[record["key"]] = record

    def get(self, key: str) -> dict[str, Any] | None:
        return self._records.get(key)

    def put(self, key: str, record: dict[str, Any]) -> None:
        self._records[key] = {"key": key, **record}

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        lines = [
            json.dumps(self._records[key], ensure_ascii=False, sort_keys=True)
            for key in sorted(self._records)
        ]
        temporary.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        temporary.replace(self.path)


class ChatCompletionsClient(Protocol):
    chat: Any


def public_payload_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class LanguageEnrichmentRunner:
    """Cache-first wrapper around an OpenAI-compatible DeepSeek client."""

    def __init__(
        self,
        *,
        client: ChatCompletionsClient | None,
        model: str,
        cache: JsonlRewriteCache | None = None,
        prompt_version: str = PROMPT_VERSION,
        unavailable_error: str = "missing_api_key",
    ) -> None:
        self.client = client
        self.model = model
        self.cache = cache or JsonlRewriteCache()
        self.prompt_version = prompt_version
        self.unavailable_error = unavailable_error
        self.stats = {
            "requested": 0,
            "succeeded": 0,
            "cache_hit": 0,
            "fallback": 0,
            "skipped": 0,
        }

    def _cache_key(self, source_hash: str, style: str) -> str:
        raw = f"{source_hash}:{style}:{self.prompt_version}:{self.model}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _messages(self, source: dict[str, Any], style: str) -> list[dict[str, str]]:
        system = (
            "你是电商纠纷公开文本改写器。请只根据公开信息改写语言风格，输出 JSON。"
            "不要推断、补充或输出责任、赔付、升级人工等结论；不要新增事实、金额、时间或证据。"
            f"当前风格：{STYLE_INSTRUCTIONS[style]}"
            "JSON 必须包含 buyer_claim、merchant_response、chat_log、evidence_descriptions 四个字段。"
        )
        user = json.dumps(
            {"style": style, "public_observation": source},
            ensure_ascii=False,
            sort_keys=True,
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _fallback(self, style: str, source_hash: str, error: str) -> EnrichmentResult:
        self.stats["fallback"] += 1
        return EnrichmentResult("fallback", style, None, source_hash, error)

    def enrich(
        self,
        instance: FactInstance,
        *,
        seed: int,
        ratio: float = 0.5,
        force: bool = False,
    ) -> EnrichmentResult:
        source = public_payload(instance)
        source_hash = public_payload_hash(source)
        style = style_for_case(instance.case_id, seed)
        if not force:
            if not 0.0 <= ratio <= 1.0:
                raise ValueError("language enrichment ratio must be between 0 and 1")
            selector = int(hashlib.sha256(f"{seed}:{instance.case_id}:language-enrich".encode()).hexdigest()[:8], 16) / 0x100000000
            if selector >= ratio:
                self.stats["skipped"] += 1
                return EnrichmentResult("skipped", style, None, source_hash)

        self.stats["requested"] += 1
        key = self._cache_key(source_hash, style)
        cached = self.cache.get(key)
        if cached is not None:
            if cached.get("status") == "success":
                try:
                    rewrite = validate_rewrite(source, cached["rewrite"])
                except (KeyError, RewriteValidationError) as exc:
                    return self._fallback(style, source_hash, f"invalid_cache: {exc}")
                self.stats["succeeded"] += 1
                self.stats["cache_hit"] += 1
                return EnrichmentResult("cache_hit", style, rewrite, source_hash)
            if self.client is None:
                return self._fallback(style, source_hash, str(cached.get("error", "cached_fallback")))

        if self.client is None:
            error = self.unavailable_error
            self.cache.put(key, {
                "source_public_hash": source_hash,
                "style": style,
                "prompt_version": self.prompt_version,
                "model": self.model,
                "status": "fallback",
                "error": error,
            })
            return self._fallback(style, source_hash, error)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self._messages(source, style),
                response_format={"type": "json_object"},
                extra_body={"thinking": {"type": "disabled"}},
                max_tokens=512,
            )
            content = response.choices[0].message.content
            if not isinstance(content, str) or not content.strip():
                raise ValueError("empty response")
            candidate = json.loads(content)
            rewrite = validate_rewrite(source, candidate)
        except json.JSONDecodeError as exc:
            error = f"invalid_json: {exc}"
            self.cache.put(key, {
                "source_public_hash": source_hash,
                "style": style,
                "prompt_version": self.prompt_version,
                "model": self.model,
                "status": "fallback",
                "error": error,
            })
            return self._fallback(style, source_hash, error)
        except RewriteValidationError as exc:
            error = f"invalid_rewrite: {exc}"
            self.cache.put(key, {
                "source_public_hash": source_hash,
                "style": style,
                "prompt_version": self.prompt_version,
                "model": self.model,
                "status": "fallback",
                "error": error,
            })
            return self._fallback(style, source_hash, error)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self.cache.put(key, {
                "source_public_hash": source_hash,
                "style": style,
                "prompt_version": self.prompt_version,
                "model": self.model,
                "status": "fallback",
                "error": error,
            })
            return self._fallback(style, source_hash, error)

        self.cache.put(key, {
            "source_public_hash": source_hash,
            "style": style,
            "prompt_version": self.prompt_version,
            "model": self.model,
            "status": "success",
            "rewrite": rewrite.model_dump(mode="json"),
            "rewrite_sha256": hashlib.sha256(
                json.dumps(rewrite.model_dump(mode="json"), ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest(),
        })
        self.stats["succeeded"] += 1
        return EnrichmentResult("enriched", style, rewrite, source_hash)

    def flush_cache(self) -> None:
        self.cache.save()


def build_deepseek_runner(
    *,
    cache_path: str | Path | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> LanguageEnrichmentRunner:
    """Build a lazy DeepSeek runner without importing OpenAI in offline mode."""
    resolved_model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    resolved_base_url = base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    resolved_key = api_key or os.getenv("DEEPSEEK_API_KEY")
    client = None
    unavailable_error = "missing_api_key"
    if resolved_key:
        try:
            from openai import OpenAI

            client = OpenAI(
                api_key=resolved_key,
                base_url=resolved_base_url,
                timeout=30.0,
                max_retries=1,
            )
        except Exception as exc:
            unavailable_error = f"client_init: {type(exc).__name__}: {exc}"
    return LanguageEnrichmentRunner(
        client=client,
        model=resolved_model,
        cache=JsonlRewriteCache(cache_path),
        unavailable_error=unavailable_error,
    )


def public_payload(instance: FactInstance) -> dict[str, Any]:
    """Return only public fields that may be shown to a paraphrasing model."""
    observation = instance.observation
    return {
        "case_id": observation.case_id,
        "order_id": observation.order_id,
        "buyer_id": observation.buyer_id,
        "merchant_id": observation.merchant_id,
        "item_name": observation.item_name,
        "order_amount": observation.order_amount,
        "claim_type": observation.claim_type,
        "buyer_claim": observation.buyer_claim,
        "buyer_requested_amount": observation.buyer_requested_amount,
        "merchant_response": observation.merchant_response,
        "chat_log": list(observation.chat_log),
        "evidence": [
            {
                "evidence_id": evidence.evidence_id,
                "type": evidence.type,
                "description": evidence.description,
                "source": evidence.source,
                "visible": evidence.visible,
            }
            for evidence in observation.evidence
        ],
    }


def style_for_case(case_id: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{case_id}:language-style".encode("utf-8")).digest()
    return STYLE_PROFILES[int.from_bytes(digest[:8], "big") % len(STYLE_PROFILES)]


def _text_from_rewrite(candidate: LanguageRewrite) -> str:
    return "\n".join([
        candidate.buyer_claim,
        candidate.merchant_response,
        *candidate.chat_log,
        *candidate.evidence_descriptions.values(),
    ])


def _source_text(source: dict[str, Any]) -> str:
    return "\n".join([
        str(source.get("buyer_claim", "")),
        str(source.get("merchant_response", "")),
        *(str(item) for item in source.get("chat_log", [])),
        *(str(item.get("description", "")) for item in source.get("evidence", [])),
    ])


def _validate_decision_language(text: str) -> None:
    lowered = text.lower()
    for keyword in (*HIDDEN_KEYWORDS, *DECISION_PATTERNS):
        if keyword.lower() in lowered:
            raise RewriteValidationError(f"decision or hidden field leaked: {keyword}")


def validate_rewrite(source: dict[str, Any], candidate: dict[str, Any] | LanguageRewrite) -> LanguageRewrite:
    """Validate a rewrite against the public source without using hidden truth."""
    try:
        parsed = candidate if isinstance(candidate, LanguageRewrite) else LanguageRewrite.model_validate(candidate)
    except ValidationError as exc:
        raise RewriteValidationError(f"schema validation failed: {exc}") from exc

    source_evidence = source.get("evidence", [])
    expected_ids = {str(item["evidence_id"]) for item in source_evidence}
    if set(parsed.evidence_descriptions) != expected_ids:
        raise RewriteValidationError("evidence IDs changed")
    if len(parsed.chat_log) != len(source.get("chat_log", [])):
        raise RewriteValidationError("chat log event count changed")

    source_numbers = set(NUMBER_PATTERN.findall(_source_text(source)))
    candidate_numbers = set(NUMBER_PATTERN.findall(_text_from_rewrite(parsed)))
    missing_numbers = source_numbers - candidate_numbers
    if missing_numbers:
        raise RewriteValidationError(f"numeric anchor missing: {sorted(missing_numbers)}")

    _validate_decision_language(_text_from_rewrite(parsed))
    return parsed


def apply_rewrite(instance: FactInstance, rewrite: LanguageRewrite) -> None:
    """Apply a validated rewrite in-place while leaving hidden truth untouched."""
    observation = instance.observation
    observation.buyer_claim = rewrite.buyer_claim
    observation.merchant_response = rewrite.merchant_response
    observation.chat_log = list(rewrite.chat_log)
    for evidence in observation.evidence:
        evidence.description = rewrite.evidence_descriptions[evidence.evidence_id]
