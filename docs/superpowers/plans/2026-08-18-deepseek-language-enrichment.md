# DeepSeek Language Enrichment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an optional DeepSeek API rewrite layer that increases public-language diversity without exposing hidden labels or changing the deterministic default data pipeline.

**Architecture:** Generate each `FactInstance` structurally as today, optionally send only its public text fields to a cache-first DeepSeek JSON client, validate invariants and leakage, and apply a valid rewrite before the existing SFT/GRPO renderers run. Any missing key, API error, invalid JSON, or invariant violation falls back to the original observation and is counted in the generation report.

**Tech Stack:** Python 3.11, Pydantic v2, OpenAI-compatible `openai` client, DeepSeek `deepseek-v4-flash`, JSON Output, deterministic SHA-256 cache keys, pytest fake-client tests.

---

### Task 1: Add the optional DeepSeek dependency and configuration contract

**Files:**
- Modify: `pyproject.toml`
- Modify: `configs/data.yaml`
- Test: `tests/test_project_contract.py`

- [ ] **Step 1: Write the failing configuration contract test**

Assert the optional `data` extra includes `openai`, the default configuration disables enrichment, and the default model is `deepseek-v4-flash`.

- [ ] **Step 2: Run the focused test and verify it fails**

Run: `python -m pytest tests/test_project_contract.py -q`

Expected: FAIL because the optional extra and configuration keys do not exist.

- [ ] **Step 3: Implement the minimal dependency/configuration change**

Add `data = ["openai>=2.0,<3"]` to optional dependencies and add:

```yaml
language_enrichment:
  enabled: false
  ratio: 0.5
  model: deepseek-v4-flash
  base_url: https://api.deepseek.com
  cache_file: language_enrichment_cache.jsonl
```

- [ ] **Step 4: Run the focused test and verify it passes**

Run: `python -m pytest tests/test_project_contract.py -q`

Expected: PASS.

### Task 2: Implement public-only rewrite schema and invariant validation

**Files:**
- Create: `dispute_agent/data/language_enrichment.py`
- Test: `tests/unit/data/test_language_enrichment.py`

- [ ] **Step 1: Write failing tests for public payload, style selection, and validation**

Cover: deterministic style selection, exact output keys, preservation of numeric anchors and evidence IDs, rejection of hidden labels/decision conclusions, and applying a valid rewrite without changing hidden truth.

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `python -m pytest tests/unit/data/test_language_enrichment.py -q`

Expected: FAIL because the module and public rewrite types do not exist.

- [ ] **Step 3: Implement the validation module**

Define `STYLE_PROFILES`, `LanguageRewrite`, `RewriteValidationError`, `public_payload(instance)`, `style_for_case(case_id, seed)`, `validate_rewrite(source, candidate)`, and `apply_rewrite(instance, candidate)`. The candidate model must forbid extra keys and only allow `buyer_claim`, `merchant_response`, `chat_log`, and `evidence_descriptions`.

- [ ] **Step 4: Run the focused tests and verify they pass**

Run: `python -m pytest tests/unit/data/test_language_enrichment.py -q`

Expected: PASS.

### Task 3: Implement the cache-first DeepSeek client

**Files:**
- Modify: `dispute_agent/data/language_enrichment.py`
- Test: `tests/unit/data/test_language_enrichment.py`

- [ ] **Step 1: Write failing fake-client tests**

Cover successful JSON rewrite, cache hit without a second client call, missing API key fallback, malformed JSON fallback, and client exception fallback. Assert prompts contain public fields and do not contain hidden field names or values.

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `python -m pytest tests/unit/data/test_language_enrichment.py -q`

Expected: FAIL because the client, cache, and runner are not implemented.

- [ ] **Step 3: Implement the client and cache**

Define `LanguageEnrichmentRunner`, `JsonlRewriteCache`, and `build_deepseek_runner`. Use `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL`, and `DEEPSEEK_MODEL`; call Chat Completions with `response_format={"type": "json_object"}`, `extra_body={"thinking": {"type": "disabled"}}`, bounded timeout/retries, and a prompt that explicitly requests JSON. Cache records must contain only public source hash, style, prompt version, model, status, rewritten public fields, and error metadata.

- [ ] **Step 4: Run the focused tests and verify they pass**

Run: `python -m pytest tests/unit/data/test_language_enrichment.py -q`

Expected: PASS.

### Task 4: Integrate optional enrichment into deterministic dataset generation

**Files:**
- Modify: `scripts/generate_data.py`
- Modify: `dispute_agent/data/splits.py`
- Test: `tests/leakage/test_dataset_leakage.py`
- Test: `tests/integration/test_data_generation.py`

- [ ] **Step 1: Write failing integration tests**

Use a fake runner to enrich a fixture and assert SFT and GRPO rows share the same rewritten public observation, GRPO still has no `messages`, fallback counts are recorded, and disabled mode makes zero enrichment calls.

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `python -m pytest tests/leakage/test_dataset_leakage.py tests/integration/test_data_generation.py -q`

Expected: FAIL because the generator has no enrichment switch or stats.

- [ ] **Step 3: Implement the CLI/config integration**

Add `--enrich-language`, `--language-ratio`, and `--language-cache` arguments. Enrich each selected `FactInstance` before `_render_row`; preserve deterministic non-enriched rows, write cache and `language_enrichment_report.json`, include their hashes when enrichment is enabled, and keep existing quality checks unchanged.

- [ ] **Step 4: Run the focused tests and verify they pass**

Run: `python -m pytest tests/leakage/test_dataset_leakage.py tests/integration/test_data_generation.py -q`

Expected: PASS.

### Task 5: Document usage and run local verification

**Files:**
- Modify: `README.md`
- Modify: `.gitignore`
- Test: existing focused suite plus a 24-row fixture generation

- [ ] **Step 1: Document installation and safe usage**

Document `uv pip install -e ".[dev,data]"`, `DEEPSEEK_API_KEY`, the explicit `--enrich-language` command, cache behavior, and the fact that generated data is uploaded to the training server rather than committed to GitHub. Ignore `data/processed/` and language cache outputs.

- [ ] **Step 2: Run all focused tests**

Run: `python -m pytest tests/test_project_contract.py tests/leakage/test_dataset_leakage.py tests/unit/training/test_grpo_dataset.py tests/unit/training/test_grpo_config.py tests/unit/agent/test_curriculum_runtime.py tests/integration/test_lightning_rollout.py tests/unit/training/test_grpo_runtime.py tests/integration/test_phase0_contract.py tests/unit/data/test_language_enrichment.py tests/integration/test_data_generation.py -q`

Expected: all tests pass; no real DeepSeek request is made.

- [ ] **Step 3: Generate and inspect a deterministic fixture**

Run: `python scripts/generate_data.py --seed 20260817 --fixture-size 24 --output data/processed-fixture`.

Expected: 24 rows, zero trace validation errors, and no GRPO public row containing `messages`.

- [ ] **Step 4: Commit the implementation**

```bash
git add pyproject.toml configs/data.yaml dispute_agent/data/language_enrichment.py scripts/generate_data.py dispute_agent/data/splits.py tests README.md .gitignore
git commit -m "feat: add deepseek language enrichment"
```

