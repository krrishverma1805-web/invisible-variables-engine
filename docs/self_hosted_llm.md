# Self-Hosted LLM Compatibility

**Effective date:** 2026-04-29
**Owner:** ML lead + platform lead
**Plan reference:** §91 + §140 + §141

This document covers the operator path for running IVE without sending
any prompt content to a third-party API. It's the answer for users with
PII/PHI/GDPR/financial residency constraints whose `dataset_column_metadata`
sensitivity choices alone don't satisfy their threat model.

## Decision tree

```
Do you have datasets where ANY column might be sensitive
even after public/non_public tagging?
├── No  → keep using Groq with the sensitivity model (RC §16). Done.
└── Yes
    ├── ≥4×A100 80GB available → Tier 1 (full fidelity)
    ├── 1×A100 / 1×L40S available → Tier 2 (smaller model, retuned validators)
    └── CPU-only → out of scope (see §"CPU is not viable")
```

## Tier 1 — full fidelity (recommended)

**Hardware:** 4×A100 80GB, OR 2×H100 80GB.
**Model:** `meta-llama/Llama-3.3-70B-Instruct` served via vLLM.
**Latency target:** <2s per single-LV explanation; <8s per batched 8-LV call.

### Setup

```bash
# 1. Pull model weights (requires HF token + license acceptance)
huggingface-cli login
huggingface-cli download meta-llama/Llama-3.3-70B-Instruct

# 2. Run vLLM with OpenAI-compatible endpoint
docker run --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  vllm/vllm-openai:latest \
    --model meta-llama/Llama-3.3-70B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --api-key local-dev-key

# 3. Front with TLS (production)
# Option A: nginx reverse proxy with letsencrypt
# Option B: Caddy with automatic HTTPS
# See docs/runbooks/ for sample configs.
```

### IVE configuration

```env
LLM_SELF_HOSTED_MODE=true
GROQ_BASE_URL=https://your-vllm-host.internal/v1
GROQ_API_KEY=local-dev-key
GROQ_MODEL=meta-llama/Llama-3.3-70B-Instruct
LLM_VALIDATOR_PROFILE=groq_llama_3_3_70b
LLM_EXPLANATIONS_ENABLED=true
```

The `LLM_VALIDATOR_PROFILE` defaults to the same profile used for Groq
because Tier 1 serves the same model weights. No profile override needed.

### Verification

```bash
# Run the behavioral suite against the local endpoint
GROQ_BASE_URL=https://your-vllm-host.internal/v1 \
  poetry run pytest tests/llm/ tests/unit/test_behavioral_corpus.py -q
```

Expected: ≥95% pass rate (matches Groq baseline within statistical noise).

## Tier 2 — single-GPU, smaller model

**Hardware:** 1×A100 80GB or 1×L40S 48GB.
**Model:** `meta-llama/Llama-3.1-8B-Instruct`.
**Latency target:** <1s per call (model is much smaller).

### Quality trade-off

The 8B model passes the validator stack at a noticeably lower rate than
70B (~85% vs ~95%). Failures route to rule-based fallback automatically;
end-user experience degrades gracefully but the fallback rate metric
(`ive_llm_fallback_total`) stays measurably higher.

### Setup

```bash
docker run --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  vllm/vllm-openai:latest \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --max-model-len 8192 \
    --api-key local-dev-key
```

### IVE configuration

```env
LLM_SELF_HOSTED_MODE=true
GROQ_BASE_URL=https://your-vllm-host.internal/v1
GROQ_API_KEY=local-dev-key
GROQ_MODEL=meta-llama/Llama-3.1-8B-Instruct
LLM_VALIDATOR_PROFILE=vllm_llama_3_1_8b
LLM_BATCH_SIZE_LVS=4              # smaller context window
LLM_EXPLANATIONS_ENABLED=true
```

The Tier-2 profile (`src/ive/llm/profiles/vllm_llama_3_1_8b.py`) carries:

- Relaxed numeric tolerance (±5% vs default ±2%)
- Extra banned phrases known from 8B output (over-confident superlatives)
- Smaller max-output tokens (160 vs 512 default)
- Pre-flight check: behavioral suite must reach ≥80% pass-rate before
  promoting; below that, alert and stay on Groq/Tier-1.

## CPU is not viable

A 70B model on CPU runs at multi-second-per-token speed. A single LV
explanation would take 30–60 seconds. Batched calls become unworkable.

If GPU is unavailable:
- Use rule-based explanations only (`LLM_EXPLANATIONS_ENABLED=false`).
- Functionality preserved; output is templated rather than analytical.
- All other IVE features (HPO, ensembles, CI, FPR sentinel) work
  unchanged on CPU.

## Operator runbook

### Health check

```bash
curl -s http://your-vllm-host.internal/v1/models \
  -H "Authorization: Bearer local-dev-key" | jq '.data[].id'
```

### Watching for drift

Run the weekly behavioral drift check (plan §164) against the
self-hosted endpoint. Local model updates (e.g., quantization tweaks)
are detected within 7 days; the daily smoke (5 prompts) catches severe
regressions in ≤24 h.

### Backup / failover

When the self-hosted endpoint is unreachable, the IVE circuit breaker
opens after `groq_circuit_breaker_threshold` failures (default 5). All
LV rows route to rule-based until the breaker closes. No data loss; no
operator action required beyond restoring the endpoint.

### Cost (Tier 1, illustrative)

| Component | Spec | Approx. monthly |
|---|---|---|
| 4×A100 80GB (cloud) | g6.48xlarge or equiv | $14k–18k |
| 4×A100 80GB (owned) | Hardware + power | $1.5k–3k (amortized) |
| Compare to Groq | 100k experiments/day | $730k/year ($60k/mo) |

Self-hosted breaks even at ~10k experiments/day on cloud GPUs, lower on
owned hardware.

## Quality regression checks for self-hosted

Run on every config change:

```bash
# 1. Behavioral corpus (10 cases)
poetry run pytest tests/unit/test_behavioral_corpus.py -q

# 2. Adversarial corpus (20 cases — sanitization is upstream of model)
poetry run pytest tests/unit/test_adversarial_corpus.py -q

# 3. Sensitive-data egress (4 cases)
poetry run pytest tests/integration/test_sensitive_data_egress.py -q
```

All three suites must remain green when switching from Groq to a
self-hosted endpoint.

## What does NOT change with self-hosting

- API contracts (`docs/RESPONSE_CONTRACT.md` unchanged).
- Validator stack (`src/ive/llm/validators.py` is model-agnostic).
- Sensitivity model (RC §16).
- Cache key structure (cache invalidates on `model_id` change).
- Fallback behavior (rule-based prose, identical end-user UX).

## What changes

- `docs/RESPONSE_CONTRACT.md §5` egress inventory becomes
  network-boundary-only (no provider tracking).
- Cost model (RC §15) shifts from per-token to per-GPU-hour.
- `model_version` opacity (RC §19) is reduced — operator controls
  weight updates.
- Latency target shifts (Tier 1 ~2s vs Groq ~150ms).
