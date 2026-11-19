# Plan: Replace AWS-Bedrock AI layer with a Kilo OpenAI-compatible shim (local podman runtime)

> **Direction (updated):** No AWS infrastructure is used as part of this work. The
> **OpenAI-compatible (Kilo) implementation is the default provider** (`COA_LLM_PROVIDER`
> defaults to `openai`); `bedrock` remains an explicit opt-in for legacy deployments only.
> The LLM/embedding path runs entirely against the Kilo AI Gateway (or a local
> OpenAI-compatible endpoint in podman), never against AWS Bedrock.
>
> **Live-verified against the real gateway** (via `.env`, never committed):
> `https://ai-gateway.kilo.ai/api/v1` + `~deepseek/deepseek-flash-latest` —
> `OpenAICompatibleClient.invoke` returns parsed JSON with usage, and the serve
> `OpenAICompatibleLLMClient.converse` + SSE `converse_stream` stream real tokens.
> Env contract: `KILO_*` names (`KILO_BASE_URL`, `KILO_API_KEY`, `KILO_MODEL`,
> `KILO_MAX_TOKENS`, `KILO_TEMPERATURE`, `KILO_TIMEOUT`) with `COA_*` aliases;
> base URLs are normalized to append `/v1`. The gateway serves NO embedding
> models, so embeddings use a dedicated local endpoint
> (`COA_LLM_EMBED_BASE_URL`, e.g. Ollama `http://localhost:11434/v1`,
> `COA_EMBED_MODEL=nomic-embed-text`).

## Findings — important correction

There is **no litellm anywhere in this codebase** — not in `uv.lock`, not transitively, not in
git history. The AI layer today is **direct AWS Bedrock**, plus a few Bedrock-coupled libraries.
So "replace litellm with a Kilo OpenAI shim" really means *replace the Bedrock LLM/embedding/
guardrail layer with an OpenAI-compatible transport shim pointed at the Kilo AI Gateway*
(`https://ai-gateway.kilo.ai/api/v1`, OpenAI-compatible `/chat/completions`), so the AI path
stops needing AWS and can run in podman.

### Current AWS-AI coupling (the actual surface to shim)

| Layer | Today | Consumers |
|---|---|---|
| Chat/JSON LLM | `coa_common/bedrock.py` `BedrockClient.invoke()` (Converse API, Claude) | sources enrichment (`table_enricher.py:195`, `relationship_inferrer.py`, `cross_source_relationship_inferrer.py`, `cross_source_orchestrator.py:237`), ontology-engine `nl_generator.py:176` |
| Embeddings | `coa_common/embeddings.py` `BedrockEmbedder` (InvokeModel, Cohere/Titan) | serve `clients/bedrock.py`, metric-service `opensearch_client.py`, sources `kg_build/graph_build.py`, ontology-engine `bedrock_embeddings.py`, llama-index adapter |
| GraphRAG LLM | llama-index `llms.bedrock_converse.BedrockConverse` (patched in `lexical/graphrag_patches.py`) | serve Tier-3 lexical retrieval |
| Agent runtime | `strands-agents` `BedrockModel` (`sql_agent.py:444`) | NL->SQL agentic Tier-2 |
| AgentCore | `bedrock-agentcore==1.18.1` dep | serve SSE runtime |
| Guardrails | Bedrock Guardrail API (`guardrail_screener.py`, `guardrail_metrics.py`, `bedrock.py` guardrail config) | serve tier3 `chunk_screener.py`/`synthesizer.py`, sources kg_build |
| Cost | `bedrock_metrics.py` `CostTracker` (Bedrock pricing per model id) | chat + embed paths |

Rest of the AWS surface (for the podman goal): 103 `import boto3` sites across dynamodb, s3,
sqs, ssm, secretsmanager, sts, athena, glue, opensearchserverless (AOSS), cognito, neptune,
textract, redshift-data, lambda, stepfunctions, events, etc.

## Plan

### Phase 0 — Introduce the provider seam (no behavior change) — ✅ DONE
- [x] New `libs/common/src/coa_common/llm/` package: `protocol.py` (abstract `LLMClient` +
  `Embedder`, mirroring the existing public surface exactly:
  `invoke(system_prompt, user_prompt|PromptSegment, max_tokens) -> InvocationResult(result,
  input_tokens, output_tokens, latency_ms)` and
  `embed_query/embed_document/embed_documents`), plus `openai/` implementations.
- [x] `coa_common/bedrock.py` and `embeddings.py` remain the Bedrock implementations; factory
  selects by env (`COA_LLM_PROVIDER=bedrock|openai`) defaulting to `bedrock` — zero risk,
  deployable. 20 new unit tests in `libs/common/tests/unit/test_llm_openai.py` (red->green),
  ruff-clean.
- [x] Keep `BEDROCK_CHAT_MODEL_ID`/`BEDROCK_EMBED_MODEL_ID` as-is; added `COA_LLM_BASE_URL`
  (defaults to `https://ai-gateway.kilo.ai/api/v1`), `COA_LLM_API_KEY`,
  `COA_CHAT_MODEL`/`COA_EMBED_MODEL` (gateway ids like `anthropic/claude-haiku-4.5`).
- [x] **Default provider flipped to `openai` (Kilo) — no AWS in the LLM path.** Factory default
  is now the OpenAI-compatible client/embedder; `COA_LLM_PROVIDER=bedrock` opts back in.
  Factory tests updated (red->green); full libs/common (1185), sources (2009), and
  ontology-engine (3613) unit suites pass with the new default.

### Phase 1 — OpenAI-compatible chat shim — ✅ DONE
- [x] `openai/chat.py`: `invoke()` -> `POST {base}/chat/completions` with
  `messages=[{role:"system"},{role:"user"}]`, `max_tokens`. Preserves all BedrockClient contracts:
  - [x] `finish_reason=="length"` -> `BedrockTruncationError` equivalent (`stop_reason` semantics,
    `usage.completion_tokens`)
  - [x] JSON parse + code-fence stripping + empty-content `ValueError` (same post-processing
    as `bedrock.py:354-370`)
  - [x] input-size cap -> `InputTooLargeError`; usage -> `CostTracker.record`
  - [ ] `PromptSegment` guarded/unguarded -> local guardrail decision (see Phase 3)
- [x] Cut over chat consumers to `get_llm_client(...)` (sources enrichment
  `table_enricher.py`, `cross_source_orchestrator.py`, annotations in
  `relationship_inferrer.py`/`cross_source_relationship_inferrer.py`; ontology-engine
  `nl_generator.py`); tests re-pointed at the factory. Full-env suites pass:
  libs/common 1185, sources 2009, ontology-engine 3613.

### Phase 2 — Embeddings + GraphRAG — ✅ DONE
- [x] `openai/embed.py`: `embed_query/embed_document/embed_documents` ->
  `POST {base}/embeddings` (single-string for 1 input, batched lists clamped to
  [1, 2048] otherwise; order preserved; token estimate + `CostTracker.record`).
  Default embed model `nomic-embed-text` (local-server friendly); `COA_EMBED_MODEL`
  overrides. **Open question carried:** Kilo Gateway embedding support unverified —
  local podman embedding container (e.g. Ollama `/v1/embeddings`) is the fallback.
- [x] Provider-aware LlamaIndex helpers: `get_llama_index_embedding` (BaseEmbedding
  adapter over the OpenAI embedder, picklable) and `get_llama_index_llm`
  (LlamaIndex `OpenAI(api_base=COA_LLM_BASE_URL)`); `None` for both in bedrock mode.
- [x] Embedder consumers wired through the factory: metric-service
  `opensearch_client.py`, ontology-engine `bedrock_embeddings.py`, serve
  `clients/bedrock.py` (query embedder), `baseline_retriever.py` +
  `kg_build/graph_build.py` (GraphRAG `embed_model`).
- [x] GraphRAG LLM routed to the provider: `graphrag_patches.patch_graphrag_llm_provider`
  installs the OpenAI LLM on `extraction_llm`/`response_llm` and patches
  `GraphRAGConfig.to_llm` so every internal construction point gets the OpenAI
  instance; `kg_build` sets `extraction_llm` directly in openai mode (Bedrock
  ARN path untouched under `COA_LLM_PROVIDER=bedrock`).
- [x] Usage accounting generalized: `llm_usage.record_usage` accepts Bedrock
  (`inputTokens`/`outputTokens`) and OpenAI (`prompt_tokens`/`completion_tokens`)
  shapes; `patch_llm_usage_accounting` wraps the provider's LLM class
  (BedrockConverse and/or OpenAI) and normalizes `ChatResponse.raw` usage.
- [x] Strands NL→SQL agent: `SqlAgent._build_model` builds `OpenAIModel` by default
  (`client_args.base_url/api_key` → Kilo gateway, `model_id`, `params.max_tokens`)
  and `BedrockModel` under `COA_LLM_PROVIDER=bedrock`. Added deps to coa-serve:
  `llama-index-llms-openai`, `openai`.
- [x] Full suites green: libs/common 1185, metric-service 1339, ontology-engine 3613,
  sources 2009, context-manager 5300. Ruff-clean.

### Phase 3 — Guardrails + serve LLM client (OpenAI-compatible) — 🔄 IN PROGRESS
- [x] **Serve LLM client**: new `coa_serve/clients/openai.py`
  `OpenAICompatibleLLMClient` mirrors `BedrockLLMClient`'s surface (`converse`,
  `converse_stream`, `embed`, `health_check`, `close`, `model_id`); factory
  `build_llm_client()` picks it by provider; `main.py` routes through it. SSE
  streaming parses `data:` events with `stream_options.include_usage`; usage
  recorded via the shared tracker. 11 new unit tests; live-verified against the
  gateway (converse + stream).
- [x] **Guardrails are a no-op locally**: outcomes always `NONE`, streams never
  raise `GuardrailBlockedError`; when a `guardrail_id` is configured an `ALLOW`
  decision metric is emitted so dashboards keep counting the guarded path.
- [ ] **Cost/pricing table — DEFERRED (out of scope per direction)**: the shared
  `CostTracker` family-fallback already prices gateway ids; no dedicated OpenAI
  pricing table is being built in this work.
- [ ] **`bedrock-agentcore` — decision pending**: it is the AgentCore SSE runtime
  dependency (not a model call); keep for now, revisit in Phase 4/5.

### Phase 4 — Local podman stack — ✅ DONE (running + smoke-tested)
- [x] `infra/local/docker-compose.yml` (podman-compose): LocalStack (dynamodb, s3, sqs,
  ssm, secretsmanager, sts, cognito, lambda, stepfunctions, events, logs, cloudwatch,
  athena/glue best-effort), OpenSearch 2.19 single-node (plain HTTP, security off),
  Postgres 16 (host port 5433 — 5432 is commonly a local Postgres), Ollama 0.5.7
  (`/v1` embeddings, nomic-embed-text).
- [x] AWS endpoint injection is **zero-code**: botocore 1.43 honors `AWS_ENDPOINT_URL`
  natively, so all 103 boto3 construction sites route to LocalStack via env
  (`scripts/local-env.sh` + root `.env`). Verified: `dynamodb list_tables` through the
  env-only path.
- [x] OpenSearch local transport: `AossVectorClient` builds an **unsigned** client for
  plain-`http://` endpoints (local), keeping SigV4 for `https://` AOSS (2 new tests).
- [x] `make local-up / local-down / local-env` targets + `scripts/local-smoke.sh`.
- [x] **Smoke-tested live**: localstack health + dynamodb, opensearch health, ollama
  `/v1/models` + real 768-dim nomic-embed-text embeddings through the shim, and the
  Kilo-gateway chat in the same run. All 4 containers healthy.
- [ ] Neptune not emulated — graph tiers stay remote/AWS-only (documented limitation).

### Phase 5 — Cutover & cleanup — 🔄 MOSTLY DONE
- [x] Defaults flipped: `COA_LLM_PROVIDER` defaults to `openai` everywhere (factory, serve
  client, GraphRAG, strands); Bedrock remains an explicit, documented opt-in — the
  Bedrock implementations are kept (dormant) rather than deleted so `COA_LLM_PROVIDER=bedrock`
  still works for legacy deployments.
- [x] Local podman smoke harness wired: `make local-smoke` (LocalStack health + dynamodb via
  `AWS_ENDPOINT_URL`, OpenSearch, Ollama embeddings) — passes against the live stack.
- [x] ruff-clean + full unit suites green: libs/common 1187, metric-service 1339,
  ontology-engine 3613, sources 2009, context-manager 5300.
- [ ] Deferred (documented, not doing in this work): deleting the dormant Bedrock AI paths;
  stripping `strands-agents`/`bedrock-agentcore` boto3 coupling (runtime-only deps of the
  legacy provider).

### Risks / open questions
1. Gateway embeddings support — needs a verify or a local embedding container.
2. Guardrail parity locally is behavior-changing (AWS guardrail -> allow + Presidio) —
   acceptable only for dev.
3. LocalStack Athena/Glue/Neptune/Textract coverage is partial — those paths may stay AWS-only
   (documented), or get local fakes.
4. `bedrock-agentcore` is the heaviest AWS root in serve; decide replace-vs-keep before Phase 3.
5. Model id mapping (Bedrock profile ids -> gateway ids) must be env-driven per deployment, not
   hardcoded.

### Suggested order of work
Phase 0 -> Phase 1 (chat shim, run enrichment + nl_generator locally) -> Phase 2 -> Phase 3 ->
Phase 4 -> Phase 5. Phase 0+1 deliver the "kilo OpenAI shim" with zero risk; Phases 2-3 complete
the AI de-AWS; Phases 4-5 are the broader podman-everything goal.
