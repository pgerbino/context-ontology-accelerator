#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Smoke-check the local no-AWS podman stack (infra/local/docker-compose.yml).
# Requires: stack up (`make local-up`), envs sourced (`eval "$(scripts/local-env.sh)"`).
# Exits non-zero on the first failed check.

set -euo pipefail

fail() { echo "FAIL: $1" >&2; exit 1; }
ok() { echo "ok: $1"; }

# 1. LocalStack health + DynamoDB via the native AWS_ENDPOINT_URL injection.
if curl -sf http://localhost:4566/_localstack/health >/dev/null 2>&1; then
  ok "localstack health"
else
  fail "localstack not healthy on :4566"
fi
export AWS_ENDPOINT_URL=http://localhost:4566 AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test
export AWS_REGION=us-east-1 AWS_EC2_METADATA_DISABLED=true
if uv run python -c 'import boto3; boto3.client("dynamodb", region_name="us-east-1").list_tables()' >/dev/null 2>&1; then
  ok "dynamodb via AWS_ENDPOINT_URL"
else
  fail "dynamodb list_tables failed against LocalStack"
fi

# 2. OpenSearch (plain HTTP — unsigned client path).
if curl -sf http://localhost:9200/_cluster/health >/dev/null 2>&1; then
  ok "opensearch on :9200"
else
  fail "opensearch not reachable on :9200"
fi

# 3. Ollama OpenAI-compatible /v1/models (+ a real embed if nomic-embed-text is pulled).
if curl -sf http://localhost:11434/v1/models >/dev/null 2>&1; then
  ok "ollama /v1/models"
else
  fail "ollama OpenAI-compatible endpoint not reachable on :11434"
fi
if python3 - <<'PY' >/dev/null 2>&1
import json, urllib.request
req = urllib.request.Request(
    "http://localhost:11434/v1/embeddings",
    data=json.dumps({"model": "nomic-embed-text", "input": "hello"}).encode(),
    headers={"Content-Type": "application/json"},
)
resp = json.load(urllib.request.urlopen(req))
assert resp["data"][0]["embedding"]
PY
then
  ok "embeddings via local nomic-embed-text"
else
  echo "note: embeddings skipped — run: podman exec coa-local-ollama ollama pull nomic-embed-text"
fi

echo "local stack smoke ok"
