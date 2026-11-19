#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Local (no-AWS) environment for the podman stack — see infra/local/docker-compose.yml.
# Source it: eval "$(scripts/local-env.sh)"
#
# botocore >= 1.34 resolves AWS_ENDPOINT_URL natively, so every boto3 client in
# the repo routes to LocalStack without code changes. OpenSearch runs plain-HTTP
# (the client switches to an unsigned transport on http:// endpoints).

cat <<'EOF'
export AWS_ENDPOINT_URL=http://localhost:4566
export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test
export AWS_REGION=us-east-1
export AWS_DEFAULT_REGION=us-east-1
export AWS_EC2_METADATA_DISABLED=true

export OPENSEARCH_ENDPOINT=http://localhost:9200
export OSS_ENDPOINT=http://localhost:9200

# Local OpenAI-compatible embeddings (Ollama). The Kilo gateway serves chat
# only; see the root .env for COA_LLM_BASE_URL / COA_LLM_API_KEY / COA_CHAT_MODEL.
export COA_LLM_EMBED_BASE_URL=http://localhost:11434/v1
export COA_EMBED_MODEL=nomic-embed-text
export COA_LLM_PROVIDER=openai
EOF
