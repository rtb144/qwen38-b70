#!/usr/bin/env bash
# Hit llama-server /v1/chat/completions: short prompt, ~32k prompt, retrieval check.
# Usage: ./scripts/bench-api.sh [base_url]
#   base_url default: http://127.0.0.1:${HOST_PORT:-8016}
# The 32k prompt is passed via a temp file, not argv (MAX_ARG_STRLEN).
# If LLAMA_API_KEY is set, it is sent as Authorization: Bearer.
set -euo pipefail

BASE="${1:-http://127.0.0.1:${HOST_PORT:-8016}}"
BASE="${BASE%/}"
MODEL_NAME="${MODEL_NAME:-qwen3.8-27b}"

need() { command -v "$1" >/dev/null 2>&1 || { echo "need $1"; exit 1; }; }
need curl
need jq
need python3

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

CURL_AUTH=()
if [ -n "${LLAMA_API_KEY:-}" ]; then
  CURL_AUTH=(-H "Authorization: Bearer ${LLAMA_API_KEY}")
fi

echo "Waiting for $BASE/health ..."
ok=0
for _ in $(seq 1 60); do
  if curl -sf "${CURL_AUTH[@]}" "$BASE/health" >/dev/null 2>&1; then
    ok=1
    break
  fi
  sleep 3
done
if [ "$ok" != 1 ]; then
  echo "server not healthy at $BASE/health"
  exit 1
fi

PARA='The Gated Delta Network is a linear-attention variant that replaces softmax with a gated delta rule update over a recurrent state. Each layer maintains a fixed-size state matrix that is updated in O(1) per token, independent of sequence length, which makes decode throughput nearly flat with context. The delta rule is h = h + g * (v - h . k) where g is a data-dependent sigmoid gate, k a key, and v a value. Aggressive negative g values cause rapid forgetting of old state, keeping the state bounded. Full attention is used every fourth layer to recover exact long-range tokens that the linear path would otherwise forget.'

# Unique fact for the retrieval probe (also pads the long prompt).
SECRET='The retrieval canary is PROJECT-ZEPHYR-4419 and the warehouse code is NX-77-MINT.'

chat() {
  local prompt="$1" max_tokens="$2"
  local prompt_file
  prompt_file="$(mktemp "$WORKDIR/prompt.XXXXXX")"
  # The ~32k prompt exceeds Linux MAX_ARG_STRLEN (~128 KiB), so it cannot
  # be passed as a python argv element or an environment variable.
  printf '%s' "$prompt" > "$prompt_file"
  python3 - "$BASE" "$MODEL_NAME" "$prompt_file" "$max_tokens" "${LLAMA_API_KEY:-}" <<'PY'
import json, pathlib, sys, urllib.request

base, model, prompt_path, max_tokens, api_key = (
    sys.argv[1],
    sys.argv[2],
    sys.argv[3],
    int(sys.argv[4]),
    sys.argv[5],
)
prompt = pathlib.Path(prompt_path).read_text()
body = {
    "model": model,
    "max_tokens": max_tokens,
    "temperature": 0,
    "messages": [{"role": "user", "content": prompt}],
}
headers = {"Content-Type": "application/json"}
if api_key:
    headers["Authorization"] = "Bearer " + api_key
req = urllib.request.Request(
    base + "/v1/chat/completions",
    data=json.dumps(body).encode(),
    headers=headers,
)
with urllib.request.urlopen(req, timeout=3600) as resp:
    sys.stdout.write(resp.read().decode())
PY
}

print_timings() {
  local label="$1" json="$2"
  echo "$json" | jq -r --arg l "$label" '
    . as $r
    | {
        label: $l,
        prompt_tokens: ($r.usage.prompt_tokens // $r.timings.prompt_n // null),
        completion_tokens: ($r.usage.completion_tokens // $r.timings.predicted_n // null),
        prompt_tok_s: ($r.timings.prompt_per_second // null),
        generation_tok_s: ($r.timings.predicted_per_second // null),
        prompt_ms: ($r.timings.prompt_ms // null),
        predicted_ms: ($r.timings.predicted_ms // null)
      }'
}

echo
echo "=== short prompt ==="
SHORT_PROMPT='Reply with the single word: pong'
SHORT_JSON="$(chat "$SHORT_PROMPT" 16)"
print_timings short "$SHORT_JSON"
echo "text: $(echo "$SHORT_JSON" | jq -r '.choices[0].message.content // empty' | tr '\n' ' ')"

echo
echo "=== ~32k-token prompt (repeated filler + canary) ==="
LONG_PROMPT="$(python3 - "$PARA" "$SECRET" <<'PY'
import sys
para, secret = sys.argv[1], sys.argv[2]
# ~32k tokens ≈ 24k words of this dense paragraph; 380 repeats is a conservative pad.
body = (para + " ") * 380
print(
    "Read the text. Near the end is a retrieval canary. "
    "What is the warehouse code? Answer with only the code.\n\n"
    + body
    + "\n\n"
    + secret
)
PY
)"
LONG_JSON="$(chat "$LONG_PROMPT" 32)"
print_timings long32k "$LONG_JSON"
LONG_TEXT="$(echo "$LONG_JSON" | jq -r '.choices[0].message.content // empty')"
echo "text: $(echo "$LONG_TEXT" | tr '\n' ' ')"

echo
echo "=== retrieval accuracy ==="
if echo "$LONG_TEXT" | grep -qi 'NX-77-MINT'; then
  echo "PASS: warehouse code NX-77-MINT found in completion"
  RETRIEVAL=1
else
  echo "FAIL: expected NX-77-MINT in completion"
  RETRIEVAL=0
fi

jq -n \
  --argjson retrieval "$RETRIEVAL" \
  --argjson short "$SHORT_JSON" \
  --argjson long "$LONG_JSON" \
  '{
     retrieval_pass: $retrieval,
     short: {
       prompt_tok_s: ($short.timings.prompt_per_second // null),
       generation_tok_s: ($short.timings.predicted_per_second // null),
       prompt_tokens: ($short.usage.prompt_tokens // null)
     },
     long32k: {
       prompt_tok_s: ($long.timings.prompt_per_second // null),
       generation_tok_s: ($long.timings.predicted_per_second // null),
       prompt_tokens: ($long.usage.prompt_tokens // null)
     }
   }'
