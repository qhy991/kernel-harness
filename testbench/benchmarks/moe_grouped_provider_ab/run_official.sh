#!/usr/bin/env bash
# OFFICIAL-Anthropic side of the grouped-MoE A/B.
#
# IMPORTANT: log into claude.ai first, then run THIS immediately. Do NOT run any
# `claude -p ...` probe beforehand — each probe consumes a single-use OAuth refresh
# token and breaks the chain for this long run.
#
# Reaches official (not Infini) via `--settings` blanking the ANTHROPIC_* env that
# ~/.claude/settings.json injects, while using the default ~/.claude config so the
# subscription token refreshes in place with full account context.
set -uo pipefail
BENCH=/home/qinhaiyan/kernel-harness/testbench/benchmarks/moe_grouped_provider_ab
cd /home/qinhaiyan/kernel-harness/testbench

unset ANTHROPIC_BASE_URL ANTHROPIC_AUTH_TOKEN ANTHROPIC_API_KEY ANTHROPIC_MODEL \
      ANTHROPIC_DEFAULT_OPUS_MODEL ANTHROPIC_DEFAULT_SONNET_MODEL ANTHROPIC_DEFAULT_HAIKU_MODEL

SETTINGS='{"env":{"ANTHROPIC_BASE_URL":"","ANTHROPIC_AUTH_TOKEN":"","ANTHROPIC_API_KEY":"","ANTHROPIC_MODEL":"","ANTHROPIC_DEFAULT_OPUS_MODEL":"","ANTHROPIC_DEFAULT_SONNET_MODEL":"","ANTHROPIC_DEFAULT_HAIKU_MODEL":""}}'
PROMPT=$(cat "$BENCH/.prompt_official.txt")

/home/qinhaiyan/.local/bin/claude -p "$PROMPT" --model claude-opus-4-8 \
  --settings "$SETTINGS" --permission-mode bypassPermissions --verbose \
  2>&1 | tee "$BENCH/results/run_official.log"
echo "OFFICIAL_DONE_EXIT=${PIPESTATUS[0]}" >> "$BENCH/results/run_official.log"
