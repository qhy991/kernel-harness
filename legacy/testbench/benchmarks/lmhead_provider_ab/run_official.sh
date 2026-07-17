#!/usr/bin/env bash
set -uo pipefail
BENCH=/home/qinhaiyan/kernel-harness/testbench/benchmarks/lmhead_provider_ab
cd /home/qinhaiyan/kernel-harness/testbench
unset ANTHROPIC_BASE_URL ANTHROPIC_AUTH_TOKEN ANTHROPIC_API_KEY ANTHROPIC_MODEL ANTHROPIC_DEFAULT_OPUS_MODEL ANTHROPIC_DEFAULT_SONNET_MODEL ANTHROPIC_DEFAULT_HAIKU_MODEL
SETTINGS='{"env":{"ANTHROPIC_BASE_URL":"","ANTHROPIC_AUTH_TOKEN":"","ANTHROPIC_API_KEY":"","ANTHROPIC_MODEL":"","ANTHROPIC_DEFAULT_OPUS_MODEL":"","ANTHROPIC_DEFAULT_SONNET_MODEL":"","ANTHROPIC_DEFAULT_HAIKU_MODEL":""}}'
PROMPT=$(cat "$BENCH/.prompt_official.txt")
/home/qinhaiyan/.local/bin/claude -p "$PROMPT" --model claude-opus-4-8 --settings "$SETTINGS" --permission-mode bypassPermissions --verbose 2>&1 | tee "$BENCH/results/run_official.log"
echo "OFFICIAL_DONE_EXIT=${PIPESTATUS[0]}" >> "$BENCH/results/run_official.log"
