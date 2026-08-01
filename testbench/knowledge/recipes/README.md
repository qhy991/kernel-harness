# Atomic experiment recipes

Each JSON file is one bounded experiment with one objective, its own acceptance
states, artifacts, and stop rule. Do not ask an agent to “run the recipe” without
naming an ID: select one item from [`index.json`](index.json), finish it, persist its
terminal state, and only then choose another.

```bash
python3 testbench/bin/recipe.py list
python3 testbench/bin/recipe.py show 05-decode-kv-context-matrix
python3 testbench/bin/recipe.py check
```

The numeric prefixes provide a useful default order but are not a mandatory bundled
workflow. Technique recipes (`20+`) should be selected only after their preconditions
match a measured bottleneck. Negative-result recipes preserve stop signals so later
agents do not repeat the same knob search.
