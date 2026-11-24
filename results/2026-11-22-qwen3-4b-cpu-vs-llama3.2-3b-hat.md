# Qwen3 4B on the CPU against Llama 3.2 3B on the HAT

Raspberry Pi 5 (8 GB), Hailo runtime 5.1.1, Model Zoo GenAI 5.1.1. 22 November 2026. Medians of three runs after one warm-up; temperature 0, at most 200 tokens. Pi load is the Pi's own processor, as a share of all cores.

Run with an earlier version of `bench.py` that had no "first token, run 1" or "words a second" column. Its third prompt was a long prompt from my own agent (about 420 words), not the one in today's `bench.py`, so that row is not comparable with later runs. Two different models: see the same-model run for the fair comparison.

| Where | Model | Prompt | First token (s) | Writing (tok/s) | Tokens | Total (s) | Pi load | HAT held by |
|---|---|---|---|---|---|---|---|---|
| cpu | qwen3:4b-instruct | one line | 0.22 | 5.71 | 8 | 1.7 | 100.0% | hailo-ollama |
| cpu | qwen3:4b-instruct | paragraph | 0.23 | 4.86 | 102 | 21.2 | 100.0% | hailo-ollama |
| cpu | qwen3:4b-instruct | long prompt (agent's own) | 0.28 | 4.32 | 32 | 7.7 | 100.0% | hailo-ollama |
| hat | llama3.2:3b | one line | 1.09 | 2.65 | 18 | 7.9 | 0.4% | hailo-ollama |
| hat | llama3.2:3b | paragraph | 1.09 | 2.65 | 200 | 76.1 | 0.2% | hailo-ollama |
| hat | llama3.2:3b | long prompt (agent's own) | 7.13 | 2.65 | 200 | 82.1 | 0.2% | hailo-ollama |

First runs, from the printed log: CPU one line 0.25 s, paragraph 0.92 s, long prompt 0.84 s. The HAT's first runs matched its medians.

"HAT held by" on the CPU rows means hailo-ollama had the chip open while the CPU model ran; it was not doing the work.
