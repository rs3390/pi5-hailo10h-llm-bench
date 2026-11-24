# Is the Raspberry Pi AI HAT+ 2 faster for LLMs?

[![tests](https://github.com/rs3390/pi5-hailo10h-llm-bench/actions/workflows/tests.yml/badge.svg)](https://github.com/rs3390/pi5-hailo10h-llm-bench/actions/workflows/tests.yml)

On the same model, the AI HAT+ 2 writes at about half the speed of the Raspberry Pi 5's own CPU. It does leave the CPU almost idle while it writes.

I run an agent on a Pi 5 (8 GB); I wrote about building it in [I built a locally hosted AI Agent!](https://raghavsagar.substack.com/p/i-built-an-agent-and-you-can-too) I bought the AI HAT+ 2 (the Hailo-10H chip, 40 TOPS, 8 GB of its own memory) hoping to move the agent's model onto it. This repo is the script I used to find out, `bench.py`, and what I found. Run it on your own Pi.

## Results

Raspberry Pi 5, 8 GB. Hailo runtime 5.1.1 (`hailo-h10-all`) and Hailo's Model Zoo GenAI 5.1.1. The same model on both: Llama 3.2 3B. Temperature 0, at most 200 tokens, three runs each after a warm-up. 22 November 2026.

| Where | Prompt | First token, run 1 (s) | Writing (tokens/s) | Pi load |
|---|---|---|---|---|
| CPU (Ollama) | one line | 0.22 | 6.15 | ~100% |
| HAT (hailo-ollama) | one line | 1.10 | 2.65 | 0.6% |
| CPU (Ollama) | paragraph | 0.80 | 5.20 | ~100% |
| HAT (hailo-ollama) | paragraph | 1.08 | 2.66 | 0.2% |
| CPU (Ollama) | long prompt | _to come_ | _to come_ | |
| HAT (hailo-ollama) | long prompt | _to come_ | _to come_ | |

The HAT was slower at both steps: reading the prompt, and writing the answer. My first runs used a long prompt from my own agent, so the long-prompt rows will come from `bench.py`'s own prompt (about 400 words), which anyone can repeat.

I also ran my agent's model, Qwen3 4B, on the CPU against Llama 3.2 3B on the HAT. Qwen wrote at 4.3 to 5.7 tokens a second. The HAT, running the smaller model, still wrote at 2.65.

The tables the script wrote, with my notes on each run, are in [`results/`](results/).

## What I found along the way

1. **The HAT can only run what Hailo has compiled for it.** Model Zoo GenAI 5.1.1 serves five models, the largest `llama3.2:3b`. No Qwen3. Hailo's current model list does include Qwen3-4B-Instruct-2507, but in a newer Model Zoo, and people on Hailo's forum have reported the newer packages breaking against the 5.1.1 runtime on the Pi. I stayed on 5.1.1.
2. **hailo-ollama answers HTTP 500 to `"keep_alive": "10m"`.** Ollama accepts it. hailo-ollama's JSON mapper fails with `Value is NOT a Primitive type`. Leave `keep_alive` out of requests to the HAT. `bench.py` does.
3. **The same model answers differently on the HAT.** On the CPU, Llama stopped by itself at 8 and 103 tokens. On the HAT it ran on to 18 and 200 tokens (the cap), on the same prompts. Hailo's compiled version seems to handle the prompt format or the end of an answer differently. Check its answers as well as its speed.
4. **Ollama on the CPU remembers a prompt it has just read.** Runs 2 and 3 of a prompt reach the first token much faster than run 1 (0.80 s, then 0.24 s). The HAT does not do this. Compare first runs, which `bench.py` reports in their own column.
5. **Tokens a second is not quite fair across two different models**, because each model cuts text into tokens differently. `bench.py` also reports words a second.

## Why the HAT is slower: a hypothesis

I expected 40 TOPS to beat four ARM cores. I have not measured why it does not. My hypothesis is that the speed limit for writing an answer is memory, not arithmetic. To write each token, the model reads nearly all of its weights from memory once (Llama 3.2 3B at 4 bits is about 2 GB). So the limit would be how fast the chip can read its memory (bigger model = more to read per token = fewer tokens a second). The TOPS figure would matter more for camera work, where there is a lot of arithmetic for each byte read.

Hailo's own page for Qwen3 1.7B on the Hailo-10H lists 4.72 tokens a second. That is about what the Pi 5's CPU gets from a model more than twice the size, which fits the hypothesis.

## What the HAT is still good for

- **Work alongside the CPU.** The Pi stays free while the HAT writes, and the HAT's model does not use the Pi's RAM. A background job can run on the HAT while the CPU answers you. I have not measured this yet.
- **What the chip is designed for:** vision, and speech-to-text with Whisper. I have not tested these either. They are next for my agent.

## Run it yourself

You need the Hailo runtime, Hailo's Model Zoo GenAI 5.1.1 with `hailo-ollama` running, and Ollama for the CPU side. [`docs/setup.md`](docs/setup.md) has every step and the problems I hit. Then:

```
python3 bench.py check --probe llama3.2:3b        # is the HAT there, and doing the work?
python3 bench.py speed --cpu-model llama3.2:3b    # the same model on both
```

Results are saved in `results/`. `--only hat` or `--only cpu` runs one side; `--runs 5` runs each prompt five times. `bench.py` needs Python 3.10 or later and nothing else. The tests run anywhere, with no Pi: `python3 -m unittest`

## Assumptions and limits

- One Pi, one HAT, one day. Speeds will differ with cooling, power supply and software versions.
- Hailo's compiled model and Ollama's are both about 4 bits, but they are not the same file. That difference is still in the comparison.
- Three prompts, each run three times. A wider test would change the numbers a little. I doubt it would change the conclusion.
- Processor load is read from `/proc/stat`, so it needs Linux. On a Mac it reads n/a.

## Licence

MIT. See [`LICENSE`](LICENSE).

Raghav Sagar · [raghavsagar.substack.com](https://raghavsagar.substack.com)
