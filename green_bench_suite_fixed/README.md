# green_bench_suite

Production-grade experiment runner for your `green_agent` planning benchmark.

This package lives **outside** (i.e., sibling to) `green_agent/` so you can evolve the benchmark independently from the experiment orchestration.

## What it does

Runs an **All×All** matrix:

- **models** (either launched via `vec_inf` *or* pointing at pre-launched OpenAI-compatible endpoints)
- × **5 prompting strategies** (baseline + 4 literature-backed variants)
- × **N domains**
- × **all problems in each domain's `prompts.json`**

For each *single problem run*, it creates a separate **MLflow run**, logs:
- **params:** model_name, strategy, temperature=0, max_tokens (8192 for Thinking, 4096 otherwise)
- **metrics:** success (parsed plan?), duration_seconds (LLM latency), plus useful extras (valid, plan_length, etc.)
- **artifacts:** raw_response.txt (required), plus plan + VAL logs for debugging

Also logs:
- **git_commit** as an MLflow tag (**required**)
- warns if repo is dirty (uncommitted changes)

## Install deps

You need:
- `mlflow`
- `gitpython`
- `openai`
- VAL available in `$PATH` or `VAL_PATH`, or pass `--val-path`.

Optional:
- your internal `vec_inf` package (only needed if you want the benchmark runner to *launch* models)

Example:
```bash
pip install mlflow gitpython openai
```

## Run

From your repo root (or anywhere—git root is auto-detected):

```bash
# Auto mode: tries vec_inf; if vec_inf is not importable it falls back to manual mode.
python -m green_bench --experiment Planning_Benchmark_v1
```

### Manual / tunnel mode (no vec_inf required)

If you already launched a model and exposed it locally via an SSH tunnel, run:

```bash
python -m green_bench \
  --mode manual \
  --base-url http://localhost:5679/v1 \
  --experiment Planning_Benchmark_v1
```

Notes:
- If you do **not** override `--models`, the runner will try to auto-detect the served model id via `/v1/models`.
- To run multiple models in one invocation, use `--base-url-map` (one endpoint per model), or run the script multiple times (one tunnel per run).

Common options:

```bash
python -m green_bench \
  --domains blocks,balancer,hospital,gripper,logistics \
  --models Kimi-K2-Thinking,Qwen2.5-72B-Instruct \
  --strategies baseline,zero_shot_cot,plan_and_solve,step_back,self_refine \
  --val-path /path/to/Validate \
  --mlflow-uri http://your-mlflow-server:5000 \
  --out-dir out_bench \
  --limit-problems 2
```

## Output structure

Artifacts are written locally under:

```
out_bench/planning_benchmark_<timestamp>/
  model=<MODEL>/
    domain=<DOMAIN>/
      strategy=<STRATEGY>/
        p01/
          raw_response.txt
          plan.plan
          val_stdout.txt
          val_stderr.txt
          val_trace.json
```

Each of these files is also logged to MLflow as artifacts.
