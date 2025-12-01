# LLM Planning Evaluation

This repository contains a framework for evaluating Large Language Models (LLMs) on planning tasks.
[write more...]

## Directory Structure

* `experiments/`: Scripts for running grid searches and logging to MLflow.
* `green_agent/`: The validation logic, CLI tools, and A2A server integration.
* `purple_agent/`: The generation logic (OpenAI wrappers, etc.).
* `helpful/`: Utilities for calculating optimal plans using classical planners (FF).
* `examples/`: Contains PDDL domains and problems (e.g., `blocks/`).

## Prerequisites

1.  VAL (Validate): You must have the [KCL PDDL Validator](https://github.com/KCL-Planning/VAL) installed and accessible via `validate` on your PATH (or set `VAL_PATH` in `.env`).

2.  uv: This project uses `uv` for dependency management.
    ```bash
    # Install uv (macOS/Linux)
    curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
    ```

## Installation

1.  Clone the repository:
    ```bash
    git clone <your-repo-url>
    cd Oriol-TFM
    ```

2.  Sync dependencies:
    This creates a virtual environment and installs all required packages.
    ```bash
    uv sync
    ```

3.  Environment Setup:
    ```bash
    cp .env.example .env
    # Edit .env with your OpenAI API Key and path to VAL
    ```

## Usage

All commands can be run via `uv run` to ensure they use the correct environment.

### 1. Evaluating a Specific Problem
Run the Green Agent CLI to evaluate a specific problem index for a domain.

```bash
uv run python -m green_agent.cli evaluate \
    --domain blocks \
    --index 1 \
    --model gpt-4o-mini \
    --out out/manual_test
```

### 2\. Running a Domain Benchmark

Evaluate a range of problems in batch mode.

```bash
uv run python -m green_agent.cli evaluate-domain \
    --domain blocks \
    --start 1 --end 5 \
    --model gpt-4o-mini \
    --llm-workers 4
```

### 3\. Running the Experiment Grid (MLflow)

Run a comprehensive grid search across different prompting techniques (CoT, etc.).

```bash
# Ensure you have MLflow installed or configured
uv run python -m experiments.run_grid \
    --domain blocks \
    --techniques base,cot \
    --start 1 --end 3
```

### 4\. Calculating Optimal Baselines

If you have the `ff` planner installed, you can pre-calculate optimal plan lengths for scoring.

```bash
uv run python -m helpful.optimal_plans --domain blocks --planner bfs
```

## A2A Server Mode

You can run the Green Agent as a server compliant with the A2A (Agent-to-Agent) protocol.

```bash
uv run python -m green_agent.a2a_server --port 9009
```
