# /Oriol-TFM/experiments/run_grid_plus.py
"""
Run full-factorial benchmarks over (techniques × problems × trials) with a
provider-agnostic model registry (OpenAI, local OpenAI-compat, Anthropic, Google).

Usage (example):
  export OPENAI_API_KEY=...
  export ANTHROPIC_API_KEY=...
  export GEMINI_API_KEY=...

  # Local vLLM already tunneled at http://localhost:5678/v1 via your script
  python -m experiments.run_grid_plus --domain blocks --start 1 --end 5 \
    --config experiments/example_config.yaml \
    --trials 3 --out out/mega-$(date +%Y%m%d-%H%M%S)

Outputs:
  <out>/experiments.csv   -- one row per (problem, technique, trial)
  <out>/records.jsonl     -- full records, including artifact paths
  per-problem folders with purple/VAL artifacts (as in evaluate_once)

Notes:
- The harness uses the Green Agent's evaluate_once/evaluate_domain path and the new 'strategy' purple kind.
- Techniques pick specific roles/models from the config.
"""
from __future__ import annotations
import argparse, os, json, time, csv, re, yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from green_agent.config import EvalConfig
from green_agent.runner import evaluate_once
from green_agent.cli import _resolve_paths

def _load_yaml(p: str) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _infer_end(domain: str, start: int) -> int:
    probs_dir = Path("examples") / domain / "problems_pddl"
    ids = sorted([int(re.search(r"(\d+)", p.stem).group(1)) for p in probs_dir.glob("problem*.pddl")])
    return ids[-1] if ids else start

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--config", required=True, help="YAML with 'models:' and 'techniques:'")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--out", default="out/exp")
    ap.add_argument("--val-path", default=None)
    args = ap.parse_args()

    cfg_data = _load_yaml(args.config)
    models = cfg_data["models"]      # list of model dicts
    techniques = cfg_data["techniques"]  # list of technique dicts

    out_root = Path(args.out); out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / "experiments.csv"
    jsonl_path = out_root / "records.jsonl"

    # CSV header
    if not csv_path.exists():
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow([
                "ts","domain","index","technique","trial",
                "valid","cost_value","length","unsat_count","failure_reason",
                "score","optimal_cost","val_attempts","val_warning",
                "planner_id","judge_id","verifier_id","synth_id","proponent_a","proponent_b",
                "run_dir","raw_plan","norm_plan","val_stdout","val_stderr","val_trace"
            ])

    start_idx = int(args.start)
    end_idx = args.end or _infer_end(args.domain, start_idx)

    # Build a lookup for model IDs
    ID2MODEL: Dict[str, Dict[str, Any]] = {m["id"]: m for m in models}

    for idx in range(start_idx, end_idx + 1):
        auto = _resolve_paths(args.domain, idx)
        if not auto["domain"] or not auto["problem"]:
            print(f"[SKIP] Could not resolve paths for index {idx}")
            continue

        for tech in techniques:
            tech_name = tech["name"]
            roles_spec = {}
            # Fill roles from technique into strategy_params.roles
            role_ids = tech.get("roles", {})
            for role, model_id in role_ids.items():
                m = ID2MODEL[model_id]
                # Normalize provider names and resolve env keys if omitted
                roles_spec[role] = {
                    "provider": m["provider"],
                    "model": m["model"],
                    "base_url": m.get("base_url"),
                    "api_key": m.get("api_key") or os.getenv(m.get("api_key_env",""), None),
                    "temperature": m.get("temperature", 0.2),
                    "max_tokens": m.get("max_tokens", 2048),
                }

            strat_params = {"roles": roles_spec, "settings": tech.get("settings", {})}

            for t in range(1, max(1, args.trials) + 1):
                cfg = EvalConfig(
                    domain_path=auto["domain"],
                    problem_path=auto["problem"],
                    out_dir=str(out_root),
                    val_path=args.val_path,
                    purple_kind="strategy",
                    purple_url=None,
                    prompt_text=auto["prompt_text"],
                    openai_model=None,
                    llm_base_url=None,
                    llm_api_key=None,
                    check_redundancy=False,
                    optimal_cost=auto.get("optimal_cost"),
                    val_flags=("-v",),    # no VAL repair advice during eval; we just validate
                    print_card=False,
                    strategy_name=tech_name,
                    strategy_params=strat_params,
                )

                t0 = time.time()
                rec = evaluate_once(cfg)
                dt = time.time() - t0

                with open(jsonl_path, "a", encoding="utf-8") as jf:
                    jf.write(json.dumps({
                        **rec,
                        "domain_name": args.domain,
                        "index": idx,
                        "technique": tech_name,
                        "trial": t,
                        "elapsed_s": dt,
                        "roles": role_ids,
                    }, ensure_ascii=False) + "\n")

                with open(csv_path, "a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f); w.writerow([
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                        args.domain, idx, tech_name, t,
                        rec.get("valid"), rec.get("cost_value"), rec.get("length"), rec.get("unsat_count"),
                        rec.get("failure_reason"),
                        rec.get("score"), rec.get("optimal_cost"), rec.get("val_attempts"), rec.get("val_warning"),
                        role_ids.get("planner"), role_ids.get("judge"), role_ids.get("verifier"),
                        role_ids.get("synth"), role_ids.get("proponent_a"), role_ids.get("proponent_b"),
                        rec.get("run_dir"), rec.get("raw_plan_path"), rec.get("norm_plan_path"),
                        rec.get("val_stdout_path"), rec.get("val_stderr_path"), rec.get("val_trace_path"),
                    ])

                print(f"[{args.domain} p{idx:02d}] {tech_name} trial {t} → valid={rec.get('valid')} "
                      f"cost={rec.get('cost_value')} len={rec.get('length')} ({dt:.2f}s)")

    print(f"\n[OK] Wrote:\n- {csv_path}\n- {jsonl_path}")

if __name__ == "__main__":
    main()
