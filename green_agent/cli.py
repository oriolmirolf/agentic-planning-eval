from __future__ import annotations
import json, os, typer
from typing import Optional
from .config import EvalConfig
from .runner import evaluate_once

app = typer.Typer(help="Green Agent: PDDL plan benchmark runner")

@app.command()
def evaluate(
    domain: str = typer.Option(..., exists=True, help="Path to domain.pddl"),
    problem: str = typer.Option(..., exists=True, help="Path to problem.pddl"),
    prompt: Optional[str] = typer.Option(None, help="Natural-language problem description for the purple agent"),
    actions: Optional[str] = typer.Option(None, help="Optional actions cheat-sheet for the purple agent"),
    purple: str = typer.Option("openai", help="openai | http | file"),
    purple_url: Optional[str] = typer.Option(None, help="HTTP endpoint for purple agent, or file path for --purple file"),
    out: str = typer.Option("out", help="Output directory"),
    val_path: Optional[str] = typer.Option(None, help="Path to VAL (Validate) binary"),
    model: Optional[str] = typer.Option(None, help="OpenAI model name (for purple=openai)"),
    temperature: float = typer.Option(0.0, help="LLM temperature for the openai purple agent"),
    report: Optional[str] = typer.Option(None, help="Write JSONL record to this file (appended)"),
    attempts: int = typer.Option(3, min=1, help="Max plan-repair attempts"), # rn with NL feedback
):
    cfg = EvalConfig(
        domain_path=domain,
        problem_path=problem,
        out_dir=out,
        val_path=val_path,
        purple_kind=purple,
        purple_url=purple_url,
        prompt_path=prompt,
        openai_model=model,
        temperature=temperature,
        attempts=attempts,
    )
    record = evaluate_once(cfg)
    if report:
        os.makedirs(os.path.dirname(report) or ".", exist_ok=True)
        with open(report, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    else:
        typer.echo(json.dumps(record, indent=2))

# useful for debugging
@app.command()
def validate_plan(
    domain: str = typer.Option(..., exists=True, help="Path to domain.pddl"),
    problem: str = typer.Option(..., exists=True, help="Path to problem.pddl"),
    plan: str = typer.Option(..., exists=True, help="Path to plan file (.plan)"),
    val_path: Optional[str] = typer.Option(None, help="Path to VAL (Validate) binary"),
):
    from .val_wrapper import run_val
    with open(plan, "r", encoding="utf-8") as f:
        plan_text = f.read()
    res = run_val(domain, problem, plan_text, val_path=val_path)
    typer.echo("VALID" if res.ok else "INVALID")
    if res.value is not None:
        typer.echo(f"Value: {res.value}")
    if not res.ok and res.stdout:
        typer.echo(res.stdout)

if __name__ == "__main__":
    app()
