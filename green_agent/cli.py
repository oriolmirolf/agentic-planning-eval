from __future__ import annotations
import json, os, typer
from typing import Optional
from pathlib import Path
from .config import EvalConfig
from .runner import evaluate_once

app = typer.Typer(help="Green Agent: PDDL plan benchmark runner")

def _resolve_paths(example: Optional[str], index: Optional[int]) -> dict[str, Optional[str]]:
    if not example:
        return {"domain": None, "problem": None, "prompt": None}
    base = Path("examples") / example
    domain = base / "domain.pddl"

    if index is not None:
        problem = base / f"problem{index}.pddl"
        prompt  = base / f"prompt{index}.md"
        # fallback to unindexed if missing
        if not problem.exists():
            problem = base / "problem.pddl"
        if not prompt.exists():
            prompt = base / "prompt.md"
    else:
        problem = base / "problem.pddl"
        prompt  = base / "prompt.md"

    def ok(p: Path) -> Optional[str]:
        return str(p) if p.exists() else None

    return {"domain": ok(domain), "problem": ok(problem), "prompt": ok(prompt)}

@app.command()
def evaluate(
    # problem selection
    example: Optional[str] = typer.Option(None, help="Example folder name under ./examples (e.g., 'blocks')"),
    index: Optional[int] = typer.Option(None, help="Problem index (uses problem{index}.pddl / prompt{index}.md if present)"),
    # purple agent selection
    purple: str = typer.Option("openai", help="openai | a2a"),
    purple_url: Optional[str] = typer.Option(None, help="A2A endpoint of the purple agent (required if --purple a2a)"),
    out: str = typer.Option("out", help="Output directory (a fresh subfolder is created per run)"),
    val_path: Optional[str] = typer.Option(None, help="Path to VAL (Validate) binary"),
    model: Optional[str] = typer.Option(None, help="OpenAI model name (for purple=openai)"),
    check_redundancy: bool = typer.Option(False, help="Whether to check for redundant actions in the plan"),
):

    auto = _resolve_paths(example, index)
    domain = auto["domain"]
    problem = auto["problem"]
    prompt = auto["prompt"]

    if not domain or not problem:
        raise typer.BadParameter("Could not resolve domain/problem paths. Use --example (and optional --index).")

    cfg = EvalConfig(
        domain_path=domain,
        problem_path=problem,
        out_dir=out,
        val_path=val_path,
        purple_kind=purple,
        purple_url=purple_url,
        prompt_path=prompt,
        openai_model=model,
        check_redundancy=check_redundancy,
    )

    record = evaluate_once(cfg)

    os.makedirs(out, exist_ok=True)
    default_record_path = os.path.join(out, "record.json")
    with open(default_record_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

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
        
    cfg = EvalConfig(domain_path=domain, problem_path=problem, val_path=val_path)
    flags = (*cfg.val_flags, "-t", str(cfg.tolerance))
    res = run_val(domain, problem, plan_text, val_path=val_path, flags=flags)
    
    typer.echo("VALID" if res.ok else "INVALID")
    if res.value is not None:
        typer.echo(f"Value: {res.value}")
    if not res.ok and res.stdout:
        typer.echo(res.stdout)

if __name__ == "__main__":
    app()
