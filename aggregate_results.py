# aggregate_results.py
import argparse
import os
import mlflow
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Initialize Console for printing
console = Console()

def main():
    parser = argparse.ArgumentParser(description="Aggregate MLflow runs.")
    parser.add_argument("--experiment", required=True, help="Name of the MLflow experiment")
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    parser.add_argument("--output", default="aggregated_results.csv", help="Output CSV filename")
    args = parser.parse_args()

    # 1. Connect and Fetch
    mlflow.set_tracking_uri(args.tracking_uri)
    exp = mlflow.get_experiment_by_name(args.experiment)
    if not exp:
        console.print(f"[bold red]Error:[/bold red] Experiment '{args.experiment}' not found.")
        return

    console.print(f"Fetching runs for experiment: [bold]{args.experiment}[/bold] (ID: {exp.experiment_id})...")
    
    # Fetch all runs (no filter, so we get everything)
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])

    if runs.empty:
        console.print("[yellow]No runs found.[/yellow]")
        return

    # ---------------------------------------------------------
    # 2. Robust Column Mapping (Handles different metric names)
    # ---------------------------------------------------------
    def get_col(df, candidates, default_val=0.0):
        """Checks multiple column names and returns the first one found."""
        for c in candidates:
            if c in df.columns:
                return df[c].fillna(default_val)
        return pd.Series([default_val] * len(df))

    # --- IDENTIFIERS ---
    # Model (Compatible with old names and new gemini names)
    runs["_model"] = get_col(runs, ["params.model_name", "params.model", "tags.model"], "Unknown")
    runs["_model"] = runs["_model"].apply(lambda x: str(x).split("/")[-1]) # Clean path if needed

    # Technique & Domain
    runs["_tech"] = get_col(runs, ["params.strategy", "params.prompting_technique", "tags.strategy"], "Default")
    runs["_domain"] = get_col(runs, ["params.domain", "tags.domain"], "Unknown")

    # --- METRICS (Using the names you provided) ---
    # MLflow flattens metrics to 'metrics.name'. We look for your specific keys.
    runs["_score"] = get_col(runs, ["metrics.score"], 0.0)
    runs["_time"]  = get_col(runs, ["metrics.wall_time", "metrics.total_duration_seconds"], 0.0)
    
    # Success: Look for 'success', 'is_success'
    runs["_success"] = get_col(runs, ["metrics.success", "metrics.is_success"], 0.0)
    
    # Valid: Look for 'valid', 'is_valid'
    runs["_valid"] = get_col(runs, ["metrics.valid", "metrics.is_valid"], 0.0)

    # --- SOLVABILITY LOGIC ---
    # Cost = -1 means Unsolvable. We look for this tag/param to split stats.
    runs["_opt_cost"] = get_col(runs, ["params.optimal_cost", "tags.optimal_cost"], "0")
    
    def check_unsolvable(val):
        try: return float(val) == -1.0
        except: return False
    
    runs["_is_unsolvable"] = runs["_opt_cost"].apply(check_unsolvable)

    # ---------------------------------------------------------
    # 3. Calculation Helper
    # ---------------------------------------------------------
    def calc_stats(group, group_labels):
        # Split into Solvable (Plan exists) vs Unsolvable (Impossible)
        solvable = group[group["_is_unsolvable"] == False]
        unsolvable = group[group["_is_unsolvable"] == True]

        return {
            **group_labels,
            "N": len(group),
            "Success %": group["_success"].mean(),
            "Valid %":   group["_valid"].mean(),
            "Avg Score": group["_score"].mean(),
            "Avg Time":  group["_time"].mean(),
            
            # Specific breakdown
            "N_Solv": len(solvable),
            "Succ_Solv": solvable["_success"].mean() if not solvable.empty else np.nan,
            
            "N_Unsolv": len(unsolvable),
            "Succ_Unsolv": unsolvable["_success"].mean() if not unsolvable.empty else np.nan,
        }

    console.print(Panel(f"[bold]Experiment Analysis: {args.experiment}[/bold]", style="blue"))

    # ---------------------------------------------------------
    # VIEW 1: AGGREGATED BY PROMPTING TECHNIQUE
    # ---------------------------------------------------------
    rows_tech = []
    for tech, group in runs.groupby("_tech"):
        rows_tech.append(calc_stats(group, {"Technique": tech}))
    
    df_tech = pd.DataFrame(rows_tech).sort_values(by="Avg Score", ascending=False)

    table_t = Table(title="Aggregated by Prompting Technique", header_style="bold cyan")
    table_t.add_column("Technique", style="white")
    table_t.add_column("N", justify="right")
    table_t.add_column("Success %", style="bold green")
    table_t.add_column("Valid %", style="yellow")
    table_t.add_column("Solvable %", justify="right", style="dim")
    table_t.add_column("Unsolv %", justify="right", style="dim")
    table_t.add_column("Score", style="bold magenta")
    table_t.add_column("Time", justify="right")

    for _, row in df_tech.iterrows():
        # Format breakdown columns
        s_solv = f"{row['Succ_Solv']:.0%}" if pd.notna(row['Succ_Solv']) else "-"
        s_unsolv = f"{row['Succ_Unsolv']:.0%}" if pd.notna(row['Succ_Unsolv']) else "-"
        
        table_t.add_row(
            str(row["Technique"]),
            str(row["N"]),
            f"{row['Success %']:.1%}",
            f"{row['Valid %']:.1%}",
            s_solv,   # Accuracy on solvable problems
            s_unsolv, # Accuracy on impossible problems
            f"{row['Avg Score']:.3f}",
            f"{row['Avg Time']:.1f}s"
        )
    console.print(table_t)
    console.print("\n")

    # ---------------------------------------------------------
    # VIEW 2: AGGREGATED BY DOMAIN
    # ---------------------------------------------------------
    rows_domain = []
    for domain, group in runs.groupby("_domain"):
        rows_domain.append(calc_stats(group, {"Domain": domain}))
    
    df_domain = pd.DataFrame(rows_domain).sort_values(by="Domain")

    table_d = Table(title="Aggregated by Domain", header_style="bold magenta")
    table_d.add_column("Domain", style="white")
    table_d.add_column("N", justify="right")
    table_d.add_column("Success %", style="green")
    table_d.add_column("Valid %", style="yellow")
    table_d.add_column("Score", style="magenta")
    table_d.add_column("Avg Time", justify="right")

    for _, row in df_domain.iterrows():
        table_d.add_row(
            str(row["Domain"]),
            str(row["N"]),
            f"{row['Success %']:.1%}",
            f"{row['Valid %']:.1%}",
            f"{row['Avg Score']:.3f}",
            f"{row['Avg Time']:.1f}s"
        )
    console.print(table_d)

    # ---------------------------------------------------------
    # VIEW 3: SAVE DETAILED BREAKDOWN
    # ---------------------------------------------------------
    full_rows = []
    # Group by Model, Tech, AND Domain for the CSV
    for (m, t, d), group in runs.groupby(["_model", "_tech", "_domain"]):
        full_rows.append(calc_stats(group, {"Model": m, "Technique": t, "Domain": d}))
    
    df_full = pd.DataFrame(full_rows)
    df_full.to_csv(args.output, index=False)
    console.print(f"\n[dim]Detailed breakdown saved to: {args.output}[/dim]")

if __name__ == "__main__":
    main()