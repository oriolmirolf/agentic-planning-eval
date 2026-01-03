import os
import json
import re
from natsort import natsorted
from rich.console import Console
from rich.table import Table
from rich import box

def extract_number(p_id):
    """Extracts integer number from problem ID (e.g. 'p01' -> 1)."""
    nums = re.findall(r'\d+', str(p_id))
    return int(nums[0]) if nums else 0

def main():
    console = Console()
    examples_dir = "examples"
    
    if not os.path.exists(examples_dir):
        console.print(f"[bold red]Error:[/bold red] Directory '{examples_dir}' not found.")
        return

    # 1. Setup the Table
    table = Table(
        title="Optimal Costs per Domain (Problems 1-10)",
        box=box.ROUNDED,
        header_style="bold cyan",
        title_style="bold magenta"
    )
    
    table.add_column("Domain", style="green", no_wrap=True)
    for i in range(1, 11):
        table.add_column(f"P{i:02d}", justify="center", style="yellow")

    # 2. Find Domains
    domains = [d for d in os.listdir(examples_dir) if os.path.isdir(os.path.join(examples_dir, d))]
    domains = natsorted(domains)

    rows_found = 0

    # 3. Process Each Domain
    for domain in domains:
        prompts_path = os.path.join(examples_dir, domain, "prompts.json")
        
        # Initialize costs array for P01..P10 with "-" placeholder
        costs = ["-"] * 10 
        
        if os.path.exists(prompts_path):
            try:
                with open(prompts_path, 'r') as f:
                    data = json.load(f)
                    
                # Map costs: {problem_number: cost}
                cost_map = {}
                for p in data.get("problems", []):
                    p_num = extract_number(p.get("id", ""))
                    if 1 <= p_num <= 10:
                        c = p.get("optimal_cost")
                        # Handle Unsolvable (-1 or None)
                        if c == -1:
                            cost_map[p_num] = "[red]inf[/red]" # or UNS
                        elif c is None:
                            cost_map[p_num] = "?"
                        else:
                            cost_map[p_num] = str(c)
                
                # Fill the row list
                for i in range(1, 11):
                    if i in cost_map:
                        costs[i-1] = cost_map[i]
                        
            except Exception as e:
                console.print(f"[red]Error reading {domain}: {e}[/red]")
        
        # Add row to table
        table.add_row(domain, *costs)
        rows_found += 1

    # 4. Print
    if rows_found > 0:
        console.print(table)
    else:
        console.print("[yellow]No domains found in examples/.[/yellow]")

if __name__ == "__main__":
    main()