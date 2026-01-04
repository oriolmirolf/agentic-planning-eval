import os
import glob
import json
import re
import argparse
from natsort import natsorted
from openai import OpenAI
from tqdm import tqdm

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ------------------------------------------------------------------
# PROMPT TEMPLATES
# ------------------------------------------------------------------

DOMAIN_PROMPT_TEMPLATE = """You are an expert PDDL-to-English translator. Your goal is to convert a formal PDDL DOMAIN file into a "Domain Brief": a clear, technical, natural-language manual describing the physics and logic of the environment.

The audience for this brief is an AI Planner that is PDDL-agnostic. It relies entirely on your English descriptions to understand how the world works, what actions are available, and what constraints apply.

You are given the following PDDL DOMAIN:

<DOMAIN_PDDL>
{{DOMAIN_PDDL}}
</DOMAIN_PDDL>

TASK
Generate a DOMAIN BRIEF that captures the physics and logic of the domain using **technical prose** instead of formulas.

TRANSLATION RULES (Strict Compliance Required)
1. **No Predicates List**: Do not list predicates or functions explicitly. Instead, write a "World Description" section that explains the rules of the world.
2. **Action Requirements**: Do not use bullet points for "Preconditions". Instead, write a "Requirements" paragraph for each action that explains *in English* what must be true.
3. **Action Effects**: Write an "Effects" paragraph describing what changes in the world.
4. **Vocabulary**: Use the EXACT Action Names and Object Types from the PDDL.
5. **No Syntax**: Do not use Lisp-style syntax (e.g., "(at ?x ?y)"). Use clear sentences.
6. **Cost**: If the PDDL mentions `total-cost`, explicitly mention it in the Effects.

OUTPUT FORMAT

DOMAIN BRIEF

# Summary
(Give a 1-2 sentence summary of what the domain models)

# Object Types
- [Type Name]: [Short description]

# World Description
(A few numbered points describing the physics/constraints of the world.)

# Actions
(Repeat for every action in the domain)

## Action: [Exact_Action_Name]
- Parameters: [List variables and their types]
- Requirements: [Prose description of conditions]
- Effects: [Prose description of changes]

Now, generate the DOMAIN BRIEF.
"""

INSTANCE_PROMPT_TEMPLATE = """You are an expert technical writer. Your goal is to convert a formal PDDL PROBLEM file into an "Instance Brief"—a clear, natural-language scenario description.

The audience is an AI Planner. It has already received the "Domain Brief" (physics), so your job is ONLY to define the specific objects, the starting situation, and the goal.

<DOMAIN_BRIEF>
{{DOMAIN_BRIEF_NL}}
</DOMAIN_BRIEF>

<PROBLEM_PDDL>
{{PROBLEM_PDDL}}
</PROBLEM_PDDL>

TASK
Write an INSTANCE BRIEF.

TRANSLATION RULES
1. **Exact Naming**: MUST use exact PDDL object names (e.g., "s1_eu", "truck-01").
2. **Objects Section**: Keep this as a structured list. It serves as the "Inventory".
3. **State & Goal**: Use **descriptive paragraphs**.
   - Do NOT use bullet points for the state or goal.
   - Group related facts (e.g., "All trucks are currently at the depot.").
   - Describe the goal as a mission statement (e.g., "Your job is to deliver packages A and B to the warehouse.").
4. **Closed World**: Assume unmentioned booleans are false.

OUTPUT FORMAT

INSTANCE BRIEF

# Summary
(1-2 sentences summarizing the scenario)

# Objects
- [Type Name]: [List of exact object names, comma-separated]

# Initial State
(Write 2-3 short paragraphs describing the current state of the world. Group objects by location or type. Use narrative flow.)

# Goal State
(Write a short paragraph describing the required final state. Integrate multiple goals into coherent sentences. Do NOT use bullet points.)

# Optimization Metric
(If metric exists: "Minimize [metric name]". Else: "None")

Now, generate the INSTANCE BRIEF.
"""

# ------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------

def generate_completion(full_prompt_text):
    try:
        completion = client.chat.completions.create(
            model="gpt-5-2025-08-07",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": full_prompt_text}
            ],
            temperature=0.0,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"[GENERATION FAILED: {e}]"

def extract_number(filename):
    """Extracts the first number found in a filename string."""
    nums = re.findall(r'\d+', filename)
    if nums:
        return int(nums[0])
    return 0

def get_metadata_map(old_problems):
    """Creates a lookup dictionary { problem_number_int: {metadata_dict} }"""
    meta_map = {}
    for entry in old_problems:
        p_id_str = entry.get("id", "")
        p_num = extract_number(p_id_str)
        if p_num > 0:
            meta_map[p_num] = {
                "overview": entry.get("overview", {}),
                "objects": entry.get("objects", []),
                "optimal_cost": entry.get("optimal_cost", 0),
                "difficulty": entry.get("difficulty", "unknown")
            }
    return meta_map

def process_domain(domain_name, domain_pbar, only_problems=False):
    base_path = f"examples/{domain_name}"
    domain_file = os.path.join(base_path, "domain.pddl")
    problems_path = os.path.join(base_path, "problems_pddl")
    json_output_path = os.path.join(base_path, "prompts.json")

    if not os.path.exists(domain_file):
        tqdm.write(f"Skipping {domain_name}: domain.pddl not found.")
        return

    # 1. Load OLD JSON (To backup manual metadata or reuse domain prompt)
    old_data = {"problems": [], "actions": [], "domain_prompt": ""}
    if os.path.exists(json_output_path):
        with open(json_output_path, 'r') as f:
            try:
                old_data = json.load(f)
            except json.JSONDecodeError:
                tqdm.write(f"[WARNING] {json_output_path} corrupted. Starting fresh.")
    
    metadata_map = get_metadata_map(old_data.get("problems", []))

    # 2. Determine Domain Prompt
    domain_brief = ""
    
    if only_problems and old_data.get("domain_prompt"):
        # REUSE EXISTING
        domain_brief = old_data["domain_prompt"]
        # domain_pbar.set_description(f"Domain: {domain_name} (Reusing Brief)")
    else:
        # GENERATE NEW
        with open(domain_file, 'r') as f:
            domain_pddl_content = f.read()
        
        domain_pbar.set_description(f"Domain: {domain_name} (Gen Brief)")
        domain_prompt_input = DOMAIN_PROMPT_TEMPLATE.replace("{{DOMAIN_PDDL}}", domain_pddl_content)
        domain_brief = generate_completion(domain_prompt_input)

    if not domain_brief:
        tqdm.write(f"[ERROR] No domain brief available for {domain_name}. Skipping.")
        return

    # 3. Start NEW JSON Data
    new_data = {
        "domain_prompt": domain_brief,
        "actions": old_data.get("actions", []),
        "problems": []
    }

    # 4. Generate Problem Prompts
    problem_files = natsorted(glob.glob(os.path.join(problems_path, "*.pddl")))
    
    if not problem_files:
        tqdm.write(f"No problem files found for {domain_name}")
    else:
        for p_file in tqdm(problem_files, desc=f"  Problems ({domain_name})", leave=False):
            p_filename = os.path.basename(p_file)
            p_num = extract_number(p_filename)
            p_id = f"p{p_num:02d}"  # Forces p01, p02...

            with open(p_file, 'r') as f:
                problem_pddl_content = f.read()

            instance_prompt_input = INSTANCE_PROMPT_TEMPLATE.replace("{{DOMAIN_BRIEF_NL}}", domain_brief)
            instance_prompt_input = instance_prompt_input.replace("{{PROBLEM_PDDL}}", problem_pddl_content)
            instance_brief = generate_completion(instance_prompt_input)

            # Retrieve Backup Metadata
            meta = metadata_map.get(p_num, {
                "overview": {}, 
                "objects": [], 
                "optimal_cost": 0, 
                "difficulty": "unknown"
            })

            new_entry = {
                "id": p_id,
                "prompt": instance_brief,
                "overview": meta["overview"],
                "objects": meta["objects"],
                "optimal_cost": meta["optimal_cost"],
                "difficulty": meta["difficulty"]
            }
            new_data["problems"].append(new_entry)

    # 5. Save Completely Rebuilt JSON
    try:
        with open(json_output_path, 'w') as f:
            json.dump(new_data, f, indent=2)
    except Exception as e:
        tqdm.write(f"[ERROR] Saving JSON for {domain_name}: {e}")

# ------------------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate NL prompts from PDDL.")
    parser.add_argument(
        "--only-problems", 
        action="store_true", 
        help="If set, reuses the existing 'domain_prompt' from json and only regenerates problem instances."
    )
    args = parser.parse_args()

    examples_dir = "examples"
    
    if not os.path.exists(examples_dir):
        print(f"Error: Directory '{examples_dir}' not found.")
        exit(1)

    domains = [d for d in os.listdir(examples_dir) if os.path.isdir(os.path.join(examples_dir, d))]
    domains = natsorted(domains)

    print(f"Found {len(domains)} domains.")
    
    domain_pbar = tqdm(domains, desc="Total Progress", unit="domain")
    
    for domain in domain_pbar:
        process_domain(domain, domain_pbar, only_problems=args.only_problems)
    
    print("\nAll domains processed successfully.")