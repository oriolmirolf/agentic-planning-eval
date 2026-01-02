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
# PROMPT TEMPLATES (VERBATIM FROM APPENDIX)
# ------------------------------------------------------------------

DOMAIN_PROMPT_TEMPLATE = """You are an expert PDDL-to-English translator. Your goal is to convert a formal PDDL DOMAIN file into a "Domain Brief": a strict, natural-language manual describing the physics and logic of the environment.

The audience for this brief is an AI Planner that is PDDL-agnostic. It relies entirely on your English descriptions to understand what actions are available, when they can be used, and what they change.

You are given the following PDDL DOMAIN:

<DOMAIN_PDDL>
{{DOMAIN_PDDL}}
</DOMAIN_PDDL>

TASK
Generate a DOMAIN BRIEF that captures the exact logic of the domain without using PDDL syntax.

TRANSLATION RULES (Strict Compliance Required)
1. Vocabulary: Use the EXACT Action Names and Predicate Names from the PDDL (do not synonymize).
2. No Syntax: Do not use Lisp-style syntax (e.g., "(at ?x ?y)"). Instead, write clear English sentences (e.g., "Object ?x is located at ?y").
3. Logic:
   - Translate "(not ...)" in effects as "It is no longer the case that..." or "Remove the condition that..."
   - Translate "(imply ...)" or conditional effects as "IF [condition] THEN [effect]."
   - Translate numeric changes as "Increase/Decrease [metric] by [amount]."
4. Completeness: Every single precondition and effect must be translated. Do not summarize or skip "obvious" logic.
5. Sanitization: Do not mention instance-specific data (goals, initial states, specific object names). Only describe general mechanics.

OUTPUT FORMAT

DOMAIN BRIEF

# Summary
(Give a brief example of what the domain is about)

# Object Types
- [Type Name]: [Parent type if applicable, or "Base object"]
(If untyped, state: "All objects are generic.")

# Predicates & Functions
- [exact_predicate_name]: [Short English definition of what this represents]
- [exact_function_name]: [Short English definition of what this value represents]

# Actions
(Repeat for every action in the domain)

## Action: [Exact_Action_Name]
- Parameters: [List variables and their types, e.g., ?t (truck), ?loc (location)]
- Preconditions:
  - [English sentence describing condition 1]
  - [English sentence describing condition 2]
- Effects:
  - [English sentence describing what becomes TRUE]
  - [English sentence describing what becomes FALSE]
  - [English sentence describing numeric changes or costs]

FINAL CHECK
- Are all action names identical to the PDDL?
- Is all Lisp syntax "()" removed from the descriptions?
- Are the preconditions and effects logically complete?

Now, generate the DOMAIN BRIEF.
"""

INSTANCE_PROMPT_TEMPLATE = """You are an expert PDDL-to-English translator. Your goal is to convert a specific PDDL PROBLEM file into an "Instance Brief"—a clear, natural-language description of the scenario, objects, and goals.

The audience for this brief is an AI Planner that cannot read PDDL. It has already received the "Domain Brief" (physics/logic), so your job is ONLY to define the specific objects, the starting situation, and the winning conditions for this specific instance.

You are given:
1. DOMAIN BRIEF: The natural language description of the domain logic (already processed). Use this to ensure your vocabulary matches the domain.
2. PROBLEM PDDL: The specific file to translate.

<DOMAIN_BRIEF>
{{DOMAIN_BRIEF_NL}}
</DOMAIN_BRIEF>

<PROBLEM_PDDL>
{{PROBLEM_PDDL}}
</PROBLEM_PDDL>

TASK
Write an INSTANCE BRIEF that matches the PDDL problem exactly.

TRANSLATION RULES (Strict Compliance Required)
1. Exact Naming: Use the EXACT object names from the PDDL (e.g., if the object is "truck-01", do not write "Truck 1").
2. No Physics: Do not explain HOW actions work. Do not list operators. Only list what exists and what is true right now.
3. No Syntax: Do not use Lisp-style syntax (e.g., "(at truck1 loc1)"). Use English sentences (e.g., "truck1 is at loc1").
4. Closed World Assumption: Assume that in the Initial State, only the facts listed in the PDDL are true. Everything else is false.
5. Goal Precision: If the goal contains multiple parts (e.g., "(and ...)"), list them as separate bullet points.

OUTPUT FORMAT

INSTANCE BRIEF

# Summary
(Give a brief example of what the problem instance is about)

# Objects
- [Type Name]: [List of exact object names, comma-separated]
(Repeat for all types found in the problem. If untyped, list all under "Objects".)

# Initial State
(List every fact present in the :init section as a descriptive sentence)
- [Fact 1 in plain English]
- [Fact 2 in plain English]
- [Numeric assignments, e.g., "The fuel level of truck1 is 50"]

# Goal State
(The plan is successful only when the following are TRUE)
- [Goal requirement 1 in plain English]
- [Goal requirement 2 in plain English]

# Optimization Metric
(If a :metric exists, state it here. Otherwise, write: "None - Just satisfy goals.")
- [e.g., Minimize total-cost, or Minimize fuel-used]

FINAL CHECK
- Are all object names spelled exactly as they appear in the PDDL?
- Is the output free of "()" Lisp syntax?
- Did you avoid adding any "hints" or "plans"?

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
            temperature=1.0,
            reasoning_effort="minimal",
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        # In a real run, you might want to retry, but we'll return the error for visibility
        return f"[GENERATION FAILED: {e}]"

def extract_number(filename):
    """
    Extracts the first number found in a filename string. 
    e.g. 'problem5.pddl' -> 5, 'p01.pddl' -> 1
    """
    nums = re.findall(r'\d+', filename)
    if nums:
        return int(nums[0])
    return 0

def get_metadata_map(old_problems):
    """
    Creates a lookup dictionary { problem_number_int: {metadata_dict} }
    This allows us to match 'p01' and 'p1' to the same metadata using the integer 1.
    """
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

def process_domain(domain_name, domain_pbar):
    base_path = f"examples/{domain_name}"
    domain_file = os.path.join(base_path, "domain.pddl")
    problems_path = os.path.join(base_path, "problems_pddl")
    json_output_path = os.path.join(base_path, "prompts.json")

    if not os.path.exists(domain_file):
        tqdm.write(f"Skipping {domain_name}: domain.pddl not found.")
        return

    # 1. Load OLD JSON (To backup manual metadata)
    old_data = {"problems": [], "actions": []}
    if os.path.exists(json_output_path):
        with open(json_output_path, 'r') as f:
            try:
                old_data = json.load(f)
            except json.JSONDecodeError:
                tqdm.write(f"[WARNING] {json_output_path} corrupted. Starting fresh.")
    
    # Create the metadata lookup map based on integer IDs
    metadata_map = get_metadata_map(old_data.get("problems", []))

    # 2. Start NEW JSON Data (Preserve 'actions' from old file)
    new_data = {
        "domain_prompt": "",
        "actions": old_data.get("actions", []),
        "problems": []
    }

    # 3. Generate Domain Brief
    with open(domain_file, 'r') as f:
        domain_pddl_content = f.read()

    domain_pbar.set_description(f"Domain: {domain_name} (Gen Brief)")
    domain_prompt_input = DOMAIN_PROMPT_TEMPLATE.replace("{{DOMAIN_PDDL}}", domain_pddl_content)
    domain_brief = generate_completion(domain_prompt_input)
    new_data["domain_prompt"] = domain_brief

    # 4. Generate Problem Prompts
    # natsorted ensures 'problem1' comes before 'problem10'
    problem_files = natsorted(glob.glob(os.path.join(problems_path, "*.pddl")))
    
    if not problem_files:
        tqdm.write(f"No problem files found for {domain_name}")
    else:
        for p_file in tqdm(problem_files, desc=f"  Problems ({domain_name})", leave=False):
            p_filename = os.path.basename(p_file)
            
            # --- FIXED ID LOGIC ---
            # Extract number (e.g., 5) and format as p05
            p_num = extract_number(p_filename)
            p_id = f"p{p_num:02d}"  # Forces p01, p02... p10, p11
            
            with open(p_file, 'r') as f:
                problem_pddl_content = f.read()

            # Generate Instance Brief
            instance_prompt_input = INSTANCE_PROMPT_TEMPLATE.replace("{{DOMAIN_BRIEF_NL}}", domain_brief)
            instance_prompt_input = instance_prompt_input.replace("{{PROBLEM_PDDL}}", problem_pddl_content)
            instance_brief = generate_completion(instance_prompt_input)

            # Retrieve Backup Metadata (using the integer number)
            # This ensures that if old JSON had "p1" and now we have "p01", we still find the data.
            meta = metadata_map.get(p_num, {
                "overview": {}, 
                "objects": [], 
                "optimal_cost": 0, 
                "difficulty": "unknown"
            })

            # Create clean entry
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
    examples_dir = "examples"
    
    if not os.path.exists(examples_dir):
        print(f"Error: Directory '{examples_dir}' not found.")
        exit(1)

    domains = [d for d in os.listdir(examples_dir) if os.path.isdir(os.path.join(examples_dir, d))]
    domains = natsorted(domains) # Sort domains nicely too

    print(f"Found {len(domains)} domains.")
    
    domain_pbar = tqdm(domains, desc="Total Progress", unit="domain")
    
    for domain in domain_pbar:
        process_domain(domain, domain_pbar)
    
    print("\nAll domains processed successfully.")