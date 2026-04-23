"""
SAS → Databricks Conversion Pipeline
=====================================
Step-by-step Python script replicating all backend pipeline stages.
Run sections sequentially. Each section prints its output.

Pipeline stages:
  1. Setup & Configuration
  2. LLM Service Functions
  3. LLM Prompts
  4. Load SAS File
  5. Classify SAS Complexity (LLM)
  6. Source Mapping (manual)
  7. Macro Resolution (manual)
  8. Plan Generation + Validation Loop (LLM)
  9. Plan Review & Edit
  10. Code Generation + Per-Cell Validation (LLM)
  11. Build & Save Output Notebook
  12. Inspect Generated Cells
"""

# ============================================================
# SECTION 1 — Setup & Configuration
# ============================================================
import json
import re
import time
import os
import copy
from pprint import pprint

import google.generativeai as genai
from google.generativeai import GenerationConfig

# ─── CONFIGURE THESE ────────────────────────────────────────
API_KEY          = "YOUR_GEMINI_API_KEY_HERE"         # paste your key
ANALYSIS_MODEL   = "gemma-4-26b-a4b-it"               # Classifier, Planner, Code Gen
VALIDATION_MODEL = "gemini-3.1-flash-lite-preview"    # Plan Validator, Code Validator
SAS_FILE_PATH    = r"path\to\your\file.sas"           # path to the SAS file
OUTPUT_PATH      = r"databricks_output.py"            # where to save the generated notebook
MAX_PLAN_LOOPS   = 5
MAX_CELL_RETRIES = 5
# ─────────────────────────────────────────────────────────────

print("✅ Config loaded")
print(f"   Analysis model  : {ANALYSIS_MODEL}")
print(f"   Validation model: {VALIDATION_MODEL}")
print(f"   SAS file        : {SAS_FILE_PATH}")


# ============================================================
# SECTION 2 — LLM Service Functions  (mirrors services/llm.py)
# ============================================================

def _get_client(api_key: str, model: str):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name=model)


def call_llm_validated(prompt: str, api_key: str, model: str, retries: int = 3) -> dict:
    """
    Call Gemini with JSON output mode enforced.
    Used for: Complexity Classifier, Planning LLM, Plan Validator, Code Validator.
    """
    client = _get_client(api_key, model)
    config = GenerationConfig(response_mime_type="application/json", temperature=0.2)
    last_error = None
    for attempt in range(retries):
        try:
            response = client.generate_content(prompt, generation_config=config)
            text = response.text.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
            return json.loads(text)
        except (json.JSONDecodeError, Exception) as e:
            last_error = e
            print(f"   ⚠️  call_llm_validated attempt {attempt+1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
    raise ValueError(f"call_llm_validated failed after {retries} retries: {last_error}")


def call_llm_freeform(prompt: str, api_key: str, model: str) -> str:
    """
    Call Gemini for free-form text (Python code) output.
    Used for: Code Generation cells.
    """
    client = _get_client(api_key, model)
    config = GenerationConfig(temperature=0.3)
    response = client.generate_content(prompt, generation_config=config)
    text = response.text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if len(lines) > 2:
            text = "\n".join(lines[1:-1])
    return text


print("\n✅ LLM service functions defined")
print("   call_llm_validated → JSON mode (Classifier / Validators)")
print("   call_llm_freeform  → Plain text (Code Gen)")


# ============================================================
# SECTION 3 — LLM Prompt Templates  (mirrors backend/prompts/)
# ============================================================

CLASSIFIER_PROMPT = """\
You are a SAS-to-Databricks complexity classifier.

Analyse the SAS code provided and return a JSON object with EXACTLY this structure:
{{
  "complexity_level": "low" | "medium" | "high",
  "complexity_score": <integer 1-10>,
  "complexity_reasoning": "<one paragraph explaining the score>",
  "libnames": [
    {{"alias": "<libname alias>", "source_hint": "<detected source system e.g. Netezza, Oracle, SAS dataset>"}}
  ],
  "procs": ["<list of SAS PROC types found>"],
  "macros": [
    {{
      "name": "<macro name including %>",
      "complexity": "simple" | "complex",
      "is_flagged": true | false,
      "flag_reason": "<reason or null>"
    }}
  ],
  "cross_source_joins": true | false,
  "flagged_constructs": [
    {{"construct": "<name>", "lines": "<range>", "reason": "<why flagged>"}}
  ]
}}

Flag: %include, PROC FORMAT, PROC TRANSPOSE, PROC MEANS/SUMMARY, PROC REPORT,
nested macros (depth>2), SAS date functions, any macro judged too complex.

Return ONLY the JSON. No prose, no markdown fences.

SAS CODE:
{sas_code}
"""

PREVIOUS_ATTEMPTS_TEMPLATE = """
PREVIOUS PLANNING ATTEMPTS (you have failed {n} times — study your mistakes):
{issues_summary}
Ensure you explicitly address each blocking issue listed above.
"""

PLANNER_PROMPT = """\
You are a SAS-to-Databricks planning expert. Create a step-by-step execution plan.

Return ONLY a JSON object with EXACTLY this structure:
{{
  "plan_steps": [
    {{
      "step_id": <int>,
      "step_type": "extract_stage" | "join" | "transform" | "output",
      "sas_source_block": "<reference e.g. PROC SQL block 1>",
      "instruction": "<precise English instruction for code generation>",
      "source_libname": "<libname alias or null>",
      "target_table": "<fully qualified table name or null>",
      "cross_source": true | false,
      "flagged": false,
      "flag_reason": null,
      "edited_by_user": false
    }}
  ],
  "planning_reasoning": "<chain-of-thought explaining your plan>"
}}

CRITICAL RULES:
- Every cross-source join MUST produce TWO steps: extract_stage then join.
- extract_stage target_table must be: <catalog>.<schema>_stage.<table>_stage
- Instructions must be fully self-contained — no SAS syntax.
- Steps in execution order.

SAS CODE:
{sas_code}

CLASSIFIER OUTPUT:
{classifier_output}

TRANSLATION MANIFEST:
{manifest}

MACRO RESOLUTIONS:
{macro_resolutions}

{previous_attempts_summary}
"""

PLAN_VALIDATOR_PROMPT = """\
You are a senior SAS-to-Databricks plan validator.

Return ONLY a JSON object with EXACTLY this structure:
{{
  "validation_pass": true | false,
  "issues": [
    {{"step_id": <int or null>, "severity": "blocking" | "warning", "issue": "<description>"}}
  ]
}}

VALIDATION RULES:
1. Every cross-source join must have a preceding extract_stage step.
2. Staging table names must be referenced exactly in the join step.
3. All libname aliases in the SAS code have corresponding steps.
4. Macro instructions are reflected in step instructions.
5. Steps in logical execution order.
6. No SAS syntax in any instruction.
7. No table spanning two libname aliases without a preceding extract_stage.

Set validation_pass to true only if there are NO blocking issues.

PROPOSED PLAN:
{plan_output}

ORIGINAL SAS CODE:
{sas_code}
"""

CODE_GEN_PROMPT = """\
You are a SAS-to-Databricks code generation expert. Generate ONE Databricks Python notebook cell.

OUTPUT FORMAT — return ONLY Python code (no markdown fences, no explanation).
Start with this exact header:
# COMMAND ----------
# Step {step_id} of {total_steps} | Type: {step_type}
# Instruction: {instruction_truncated}
# SAS Source: {sas_source_block}
# Validator: Approved after {{N}} loop(s)
# Flag: None

Then the PySpark code implementing the instruction.

RULES:
- Use PySpark (spark.read, spark.sql, DataFrame API).
- extract_stage: spark.read.format("jdbc") → write to Delta as saveAsTable.
- Credential values use angle-bracket placeholders: <JDBC_URL>, <USERNAME>, <PASSWORD>.
- Staging table name in Cell A MUST exactly match what Cell B references.
- Do NOT include prose — only the header comment block + Python code.

STEP INFO:
Step ID       : {step_id}
Total Steps   : {total_steps}
Step Type     : {step_type}
Instruction   : {instruction}
SAS Source Ref: {sas_source_block}
Target Table  : {target_table}

SAS SOURCE CODE BLOCK:
{sas_block}

{retry_context}
"""

RETRY_CONTEXT_TEMPLATE = """
PREVIOUS ATTEMPT FAILED — Code Validator issues (attempt {attempt} of 5):
{issues}
Fix all listed issues. Do not repeat the same mistakes.
"""

CODE_VALIDATOR_PROMPT = """\
You are a Databricks code validator.

Return ONLY a JSON object with EXACTLY this structure:
{{
  "cell_pass": true | false,
  "issues": [
    {{"severity": "blocking" | "warning", "issue": "<description>"}}
  ],
  "validator_reasoning": "<brief explanation>"
}}

CHECKS:
1. Cell starts with # COMMAND ---------- header.
2. Step type matches code pattern.
3. extract_stage: staging table name in write matches target_table exactly.
4. join: references the exact staging table from extract_stage.
5. No SAS syntax in generated Python.
6. JDBC credentials use angle-bracket placeholders.
7. Instruction intent is fully satisfied.

INSTRUCTION:
{instruction}

TARGET TABLE:
{target_table}

GENERATED CELL CODE:
{cell_code}
"""

print("\n✅ All 5 LLM prompt templates defined")


# ============================================================
# SECTION 4 — Load SAS File
# ============================================================

with open(SAS_FILE_PATH, "r", encoding="utf-8", errors="replace") as f:
    SAS_CODE = f.read()

print(f"\n✅ Loaded SAS file: {SAS_FILE_PATH}")
print(f"   Lines : {len(SAS_CODE.splitlines())}")
print(f"   Chars : {len(SAS_CODE):,}")
print()
print("─" * 60)
print("FIRST 40 LINES:")
print("─" * 60)
print("\n".join(SAS_CODE.splitlines()[:40]))


# ============================================================
# SECTION 5 — Stage 1: Complexity Classification
# ============================================================

print("\n" + "=" * 60)
print("STAGE 1: COMPLEXITY CLASSIFICATION")
print("=" * 60)
print(f"  Model: {ANALYSIS_MODEL}")

classifier_prompt = CLASSIFIER_PROMPT.format(sas_code=SAS_CODE)
CLASSIFIER_OUTPUT = call_llm_validated(classifier_prompt, API_KEY, ANALYSIS_MODEL)

print("\n─" * 60)
print("CLASSIFIER OUTPUT")
print("─" * 60)
print(f"  Complexity Level : {CLASSIFIER_OUTPUT.get('complexity_level', '?').upper()}")
print(f"  Complexity Score : {CLASSIFIER_OUTPUT.get('complexity_score', '?')}/10")
print(f"  Reasoning        : {CLASSIFIER_OUTPUT.get('complexity_reasoning', '')[:250]}...")

libnames = CLASSIFIER_OUTPUT.get("libnames", [])
print(f"\n  Libnames ({len(libnames)}):")
for lb in libnames:
    print(f"    {lb['alias']:20s}  ← {lb['source_hint']}")

procs = CLASSIFIER_OUTPUT.get("procs", [])
print(f"\n  PROCs: {', '.join(procs) if procs else 'none'}")

macros = CLASSIFIER_OUTPUT.get("macros", [])
print(f"\n  Macros ({len(macros)}):")
for m in macros:
    flag = "  ⚠️  FLAGGED" if m.get("is_flagged") else ""
    print(f"    {m['name']:30s}  [{m['complexity']}]{flag}")
    if m.get("flag_reason"):
        print(f"      reason: {m['flag_reason']}")

flagged_constructs = CLASSIFIER_OUTPUT.get("flagged_constructs", [])
if flagged_constructs:
    print(f"\n  ⚠️  Flagged Constructs ({len(flagged_constructs)}):")
    for fc in flagged_constructs:
        print(f"    {fc['construct']} (lines {fc['lines']}): {fc['reason']}")

cross = CLASSIFIER_OUTPUT.get("cross_source_joins", False)
print(f"\n  Cross-source joins detected: {'YES ⚠️' if cross else 'NO'}")
print("\n✅ Classification complete")


# ============================================================
# SECTION 6 — Stage 2: Source Mapping (edit and re-run)
# ============================================================

print("\n" + "=" * 60)
print("STAGE 2: SOURCE MAPPING")
print("=" * 60)

# ─── EDIT THIS ──────────────────────────────────────────────
LIBNAME_MAPPINGS = [
    # Example — fill in one dict per libname:
    {
        "alias": "LIBNAME_ALIAS",          # must match classifier output
        "original_source": "Netezza",      # from classifier
        "target_catalog": "prod",          # Databricks catalog
        "target_schema": "analytics",      # Databricks schema
        "jdbc_type": "netezza",            # netezza|oracle|sqlserver|postgresql|databricks|other
        "keep_original": False,
    },
    # Add more entries here...
]
# ─────────────────────────────────────────────────────────────

TRANSLATION_MANIFEST = {"libname_mappings": LIBNAME_MAPPINGS}

print(f"   {len(LIBNAME_MAPPINGS)} libname(s) mapped:")
for m in LIBNAME_MAPPINGS:
    prefix = "  🔗 JDBC" if not m["keep_original"] else "  📁 Native"
    print(f"  {prefix}  {m['alias']:20s} → {m['target_catalog']}.{m['target_schema']}  [{m['jdbc_type']}]")

print("\n✅ Source mapping defined")


# ============================================================
# SECTION 7 — Stage 3: Macro Resolution (edit and re-run)
# ============================================================

print("\n" + "=" * 60)
print("STAGE 3: MACRO RESOLUTION")
print("=" * 60)

# ─── EDIT THIS ──────────────────────────────────────────────
MACRO_RESOLUTIONS = [
    # Examples:
    # {"macro_name": "%get_region", "resolution_type": "user_instruction",
    #  "instruction": "Apply WHERE region = 'WEST' to the query", "stop_workflow": False},
    # {"macro_name": "%load_dates", "resolution_type": "skip",
    #  "instruction": None, "stop_workflow": False},
]
# ─────────────────────────────────────────────────────────────

# Auto-default any unresolved macros to 'skip'
resolved_names = {r["macro_name"] for r in MACRO_RESOLUTIONS}
for macro in CLASSIFIER_OUTPUT.get("macros", []):
    if macro["name"] not in resolved_names:
        MACRO_RESOLUTIONS.append({
            "macro_name": macro["name"],
            "resolution_type": "skip",
            "instruction": None,
            "stop_workflow": False,
        })
        print(f"  ⚙️  Auto-defaulted {macro['name']} → skip")

MACRO_RESOLUTIONS_OBJ = {"resolutions": MACRO_RESOLUTIONS}

print("\n  Macro resolutions:")
for r in MACRO_RESOLUTIONS:
    tag = f"→ '{r['instruction']}'" if r["resolution_type"] == "user_instruction" else ""
    print(f"   {r['macro_name']:30s}  [{r['resolution_type']}] {tag}")

print("\n✅ Macro resolutions defined")


# ============================================================
# SECTION 8 — Stage 4+5: Plan Generation + Validation Loop
# ============================================================

print("\n" + "=" * 60)
print("STAGE 4+5: PLAN GENERATION + VALIDATION LOOP")
print("=" * 60)
print(f"  Planning model  : {ANALYSIS_MODEL}")
print(f"  Validator model : {VALIDATION_MODEL}")
print(f"  Max loops       : {MAX_PLAN_LOOPS}")


def build_planner_prompt(sas_code, classifier_output, manifest, macro_resolutions, previous_issues=None):
    prev_summary = ""
    if previous_issues:
        lines = []
        for i, attempt in enumerate(previous_issues, 1):
            for issue in attempt.get("issues", []):
                lines.append(
                    f"  Attempt {i}: [{issue.get('severity','?')}] "
                    f"step {issue.get('step_id','?')} — {issue.get('issue','?')}"
                )
        prev_summary = PREVIOUS_ATTEMPTS_TEMPLATE.format(
            n=len(previous_issues),
            issues_summary="\n".join(lines)
        )
    return PLANNER_PROMPT.format(
        sas_code=sas_code,
        classifier_output=json.dumps(classifier_output, indent=2),
        manifest=json.dumps(manifest, indent=2),
        macro_resolutions=json.dumps(macro_resolutions, indent=2),
        previous_attempts_summary=prev_summary,
    )


LOOP_TRACE = []
previous_issues = []
PLAN_OUTPUT = None
LAST_VALIDATION = None
PLAN_STATUS = "pending"

for loop_num in range(1, MAX_PLAN_LOOPS + 1):
    print(f"\n{'─'*60}")
    print(f"  LOOP {loop_num} of {MAX_PLAN_LOOPS}")
    print(f"{'─'*60}")

    # --- Planning LLM ---
    print(f"  ⏳ Calling Planning LLM ({ANALYSIS_MODEL})...")
    plan_prompt = build_planner_prompt(
        SAS_CODE, CLASSIFIER_OUTPUT, TRANSLATION_MANIFEST, MACRO_RESOLUTIONS_OBJ,
        previous_issues=previous_issues if loop_num > 1 else None,
    )
    plan_result = call_llm_validated(plan_prompt, API_KEY, ANALYSIS_MODEL)

    steps = plan_result.get("plan_steps", [])
    reasoning = plan_result.get("planning_reasoning", "")[:300]
    print(f"  ✅ Plan generated: {len(steps)} step(s)")
    print(f"  📝 Reasoning: {reasoning}...")
    print()
    for s in steps:
        cross = "[cross-source]" if s.get("cross_source") else ""
        print(f"    Step {s['step_id']:2d}  [{s['step_type']:15s}] {cross}")
        print(f"           {s['instruction'][:100]}")

    # --- Plan Validator LLM ---
    print(f"\n  ⏳ Calling Plan Validator ({VALIDATION_MODEL})...")
    val_prompt = PLAN_VALIDATOR_PROMPT.format(
        plan_output=json.dumps(plan_result, indent=2),
        sas_code=SAS_CODE,
    )
    val_result = call_llm_validated(val_prompt, API_KEY, VALIDATION_MODEL)

    passed = val_result.get("validation_pass", False)
    issues = val_result.get("issues", [])
    blocking = [i for i in issues if i.get("severity") == "blocking"]
    warnings = [i for i in issues if i.get("severity") == "warning"]

    status_icon = "✅" if passed else "❌"
    print(f"  {status_icon} Validation {'PASSED' if passed else 'FAILED'}  "
          f"| {len(blocking)} blocking, {len(warnings)} warnings")

    if blocking:
        print("  ⛔ Blocking issues:")
        for issue in blocking:
            print(f"     Step {issue.get('step_id','?')}: {issue.get('issue','?')}")

    if warnings:
        print("  ⚠️  Warnings:")
        for issue in warnings:
            print(f"     Step {issue.get('step_id','?')}: {issue.get('issue','?')}")

    LOOP_TRACE.append({"loop": loop_num, "plan": plan_result, "validation": val_result})

    if passed or not blocking:
        PLAN_OUTPUT = plan_result
        LAST_VALIDATION = val_result
        PLAN_STATUS = "passed"
        print(f"\n✅ Plan passed validation on loop {loop_num}!")
        break

    previous_issues.append(val_result)

else:
    last = LOOP_TRACE[-1]
    PLAN_OUTPUT = last["plan"]
    LAST_VALIDATION = last["validation"]
    PLAN_STATUS = "max_loops_reached"
    print(f"\n⚠️  Max loops ({MAX_PLAN_LOOPS}) reached without full validation pass.")
    print("   Proceeding with last plan — review blocking issues above before continuing.")

print(f"\n📊 Loop summary: {len(LOOP_TRACE)} loop(s) run | Status: {PLAN_STATUS}")
print(f"   Final plan has {len(PLAN_OUTPUT.get('plan_steps', []))} steps")


# ============================================================
# SECTION 9 — Stage 6: Plan Review & Edit
# ============================================================

print("\n" + "=" * 60)
print("STAGE 6: PLAN REVIEW")
print("=" * 60)

APPROVED_PLAN = copy.deepcopy(PLAN_OUTPUT)
plan_steps = APPROVED_PLAN["plan_steps"]

for step in plan_steps:
    edited = " [EDITED]" if step.get("edited_by_user") else ""
    cross = " [cross-source]" if step.get("cross_source") else ""
    print(f"\nStep {step['step_id']:2d} | {step['step_type']:15s}{cross}{edited}")
    print(f"  SAS ref    : {step['sas_source_block']}")
    print(f"  Instruction: {step['instruction']}")
    if step.get("target_table"):
        print(f"  Target     : {step['target_table']}")


def edit_step(step_id: int, new_instruction: str):
    """Edit a plan step's instruction. Call before run_final_validation()."""
    for step in plan_steps:
        if step["step_id"] == step_id:
            old = step["instruction"]
            step["instruction"] = new_instruction
            step["edited_by_user"] = True
            print(f"✅ Step {step_id} updated")
            print(f"   OLD: {old[:120]}")
            print(f"   NEW: {new_instruction[:120]}")
            return
    print(f"❌ Step {step_id} not found")


def run_final_validation():
    """Run the Plan Validator against the current APPROVED_PLAN."""
    print(f"\n⏳ Running final Plan Validator pass ({VALIDATION_MODEL})...")
    val_prompt = PLAN_VALIDATOR_PROMPT.format(
        plan_output=json.dumps(APPROVED_PLAN, indent=2),
        sas_code=SAS_CODE,
    )
    result = call_llm_validated(val_prompt, API_KEY, VALIDATION_MODEL)
    passed = result.get("validation_pass", False)
    blocking = [i for i in result.get("issues", []) if i.get("severity") == "blocking"]
    icon = "✅" if passed else "❌"
    print(f"{icon} Final validation {'PASSED' if passed else 'FAILED'}  | {len(blocking)} blocking")
    for issue in blocking:
        print(f"  ⛔ Step {issue.get('step_id','?')}: {issue.get('issue','?')}")
    if not blocking:
        print("  Plan is approved — safe to proceed to code generation.")
    return result


# To edit and re-validate:
# edit_step(step_id=2, new_instruction="Left join member_dim_stage with claims_fact on member_id")
# run_final_validation()

print("\n✅ Plan review complete. Edit steps above if needed then proceed.")


# ============================================================
# SECTION 10 — Stage 7: Code Generation + Per-Cell Validation
# ============================================================

print("\n" + "=" * 60)
print("STAGE 7: CODE GENERATION + PER-CELL VALIDATION")
print("=" * 60)
print(f"  Code Gen model      : {ANALYSIS_MODEL}")
print(f"  Code Validator model: {VALIDATION_MODEL}")
print(f"  Max retries / cell  : {MAX_CELL_RETRIES}")

total_steps = len(APPROVED_PLAN["plan_steps"])
GENERATED_CELLS = []

for step in APPROVED_PLAN["plan_steps"]:
    print(f"\n{'─'*60}")
    print(f"  Step {step['step_id']} of {total_steps} | [{step['step_type']}]")
    print(f"  {step['instruction'][:100]}...")

    cell_code = None
    final_status = "flagged"
    flag_reason = None
    retry_context = ""

    for attempt in range(1, MAX_CELL_RETRIES + 1):
        print(f"\n  ⏳ Attempt {attempt}/{MAX_CELL_RETRIES} — generating code ({ANALYSIS_MODEL})...")

        # Code Gen (freeform)
        gen_prompt = CODE_GEN_PROMPT.format(
            step_id=step["step_id"],
            total_steps=total_steps,
            step_type=step["step_type"],
            instruction=step["instruction"],
            instruction_truncated=step["instruction"][:200],
            sas_source_block=step["sas_source_block"],
            target_table=step.get("target_table") or "N/A",
            sas_block=SAS_CODE,
            retry_context=retry_context,
        )
        try:
            generated = call_llm_freeform(gen_prompt, API_KEY, ANALYSIS_MODEL)
        except Exception as e:
            print(f"  ❌ Code generation failed: {e}")
            flag_reason = str(e)
            break

        print(f"  📝 Generated {len(generated.splitlines())} lines of code")

        # Code Validator (validated)
        print(f"  ⏳ Validating code ({VALIDATION_MODEL})...")
        val_prompt = CODE_VALIDATOR_PROMPT.format(
            instruction=step["instruction"],
            target_table=step.get("target_table") or "N/A",
            cell_code=generated,
        )
        try:
            val_result = call_llm_validated(val_prompt, API_KEY, VALIDATION_MODEL)
        except Exception as e:
            print(f"  ❌ Validation call failed: {e}")
            flag_reason = str(e)
            cell_code = generated
            break

        cell_passed = val_result.get("cell_pass", False)
        val_issues = val_result.get("issues", [])
        blocking = [i for i in val_issues if i.get("severity") == "blocking"]
        print(f"  {'✅' if cell_passed else '❌'} Validation {'PASSED' if cell_passed else 'FAILED'} "
              f"| {len(blocking)} blocking")
        print(f"     Reasoning: {val_result.get('validator_reasoning','')[:120]}")

        if cell_passed:
            cell_code = generated.replace("Approved after {N} loop(s)", f"Approved after {attempt} loop(s)")
            final_status = "approved"
            print(f"  ✅ APPROVED on attempt {attempt}")
            break
        else:
            issue_lines = "\n".join(f"  [{i.get('severity')}] {i.get('issue')}" for i in val_issues)
            print(f"  Issues:\n{issue_lines}")
            retry_context = RETRY_CONTEXT_TEMPLATE.format(attempt=attempt, issues=issue_lines)
            cell_code = generated
            flag_reason = issue_lines

    # Still flagged after retries
    if final_status == "flagged" and cell_code:
        cell_code = re.sub(r"# Validator:.*", "# Validator: FLAGGED - see below", cell_code)
        cell_code = re.sub(r"# Flag: None", f"# Flag: {flag_reason or 'Max retries exceeded'}", cell_code)
        print(f"  ⛔ FLAGGED after {MAX_CELL_RETRIES} attempts")

    GENERATED_CELLS.append({
        "step_id": step["step_id"],
        "status": final_status,
        "cell_code": cell_code or f"# COMMAND ----------\n# Step {step['step_id']} | FLAGGED\n# Flag: {flag_reason}",
        "flag_reason": flag_reason,
    })

approved_cells = [c for c in GENERATED_CELLS if c["status"] == "approved"]
flagged_cells  = [c for c in GENERATED_CELLS if c["status"] == "flagged"]
print(f"\n{'='*60}")
print("✅ CODE GENERATION COMPLETE")
print(f"   Total  : {len(GENERATED_CELLS)}")
print(f"   Approved : {len(approved_cells)}")
print(f"   Flagged  : {len(flagged_cells)}")
if flagged_cells:
    print(f"   ⛔ Flagged steps: {[c['step_id'] for c in flagged_cells]}")


# ============================================================
# SECTION 11 — Stage 8: Build & Save Output Notebook
# ============================================================

print("\n" + "=" * 60)
print("STAGE 8: BUILD & SAVE DATABRICKS NOTEBOOK")
print("=" * 60)


def build_notebook(generated_cells: list, sas_filename: str = "unknown.sas") -> str:
    """Assemble cells into a .py Databricks notebook string."""
    approved = [c for c in generated_cells if c["status"] == "approved"]
    flagged  = [c for c in generated_cells if c["status"] == "flagged"]

    run_summary = {
        "source_file": sas_filename,
        "total_plan_steps": len(generated_cells),
        "cells_approved": len(approved),
        "cells_flagged": len(flagged),
        "flagged_steps": [c["step_id"] for c in flagged],
        "complexity": CLASSIFIER_OUTPUT.get("complexity_level"),
        "complexity_score": CLASSIFIER_OUTPUT.get("complexity_score"),
        "cross_source_joins": CLASSIFIER_OUTPUT.get("cross_source_joins"),
        "plan_loops_used": len(LOOP_TRACE),
        "plan_status": PLAN_STATUS,
    }

    summary_str = json.dumps(run_summary, indent=2).replace("\n", "\n# ")
    header = (
        "# Databricks notebook source\n"
        "# Generated by SAS → Databricks Conversion Tool\n"
        "# \n"
        f"# RUN SUMMARY: {summary_str}\n"
        "# \n"
        "# Search for '<' to find all credential placeholders\n"
    )

    manifest_lines = ["# MANIFEST SUMMARY:"]
    for m in LIBNAME_MAPPINGS:
        arrow = f"{m['target_catalog']}.{m['target_schema']}" if not m["keep_original"] else "(keep original)"
        manifest_lines.append(f"#   {m['alias']} ({m['original_source']}) → {arrow}")
    header += "\n".join(manifest_lines) + "\n"

    body_parts = [header] + [c["cell_code"] for c in generated_cells]
    return "\n\n".join(body_parts)


notebook_text = build_notebook(GENERATED_CELLS, sas_filename=os.path.basename(SAS_FILE_PATH))

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(notebook_text)

print(f"✅ Notebook saved to: {OUTPUT_PATH}")
print(f"   Total lines: {len(notebook_text.splitlines()):,}")
print(f"   Total chars: {len(notebook_text):,}")
print(f"   Cells: {len(GENERATED_CELLS)} total | "
      f"{len(approved_cells)} approved | "
      f"{len(flagged_cells)} flagged")
print()
print("─" * 60)
print("NOTEBOOK PREVIEW (first 80 lines):")
print("─" * 60)
print("\n".join(notebook_text.splitlines()[:80]))


# ============================================================
# SECTION 12 — Inspect Generated Cells
# ============================================================

def show_cell(step_id: int):
    """Print the full generated cell code for a given step ID."""
    for cell in GENERATED_CELLS:
        if cell["step_id"] == step_id:
            icon = "✅" if cell["status"] == "approved" else "⛔"
            print(f"\n{icon} Cell {step_id} — {cell['status'].upper()}")
            if cell.get("flag_reason"):
                print(f"Flag reason: {cell['flag_reason']}")
            print("─" * 60)
            print(cell["cell_code"])
            print("─" * 60)
            return
    print(f"❌ No cell found for step_id={step_id}")


def show_all_cells():
    """Print a summary table of all generated cells."""
    print(f"\n{'Step':>5}  {'Status':12}  {'Type':15}  Instruction (truncated)")
    print("─" * 90)
    for cell in GENERATED_CELLS:
        step = next((s for s in APPROVED_PLAN["plan_steps"] if s["step_id"] == cell["step_id"]), {})
        icon = "✅" if cell["status"] == "approved" else "⛔"
        print(f"{cell['step_id']:>5}  {icon} {cell['status']:10}  "
              f"{step.get('step_type','?'):15}  "
              f"{step.get('instruction','')[:50]}...")


show_all_cells()
print("\nUse show_cell(step_id=1) to print any cell's full code.")
print("\n🏁 Pipeline complete.")
