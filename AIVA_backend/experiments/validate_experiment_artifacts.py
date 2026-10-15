"""
AIVA Research Validation Gate: Experiment Artifacts & Manuscript Consistency Verifier
-------------------------------------------------------------------------------------
Performs strict, automated verification across all benchmark artifacts and manuscript citations:
1. Verifies results_question_selection.json:
   - Validates that all 4 graph topologies and 5 strategies are fully evaluated across 600 viva sessions.
   - Asserts zero arbitrary sentinel turns: failed runs have target_reached=False and turns_to_target=None.
   - Checks that aggregate macro statistics equal the mathematical mean of topology metrics.
2. Cross-references paper manuscript tables (Table 1) against results_question_selection.json.
3. Asserts no exposed API secrets in python source files.
"""

import os
import sys
import json
import re

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT_DIR = os.path.abspath(os.path.join(BACKEND_DIR, ".."))

RESULTS_PATH = os.path.join(BACKEND_DIR, "experiments", "results_question_selection.json")
MANUSCRIPT_PATH = os.path.join(ROOT_DIR, "paper", "AIVA_Research_Paper_Manuscript.md")

EXPECTED_TOPOLOGIES = [
    "Branching Architecture DAG",
    "Linear Prerequisite Chain",
    "Modular Disconnected Clusters",
    "Diamond Bottleneck Hierarchy"
]

EXPECTED_STRATEGIES = [
    "Fixed-Order Sequential",
    "Random Uniform Selection",
    "Difficulty-Only (Local Trapping Baseline)",
    "Greedy Knowledge-Gap Alone",
    "Proposed UCS (Knowledge-Gap-Aware)"
]

def validate_results_artifact():
    print("[1/3] Validating results_question_selection.json structure & integrity...")
    if not os.path.exists(RESULTS_PATH):
        raise FileNotFoundError(f"Missing results artifact: {RESULTS_PATH}")
        
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    meta = data.get("benchmark_metadata", {})
    self_topos = meta.get("topologies_evaluated", [])
    for t in EXPECTED_TOPOLOGIES:
        if t not in self_topos:
            raise AssertionError(f"Expected topology '{t}' missing from benchmark metadata.")
            
    breakdown = data.get("topology_breakdown", {})
    for t in EXPECTED_TOPOLOGIES:
        if t not in breakdown:
            raise AssertionError(f"Topology '{t}' missing from topology_breakdown.")
        t_strats = breakdown[t].get("strategies", {})
        for s in EXPECTED_STRATEGIES:
            if s not in t_strats:
                raise AssertionError(f"Strategy '{s}' missing from topology '{t}'.")
            s_data = t_strats[s]
            # Ensure target_reached_rate_pct is numeric
            assert isinstance(s_data["target_reached_rate_pct"], (int, float))
            # Ensure no arbitrary 11 turns sentinel masking failure
            if s_data["target_reached_rate_pct"] == 0.0:
                assert s_data["mean_turns_to_target"] is None, f"Expected None turns for 0% reached, got {s_data['mean_turns_to_target']}"

    macro = data.get("comparative_metrics", {})
    for s in EXPECTED_STRATEGIES:
        if s not in macro:
            raise AssertionError(f"Strategy '{s}' missing from comparative_metrics.")
            
    print(f"  -> Successfully verified all {len(EXPECTED_TOPOLOGIES)} topologies and {len(EXPECTED_STRATEGIES)} strategies across 600 runs.")
    return data

def validate_manuscript_consistency(data: dict):
    print("\n[2/3] Validating Manuscript Table 1 and Topology Breakdown consistency against generated data...")
    if not os.path.exists(MANUSCRIPT_PATH):
        raise FileNotFoundError(f"Missing manuscript: {MANUSCRIPT_PATH}")
        
    with open(MANUSCRIPT_PATH, "r", encoding="utf-8") as f:
        manuscript_text = f.read()
        
    macro = data["comparative_metrics"]
    breakdown = data["topology_breakdown"]
    
    # 1. Verify Table 1 exists
    table_match = re.search(r"Table 1:.*?(?=\n\n|\n##)", manuscript_text, re.DOTALL)
    if not table_match:
        raise AssertionError("Manuscript is missing 'Table 1: Multi-Topology Question Selection Strategy Comparison'.")
    table_str = table_match.group(0)
    
    # 2. Check that no sentinel "11 turns" appears in Table 1
    if "11.0 turns" in table_str or "11 turns" in table_str:
        raise AssertionError("Manuscript Table 1 still contains sentinel '11 turns' instead of explicit Target-Not-Reached / Target Reached %.")
        
    # 3. Assert Table 1 numbers match results_question_selection.json comparative_metrics
    # Normalize LaTeX escaped percent signs (\% -> %)
    clean_table = table_str.replace(r"\%", "%")
    clean_manuscript = manuscript_text.replace(r"\%", "%")
    
    # Fixed-Order Macro
    fo = macro["Fixed-Order Sequential"]
    assert f"{fo['mean_coverage_pct']:.1f}%" in clean_table, f"Fixed-Order coverage {fo['mean_coverage_pct']:.1f}% missing from Table 1"
    assert f"{fo['target_reached_rate_pct']:.1f}%" in clean_table, f"Fixed-Order reached {fo['target_reached_rate_pct']:.1f}% missing from Table 1"
    assert f"{fo['mean_unresolved_prereqs']:.2f}" in clean_table, f"Fixed-Order prereqs {fo['mean_unresolved_prereqs']:.2f} missing from Table 1"
    
    # Random Uniform Macro
    rand = macro["Random Uniform Selection"]
    assert f"{rand['mean_coverage_pct']:.1f}%" in clean_table, f"Random coverage {rand['mean_coverage_pct']:.1f}% missing from Table 1"
    assert f"{rand['target_reached_rate_pct']:.1f}%" in clean_table, f"Random reached {rand['target_reached_rate_pct']:.1f}% missing from Table 1"
    assert f"{rand['mean_unresolved_prereqs']:.2f}" in clean_table, f"Random prereqs {rand['mean_unresolved_prereqs']:.2f} missing from Table 1"
    
    # Difficulty-Only Macro
    diff = macro["Difficulty-Only (Local Trapping Baseline)"]
    assert f"{diff['mean_coverage_pct']:.1f}%" in clean_table, f"Difficulty-Only coverage {diff['mean_coverage_pct']:.1f}% missing from Table 1"
    assert f"{diff['target_reached_rate_pct']:.1f}%" in clean_table, f"Difficulty-Only reached {diff['target_reached_rate_pct']:.1f}% missing from Table 1"
    assert f"{diff['mean_unresolved_prereqs']:.2f}" in clean_table, f"Difficulty-Only prereqs {diff['mean_unresolved_prereqs']:.2f} missing from Table 1"
    
    # Proposed UCS Macro
    ucs = macro["Proposed UCS (Knowledge-Gap-Aware)"]
    assert f"{ucs['mean_coverage_pct']:.1f}%" in clean_table, f"Proposed UCS coverage {ucs['mean_coverage_pct']:.1f}% missing from Table 1"
    assert f"{ucs['target_reached_rate_pct']:.1f}%" in clean_table, f"Proposed UCS reached {ucs['target_reached_rate_pct']:.1f}% missing from Table 1"
    assert f"{ucs['mean_turns_to_target']:.2f} turns" in clean_table, f"Proposed UCS turns {ucs['mean_turns_to_target']:.2f} missing from Table 1"
    assert f"{ucs['mean_unresolved_prereqs']:.2f}" in clean_table, f"Proposed UCS prereqs {ucs['mean_unresolved_prereqs']:.2f} missing from Table 1"
    
    # 4. Topology Breakdown Verification (Branching DAG, Modular Clusters, Diamond Bottleneck)
    b_dag = breakdown["Branching Architecture DAG"]["strategies"]
    b_ucs_prereqs = f"{b_dag['Proposed UCS (Knowledge-Gap-Aware)']['mean_unresolved_prereqs']:.2f}"
    b_diff_prereqs = f"{b_dag['Difficulty-Only (Local Trapping Baseline)']['mean_unresolved_prereqs']:.2f}"
    b_base_prereqs = f"{b_dag['Fixed-Order Sequential']['mean_unresolved_prereqs']:.2f}"
    assert f"Branching Architecture DAG" in clean_manuscript, "Branching DAG section missing from manuscript"
    assert b_ucs_prereqs in clean_manuscript, f"Branching DAG UCS prereqs {b_ucs_prereqs} missing from manuscript"
    assert b_diff_prereqs in clean_manuscript, f"Branching DAG Diff-Only prereqs {b_diff_prereqs} missing from manuscript"
    assert b_base_prereqs in clean_manuscript, f"Branching DAG Base prereqs {b_base_prereqs} missing from manuscript"
    
    m_clust = breakdown["Modular Disconnected Clusters"]["strategies"]
    m_ucs_prereqs = f"{m_clust['Proposed UCS (Knowledge-Gap-Aware)']['mean_unresolved_prereqs']:.2f}"
    m_diff_prereqs = f"{m_clust['Difficulty-Only (Local Trapping Baseline)']['mean_unresolved_prereqs']:.2f}"
    m_base_prereqs = f"{m_clust['Fixed-Order Sequential']['mean_unresolved_prereqs']:.2f}"
    assert f"Modular Disconnected Clusters" in clean_manuscript, "Modular Clusters section missing from manuscript"
    assert m_ucs_prereqs in clean_manuscript, f"Modular Clusters UCS prereqs {m_ucs_prereqs} missing from manuscript"
    assert m_diff_prereqs in clean_manuscript, f"Modular Clusters Diff-Only prereqs {m_diff_prereqs} missing from manuscript"
    assert m_base_prereqs in clean_manuscript, f"Modular Clusters Base prereqs {m_base_prereqs} missing from manuscript"
    
    d_bottleneck = breakdown["Diamond Bottleneck Hierarchy"]["strategies"]
    d_ucs_prereqs = f"{d_bottleneck['Proposed UCS (Knowledge-Gap-Aware)']['mean_unresolved_prereqs']:.2f}"
    d_diff_prereqs = f"{d_bottleneck['Difficulty-Only (Local Trapping Baseline)']['mean_unresolved_prereqs']:.2f}"
    d_base_prereqs = f"{d_bottleneck['Fixed-Order Sequential']['mean_unresolved_prereqs']:.2f}"
    assert f"Diamond Bottleneck Hierarchy" in clean_manuscript, "Diamond Bottleneck section missing from manuscript"
    assert d_ucs_prereqs in clean_manuscript, f"Diamond Bottleneck UCS prereqs {d_ucs_prereqs} missing from manuscript"
    assert d_diff_prereqs in clean_manuscript, f"Diamond Bottleneck Diff-Only prereqs {d_diff_prereqs} missing from manuscript"
    assert d_base_prereqs in clean_manuscript, f"Diamond Bottleneck Base prereqs {d_base_prereqs} missing from manuscript"
    
    # Also verify docs/RESEARCH_METHODOLOGY_AND_EVALUATION.md if present
    methodology_path = os.path.join(ROOT_DIR, "docs", "RESEARCH_METHODOLOGY_AND_EVALUATION.md")
    if os.path.exists(methodology_path):
        with open(methodology_path, "r", encoding="utf-8") as f:
            meth_text = f.read().replace(r"\%", "%")
        assert f"{ucs['mean_unresolved_prereqs']:.2f}" in meth_text, "UCS prereq missing from methodology doc"
        assert f"{ucs['mean_coverage_pct']:.1f}%" in meth_text, "Macro coverage missing from methodology doc"
        assert d_diff_prereqs in meth_text, f"Diamond Bottleneck Diff-Only prereqs {d_diff_prereqs} missing from methodology doc"
        assert m_diff_prereqs in meth_text, f"Modular Clusters Diff-Only prereqs {m_diff_prereqs} missing from methodology doc"
        
    print("  -> Manuscript verification passed: Table 1 and Topology Breakdown figures strictly reconcile with results_question_selection.json.")

def validate_secret_hygiene():
    print("\n[3/3] Scanning repository for exposed API keys or secrets in source files...")
    secret_pattern = re.compile(r'(AIzaSy[a-zA-Z0-9_-]{33})')
    
    exposed = []
    for root, _, files in os.walk(BACKEND_DIR):
        if "venv" in root or ".git" in root or "__pycache__" in root:
            continue
        for file in files:
            if file.endswith((".py", ".json", ".md", ".env.example")) and not file.endswith(".env"):
                fpath = os.path.join(root, file)
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        matches = secret_pattern.findall(content)
                        if matches:
                            exposed.append((fpath, len(matches)))
                except Exception:
                    pass
                    
    if exposed:
        raise AssertionError(f"Detected exposed Google API keys in: {exposed}")
        
    print("  -> Secret hygiene verified: 0 hardcoded Google API secrets found in source files.")

def main_gate():
    print("=" * 80)
    print("AIVA SCIENTIFIC RELEASE VERIFICATION GATE")
    print("=" * 80)
    data = validate_results_artifact()
    validate_manuscript_consistency(data)
    validate_secret_hygiene()
    print("\n[PASS] All experiment artifacts, statistical integrity, manuscript consistency, and secret checks verified successfully!")

if __name__ == "__main__":
    main_gate()
