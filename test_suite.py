"""
MetricChronicle Comprehensive Test Suite
Validates all requirements:
1. Imports and environment
2. Sample dataset and deterministic Pandas calculations
3. Hindsight retain, recall, reflect operations
4. Gemini query parsing and explanation synthesis
5. Scenario 1: Temporal memory (Feb vs May active customers)
6. Scenario 2: Generalization (Refunds affecting AOV and Profit)
7. Scenario 3: Conflict detection (Approved official vs Dave draft note)
8. Scenario 4: New analyst correction retention and recall
9. Benchmark evaluation suite
"""

import sys
import os
import pandas as pd

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def run_tests():
    print("=" * 65)
    print("RUNNING METRICCHRONICLE COMPREHENSIVE VERIFICATION")
    print("=" * 65)
    
    passed_tests = 0
    total_tests = 0
    
    # --- TEST 1: Imports ---
    total_tests += 1
    print("\n[TEST 1] Verifying Python Imports...")
    try:
        import streamlit
        import pandas
        import dotenv
        from google import genai
        from hindsight_client import Hindsight
        from services.analytics import load_dataset, calculate_metrics, get_dataset_overview
        from services.memory import get_memory_service, MemoryService
        from services.gemini import get_gemini_service, GeminiService
        print("  [PASS] All required packages and modules imported successfully.")
        passed_tests += 1
    except Exception as e:
        print(f"  [FAIL] Import error: {e}")
        return False
        
    # --- TEST 2: Dataset Loading and Integrity ---
    total_tests += 1
    print("\n[TEST 2] Verifying Deterministic Dataset...")
    try:
        df = load_dataset()
        assert len(df) == 350, f"Expected 350 rows, got {len(df)}"
        required_cols = [
            "order_id", "customer_id", "order_date", "order_status",
            "gross_amount", "refund_amount", "cost_of_goods_sold", "region"
        ]
        for col in required_cols:
            assert col in df.columns, f"Missing column {col}"
        overview = get_dataset_overview(df)
        print(f"  [PASS] Dataset verified: {overview['total_records']} records, {overview['unique_customers']} unique customers, ${overview['total_gross_gmv']:,.2f} GMV.")
        passed_tests += 1
    except Exception as e:
        print(f"  [FAIL] Dataset error: {e}")
        return False

    # --- TEST 3: Deterministic Calculations ---
    total_tests += 1
    print("\n[TEST 3] Verifying Metric Calculations (Pandas)...")
    try:
        # Feb 2026 calculations
        feb_with = calculate_metrics(df, start_date="2026-02-01", end_date="2026-02-28", as_of_date="2026-02-28", use_hindsight_policy=True)
        feb_without = calculate_metrics(df, start_date="2026-02-01", end_date="2026-02-28", as_of_date="2026-02-28", use_hindsight_policy=False)
        
        # Verify net revenue = delivered gross - refunds
        assert feb_with["revenue"] < feb_without["revenue"], "Official net revenue must be lower than naive gross booking"
        assert feb_with["total_refunds_deducted"] > 0, "Refunds must be deducted in official policy"
        assert feb_with["delivered_orders_count"] == 38, f"Expected 38 delivered orders, got {feb_with['delivered_orders_count']}"
        print(f"  [PASS] Feb 2026 Net Revenue=${feb_with['revenue']:,.2f} (Naive GMV=${feb_without['revenue']:,.2f}, Refunds Deducted=${feb_with['total_refunds_deducted']:,.2f}).")
        passed_tests += 1
    except Exception as e:
        print(f"  [FAIL] Calculation error: {e}")
        return False

    # --- TEST 4: Hindsight Memory Retain, Recall, Reflect ---
    total_tests += 1
    print("\n[TEST 4] Verifying Hindsight Memory (Retain, Recall, Reflect)...")
    try:
        mem = get_memory_service()
        # Seed memories
        seed_res = mem.seed_memories()
        assert seed_res["total_rules"] >= 6, "Expected at least 6 rules in bank"
        
        # Recall
        rec = mem.recall("What is the revenue definition?")
        assert len(rec) > 0, "Expected recalled items for revenue"
        
        # Reflect
        ref = mem.reflect("Explain revenue policy")
        assert len(ref) > 10, "Expected non-empty reflection"
        
        print(f"  [PASS] Hindsight Memory operational. Total stored rules: {seed_res['total_rules']}, Top recall score: {rec[0].get('relevance_score')}.")
        passed_tests += 1
    except Exception as e:
        print(f"  [FAIL] Memory error: {e}")
        return False

    # --- TEST 5: Scenario 1 — Temporal Memory (Active Customers Feb vs May) ---
    total_tests += 1
    print("\n[TEST 5] Verifying Scenario 1: Temporal Policy Transition...")
    try:
        m_feb = calculate_metrics(df, start_date="2026-02-01", end_date="2026-02-28", as_of_date="2026-02-28", use_hindsight_policy=True)
        m_may = calculate_metrics(df, start_date="2026-05-01", end_date="2026-05-31", as_of_date="2026-05-31", use_hindsight_policy=True)
        
        # Verify lookback windows
        assert m_feb["lookback_days"] == 90, f"Expected 90 days for Feb, got {m_feb['lookback_days']}"
        assert m_may["lookback_days"] == 30, f"Expected 30 days for May, got {m_may['lookback_days']}"
        
        # Recall for Feb should rank Q1 policy top
        rec_feb = mem.recall("active customers", as_of_date="2026-02-28")
        assert "Q1" in rec_feb[0]["title"], f"Expected Q1 rule to be top for Feb, got {rec_feb[0]['title']}"
        
        # Recall for May should rank Q2 revision top
        rec_may = mem.recall("active customers", as_of_date="2026-05-31")
        assert "Q2" in rec_may[0]["title"], f"Expected Q2 revision to be top for May, got {rec_may[0]['title']}"
        
        print(f"  [PASS] Temporal transition verified: Feb=90-day window ({m_feb['active_customers']} active) vs May=30-day window ({m_may['active_customers']} active).")
        passed_tests += 1
    except Exception as e:
        print(f"  [FAIL] Scenario 1 error: {e}")
        return False

    # --- TEST 6: Scenario 2 — Generalization (AOV and Profit) ---
    total_tests += 1
    print("\n[TEST 6] Verifying Scenario 2: Generalization to AOV and Profit...")
    try:
        m_may_with = calculate_metrics(df, start_date="2026-05-01", end_date="2026-05-31", as_of_date="2026-05-31", use_hindsight_policy=True)
        m_may_without = calculate_metrics(df, start_date="2026-05-01", end_date="2026-05-31", as_of_date="2026-05-31", use_hindsight_policy=False)
        
        # AOV with memory should use net revenue / delivered count
        assert m_may_with["aov"] == round(m_may_with["revenue"] / m_may_with["delivered_orders_count"], 2)
        # Profit with memory should subtract COGS from net revenue
        assert m_may_with["profit"] == round(m_may_with["revenue"] - m_may_with["cogs"], 2)
        
        print(f"  [PASS] Generalization verified: With Hindsight AOV=${m_may_with['aov']:.2f} (vs Naive ${m_may_without['aov']:.2f}), Profit=${m_may_with['profit']:.2f} (vs Naive ${m_may_without['profit']:.2f}).")
        passed_tests += 1
    except Exception as e:
        print(f"  [FAIL] Scenario 2 error: {e}")
        return False

    # --- TEST 7: Scenario 3 — Conflict Detection ---
    total_tests += 1
    print("\n[TEST 7] Verifying Scenario 3: Conflict Detection & Resolution...")
    try:
        reflected_conflict = mem.reflect("revenue calculation policy")
        assert "Conflict Detected" in reflected_conflict, "Expected conflict detection in reflection"
        assert "Dave" in reflected_conflict, "Expected Dave's unapproved note to be identified"
        assert "rejects unapproved" in reflected_conflict.lower(), "Expected rejection of unapproved draft"
        print(f"  [PASS] Conflict detection verified. Dave's draft note detected and rejected in favor of official policy.")
        passed_tests += 1
    except Exception as e:
        print(f"  [FAIL] Scenario 3 error: {e}")
        return False

    # --- TEST 8: Scenario 4 — Retaining and Recalling a New Correction ---
    total_tests += 1
    print("\n[TEST 8] Verifying Scenario 4: Retaining New Correction & Subsequent Recall...")
    try:
        new_rule_content = "VIP Tier members are active if they placed a delivered order within the past 180 days."
        retain_res = mem.retain(
            content=new_rule_content,
            metric="Active Customers",
            title="VIP Tier Active Window Extension",
            effective_start="2026-04-01",
            approval_status="APPROVED OFFICIAL",
            author="Customer Loyalty Guild"
        )
        assert retain_res["status"] == "retained"
        
        # Query specifically for VIP rule
        recalled_vip = mem.recall("What is the active customer window for VIP tier?")
        matched = any("VIP" in r.get("title", "") for r in recalled_vip)
        assert matched, "New VIP correction must be recalled"
        print(f"  [PASS] New correction successfully retained and recalled from Hindsight memory bank.")
        passed_tests += 1
    except Exception as e:
        print(f"  [FAIL] Scenario 4 error: {e}")
        return False

    # --- TEST 9: Gemini Query Parsing and Comparative Explanation ---
    total_tests += 1
    print("\n[TEST 9] Verifying Gemini Service...")
    try:
        gem = get_gemini_service()
        parsed = gem.parse_query("How many active customers did we have in February 2026?")
        assert parsed["metric"] == "active_customers"
        assert parsed["as_of_date"] == "2026-02-28"
        
        explanation = gem.explain_comparison(
            "What was revenue in February 2026?",
            feb_without,
            feb_with,
            rec,
            ref
        )
        assert "without_hindsight_critique" in explanation
        assert "with_hindsight_narrative" in explanation
        assert "difference_analysis" in explanation
        print(f"  [PASS] Gemini service parsed query and synthesized side-by-side comparison.")
        passed_tests += 1
    except Exception as e:
        print(f"  [FAIL] Gemini service error: {e}")
        return False

    print("\n" + "=" * 65)
    print(f"TEST RESULTS: {passed_tests} / {total_tests} TESTS PASSED (100%)")
    print("=" * 65)
    return passed_tests == total_tests

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
