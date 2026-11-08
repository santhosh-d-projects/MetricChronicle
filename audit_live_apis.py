"""
MetricChronicle Live Integration Audit
Performs a rigorous, transparent audit of:
1. Local credentials & Live Cloud API endpoints (Gemini & Hindsight Cloud)
2. Ground-truth numerical calculations against the Pandas dataset
3. All 4 core demonstration scenarios (Temporal, Generalization, Conflict, Learning)
4. Streamlit application launch and HTTP health probe
"""

import sys
import os
import subprocess
import time
from dotenv import load_dotenv

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv(".env")

def audit():
    print("=" * 70)
    print("METRICCHRONICLE LIVE INTEGRATION AUDIT")
    print("=" * 70)

    audit_results = {}

    # -------------------------------------------------------------
    # 1. AUDIT CREDENTIALS & LIVE GEMINI API
    # -------------------------------------------------------------
    print("\n[AUDIT 1] Live Google Gemini API Integration:")
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    is_gemini_placeholder = (
        not gemini_key or 
        gemini_key.startswith("YOUR_") or 
        "placeholder" in gemini_key.lower()
    )
    print(f"  - Key present: {bool(gemini_key)}")
    print(f"  - Key is placeholder: {is_gemini_placeholder}")
    
    gemini_live_success = False
    gemini_error_detail = ""
    try:
        from google import genai
        client = genai.Client(api_key=gemini_key)
        # Try primary model with fallback for transient 503 high demand
        candidate_models = ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.6-flash"]
        for cand in candidate_models:
            try:
                resp = client.models.generate_content(
                    model=cand,
                    contents="Respond with: LIVE_API_VERIFIED",
                )
                if resp and resp.text:
                    gemini_live_success = True
                    print(f"  [PASS] Live Gemini API connected via {cand}! Response: {resp.text.strip()[:60]}")
                    break
            except Exception as e_model:
                gemini_error_detail = str(e_model)[:250]
                if "503" in str(e_model) or "429" in str(e_model):
                    continue
                else:
                    raise e_model
        if not gemini_live_success:
            print(f"  [FAIL] Live Gemini API call failed: {gemini_error_detail}")
    except Exception as e:
        gemini_error_detail = str(e)[:250]
        print(f"  [FAIL] Live Gemini API call failed: {gemini_error_detail}")

    audit_results["gemini_live_api"] = {
        "passed": gemini_live_success,
        "is_placeholder": is_gemini_placeholder,
        "detail": "Live connection verified" if gemini_live_success else gemini_error_detail,
    }

    # -------------------------------------------------------------
    # 2. AUDIT CREDENTIALS & LIVE HINDSIGHT CLOUD API
    # -------------------------------------------------------------
    print("\n[AUDIT 2] Live Hindsight Cloud API Integration:")
    hindsight_key = os.getenv("HINDSIGHT_API_KEY", "")
    hindsight_base_url = os.getenv("HINDSIGHT_BASE_URL", "https://api.hindsight.vectorize.io")
    hindsight_bank = os.getenv("HINDSIGHT_BANK_ID", "metricchronicle")
    
    is_hindsight_placeholder = (
        not hindsight_key or 
        hindsight_key.startswith("YOUR_") or 
        "placeholder" in hindsight_key.lower()
    )
    print(f"  - Endpoint: {hindsight_base_url}")
    print(f"  - Bank ID: {hindsight_bank}")
    print(f"  - Key present: {bool(hindsight_key)}")
    print(f"  - Key is placeholder: {is_hindsight_placeholder}")

    hindsight_live_success = False
    hindsight_error_detail = ""
    for attempt in range(1, 4):
        try:
            from hindsight_client import Hindsight
            memory = Hindsight(base_url=hindsight_base_url, api_key=hindsight_key)
            
            # Test retain
            memory.retain(
                bank_id=hindsight_bank,
                content="MetricChronicle Audit Test: Net Revenue requires deducting refunds.",
                document_id="AUDIT-TEST-001"
            )
            # Test recall
            rec = memory.recall(bank_id=hindsight_bank, query="Net Revenue")
            # Test reflect
            ref = memory.reflect(bank_id=hindsight_bank, query="Summarize net revenue rule.")
            
            hindsight_live_success = True
            print(f"  [PASS] Live Hindsight Cloud API operational (Retain, Recall, Reflect).")
            memory.close()
            break
        except Exception as e:
            hindsight_error_detail = str(e)[:250]
            if attempt < 3:
                time.sleep(2)
            else:
                print(f"  [FAIL] Live Hindsight Cloud call failed: {hindsight_error_detail}")

    audit_results["hindsight_live_api"] = {
        "passed": hindsight_live_success,
        "is_placeholder": is_hindsight_placeholder,
        "detail": "Live Cloud Retain/Recall/Reflect verified" if hindsight_live_success else hindsight_error_detail,
    }

    # -------------------------------------------------------------
    # 3. AUDIT DATASET & NUMERICAL ACCURACY
    # -------------------------------------------------------------
    print("\n[AUDIT 3] Dataset & Deterministic Numerical Accuracy:")
    from services.analytics import load_dataset, calculate_metrics, get_dataset_overview
    df = load_dataset()
    overview = get_dataset_overview(df)
    print(f"  - Dataset Records: {overview['total_records']}")
    print(f"  - Date Range: {overview['date_min']} to {overview['date_max']}")
    print(f"  - Total GMV: ${overview['total_gross_gmv']:,.2f}")
    print(f"  - Total Refunds: ${overview['total_refunds']:,.2f}")

    # Verify Feb and May calculations
    feb_calc = calculate_metrics(df, start_date="2026-02-01", end_date="2026-02-28", as_of_date="2026-02-28", use_hindsight_policy=True)
    may_calc = calculate_metrics(df, start_date="2026-05-01", end_date="2026-05-31", as_of_date="2026-05-31", use_hindsight_policy=True)

    # Audited assertions
    num_tests_passed = True
    if feb_calc["revenue"] != 4337.16:
        num_tests_passed = False
        print(f"  [FAIL] Feb Revenue mismatch: expected 4337.16, got {feb_calc['revenue']}")
    if feb_calc["active_customers"] != 39:
        num_tests_passed = False
        print(f"  [FAIL] Feb Active Customers mismatch: expected 39, got {feb_calc['active_customers']}")
    if may_calc["active_customers"] != 33:
        num_tests_passed = False
        print(f"  [FAIL] May Active Customers mismatch: expected 33, got {may_calc['active_customers']}")
    if may_calc["aov"] != 136.08:
        num_tests_passed = False
        print(f"  [FAIL] May AOV mismatch: expected 136.08, got {may_calc['aov']}")
    if may_calc["profit"] != 3123.36:
        num_tests_passed = False
        print(f"  [FAIL] May Profit mismatch: expected 3123.36, got {may_calc['profit']}")

    if num_tests_passed:
        print("  [PASS] All deterministic calculations precisely match ground truth dataset.")
    audit_results["numerical_accuracy"] = {"passed": num_tests_passed}

    # -------------------------------------------------------------
    # 4. AUDIT CORE SCENARIOS & BUSINESS MEMORY LOGIC
    # -------------------------------------------------------------
    print("\n[AUDIT 4] Demonstration Scenarios:")
    from services.memory import get_memory_service
    mem = get_memory_service()

    # Scenario 1: Temporal definitions
    rec_feb = mem.recall("active customers", as_of_date="2026-02-28")
    rec_may = mem.recall("active customers", as_of_date="2026-05-31")
    s1_pass = ("Q1" in rec_feb[0]["title"]) and ("Q2" in rec_may[0]["title"])
    print(f"  - Scenario 1 (Temporal Shift Feb vs May): {'[PASS]' if s1_pass else '[FAIL]'}")

    # Scenario 2: Generalization
    s2_pass = (may_calc["aov"] == 136.08) and (may_calc["profit"] == 3123.36)
    print(f"  - Scenario 2 (Generalization to AOV & Profit): {'[PASS]' if s2_pass else '[FAIL]'}")

    # Scenario 3: Conflict detection
    refl_conflict = mem.reflect("revenue definition conflict")
    s3_pass = (
        ("conflict" in refl_conflict.lower()) and 
        ("official" in refl_conflict.lower()) and 
        ("reject" in refl_conflict.lower() or "supersede" in refl_conflict.lower() or "draft" in refl_conflict.lower())
    )
    print(f"  - Scenario 3 (Conflict Detection & Rejection): {'[PASS]' if s3_pass else '[FAIL]'}")

    # Scenario 4: Retain & Recall new correction
    mem.retain(
        content="Enterprise accounts have a 120-day active window.",
        metric="Active Customers",
        title="Enterprise Active Customer Exception",
        approval_status="APPROVED OFFICIAL",
        author="Enterprise Lead"
    )
    rec_ent = mem.recall("Enterprise accounts active window")
    s4_pass = any("Enterprise" in r.get("title", "") for r in rec_ent)
    print(f"  - Scenario 4 (Correction-based learning & recall): {'[PASS]' if s4_pass else '[FAIL]'}")

    audit_results["scenarios"] = {
        "scenario_1_temporal": s1_pass,
        "scenario_2_generalization": s2_pass,
        "scenario_3_conflict_detection": s3_pass,
        "scenario_4_new_correction": s4_pass,
    }

    # -------------------------------------------------------------
    # 5. AUDIT STREAMLIT APPLICATION LAUNCH & HEALTH PROBE
    # -------------------------------------------------------------
    print("\n[AUDIT 5] Streamlit Application Startup & Health Probe:")
    st_launched = False
    try:
        proc = subprocess.Popen(
            [r".venv\Scripts\streamlit.exe", "run", "app.py", "--server.port=8502", "--server.headless=true"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        time.sleep(6)
        
        import urllib.request
        req = urllib.request.urlopen("http://localhost:8502/_stcore/health", timeout=5)
        status_code = req.getcode()
        body = req.read().decode("utf-8")
        if status_code == 200 and "ok" in body:
            st_launched = True
            print(f"  [PASS] Streamlit launched successfully (HTTP {status_code} '{body.strip()}').")
        proc.terminate()
        proc.wait(timeout=5)
    except Exception as e:
        print(f"  [FAIL] Streamlit launch error: {e}")
        try:
            proc.kill()
        except Exception:
            pass

    audit_results["streamlit_launch"] = {"passed": st_launched}

    print("\n" + "=" * 70)
    print("AUDIT SUMMARY:")
    print(f"  - Live Gemini API: {'CONNECTED' if gemini_live_success else f'FAILED ({gemini_error_detail[:50]})'}")
    print(f"  - Live Hindsight Cloud: {'CONNECTED' if hindsight_live_success else f'FAILED ({hindsight_error_detail[:50]})'}")
    print(f"  - Dataset Numerical Accuracy: {'100% MATCH' if num_tests_passed else 'MISMATCH'}")
    print(f"  - Temporal Memory Scenario: {'PASS' if s1_pass else 'FAIL'}")
    print(f"  - Generalization Scenario: {'PASS' if s2_pass else 'FAIL'}")
    print(f"  - Conflict Detection Scenario: {'PASS' if s3_pass else 'FAIL'}")
    print(f"  - Correction Retention Scenario: {'PASS' if s4_pass else 'FAIL'}")
    print(f"  - Streamlit Application Startup: {'PASS (HTTP 200)' if st_launched else 'FAIL'}")
    print("=" * 70)
    
    return audit_results

if __name__ == "__main__":
    audit()
