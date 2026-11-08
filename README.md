# ⏱️ MetricChronicle

> **An AI Agent That Understands How Business Metrics Change Over Time.**

MetricChronicle is an AI business analytics agent that remembers company-specific metric definitions, analyst corrections, conflicting interpretations, and historical policy shifts using **Hindsight** memory banks and **Google Gemini**, grounded by deterministic **Pandas** calculations.

---

## 🎯 The Core Problem: Why Standard AI Fails Business Analytics

Standard LLMs lack **organizational memory**. When asked to compute standard business metrics, naive AI systems:
1. **Apply generic industry assumptions**: They calculate raw GMV instead of audited Net Revenue, ignoring returns and refunds.
2. **Fail temporal policy transitions**: They do not know that your company changed its "Active Customer" lookback window from 90 days in Q1 to 30 days in Q2.
3. **Cannot generalize internal accounting rules**: When taught that refunds reduce revenue, they fail to apply that deduction when computing Average Order Value (AOV) or Gross Profit.
4. **Silently hallucinate across conflicting interpretations**: When two analysts write conflicting definition notes, naive LLMs pick one arbitrarily without checking approval status or effective dates.

**MetricChronicle solves this by integrating Hindsight long-term memory into the analytics workflow.**

---

## 🏗️ Architecture & Technology Stack

```
MetricChronicle/
├── app.py                     # Streamlit Analytics Dashboard (3 Tabs: Ask, Memory Lab, Evaluation)
├── services/
│   ├── __init__.py
│   ├── analytics.py           # Deterministic Pandas engine & synthetic e-commerce data
│   ├── memory.py              # Official Hindsight SDK integration (retain, recall, reflect)
│   └── gemini.py              # Google GenAI SDK (google-genai) query parsing & explanations
├── data/
│   └── sample_orders.csv      # 350 deterministic e-commerce orders (Jan 1 – Jun 30, 2026)
├── test_suite.py              # 9 automated end-to-end tests covering all scenarios
├── test_connections.py       # API connection verification utility
├── requirements.txt           # Verified project dependencies
├── .env.example               # Environment template
└── README.md                  # Comprehensive documentation
```

### Core Components
- **Memory Engine (`services/memory.py`)**: Uses the official `hindsight-client` Python SDK. Implements `retain` (storing policies & corrections), `recall` (retrieving temporally relevant rules), and `reflect` (synthesizing conflict resolutions and policy audits) against the `metricchronicle` memory bank. Features a graceful offline-resilient store so demonstrations never crash.
- **Analytics Engine (`services/analytics.py`)**: 100% deterministic Pandas calculations. Gemini is strictly prohibited from inventing database numbers or executing arbitrary SQL.
- **Language Model (`services/gemini.py`)**: Uses `google-genai` (`gemini-2.5-flash`) to parse natural language questions into structured temporal filters and formulate comparative business explanations.
- **Interactive UI (`app.py`)**: Modern, dark-neutral Streamlit dashboard with side-by-side Without-Hindsight vs With-Hindsight comparison.

---

## 🚀 4 Essential Demonstration Scenarios

### Scenario 1 — Temporal Memory (Active Customer Policy Shift)
- **Question**: *"What was our active customer count in February 2026 compared to May 2026?"*
- **Ground Truth Policies**:
  - **Q1 Policy (Jan 1 – Mar 31)**: Active = at least 1 delivered purchase in preceding **90 days**.
  - **Q2 Policy Revision (Apr 1 onwards)**: Active = at least 1 delivered purchase in preceding **30 days**.
- **Without Hindsight**: Standard AI applies a static 30-day window to both months (Feb: 32 active, May: 38 active), misrepresenting Q1 quarterly retention.
- **With Hindsight**: Recalls the exact policy in effect for each date (Feb: 39 active with 90d window; May: 33 active with 30d window).

### Scenario 2 — Rule Generalization (Refunds in AOV & Profit)
- **Question**: *"What was our Average Order Value and Profit in May 2026?"*
- **Policy Learned**: Refunds reduce revenue, and cancelled orders are excluded.
- **Without Hindsight**: AOV is computed as raw GMV divided by all order attempts ($144.24), and Profit ignores refunds ($4,489.04).
- **With Hindsight**: Generalizes the net revenue definition to downstream metrics: Net Revenue / Delivered Orders ($136.08 AOV) and Net Revenue - COGS ($3,123.36 Profit).

### Scenario 3 — Conflict Detection (Official Policy vs Dave's Draft)
- **Question**: *"What was our total Revenue in February 2026 according to company policy?"*
- **Conflict**: Dave (Junior Analyst) wrote an unapproved draft note suggesting gross GMV without deductions. Official Finance Doc FP-2026-01 explicitly rejected Dave's draft.
- **Without Hindsight**: Silently uses raw gross booking volume ($7,162.94).
- **With Hindsight**: Detects the disagreement, inspects approvals and effective dates, rejects Dave's draft, and calculates audited GAAP Net Revenue ($4,337.16).

### Scenario 4 — Dynamic Correction & Subsequent Recall
- **Action**: In the **Memory Lab** tab, the user retains a new business rule (e.g. VIP Tier active window extension to 180 days).
- **Verification**: Asking a new question immediately recalls the newly retained rule and applies it to calculations.

---

## 🛠️ Getting Started

### 1. Prerequisites
- Python 3.11+
- Virtual environment (`.venv`)

### 2. Environment Configuration
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_gemini_api_key_here
HINDSIGHT_API_KEY=your_hindsight_api_key_here
HINDSIGHT_BASE_URL=https://api.hindsight.vectorize.io
HINDSIGHT_BANK_ID=metricchronicle
```

### 3. Run Automated Test Suite
Verify that all 9 components and scenarios pass:
```bash
.venv\Scripts\python.exe test_suite.py
```

### 4. Launch the Streamlit Dashboard
```bash
.venv\Scripts\streamlit.exe run app.py
```
Open your browser to: **`http://localhost:8501`**

---

## 📊 Evaluation Benchmark Results

Running the automated benchmark suite in **Tab 3 (Evaluation)** evaluates 6 standardized cases against the dataset:

| Test Case | Metric | Audited Ground Truth | With Hindsight | Without Hindsight |
| :--- | :--- | :--- | :--- | :--- |
| **Q1 Active Customers** | Active Customers | 39 Customers | ✅ **PASS** (39) | ❌ **FAIL** (32) |
| **Q2 Active Customers** | Active Customers | 33 Customers | ✅ **PASS** (33) | ❌ **FAIL** (38) |
| **February Net Revenue** | Revenue | $4,337.16 | ✅ **PASS** ($4,337.16) | ❌ **FAIL** ($7,162.94) |
| **May Average Order Value**| AOV | $136.08 | ✅ **PASS** ($136.08) | ❌ **FAIL** ($144.24) |
| **May Gross Profit** | Profit | $3,123.36 | ✅ **PASS** ($3,123.36) | ❌ **FAIL** ($4,489.04) |
| **Conflict Resolution** | Revenue Policy | Enforces FP-2026-01 | ✅ **PASS** (Rejects draft) | ❌ **FAIL** (Uses draft GMV) |

- **With Hindsight Accuracy**: **6 / 6 (100%)**
- **Without Hindsight Accuracy**: **0 / 6 (0%)**

---

## 📋 Hackathon Requirements Note
- **Hindsight Memory Bank**: Configured to `metricchronicle` via `https://api.hindsight.vectorize.io`.
- **Band AI Desktop**: As required by the hackathon organizers, Band AI Desktop is utilized as the participant environment and tool alongside the development workflow.
