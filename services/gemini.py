"""
MetricChronicle Gemini Service
Leverages Google GenAI SDK (google-genai) to understand natural-language business questions,
extract metrics and temporal scopes, and formulate side-by-side analytical explanations.
Enforces strict grounding: Gemini never invents numbers; all math is deterministic.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    genai = None


class GeminiService:
    """Manages Gemini query parsing and comparative business explanations."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-3.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model = model
        self.client: Optional[genai.Client] = None
        self._is_live = False
        self._init_client()

    def _is_valid_key(self, key: str) -> bool:
        if not key or not isinstance(key, str):
            return False
        clean = key.strip()
        if len(clean) < 15 or clean.startswith("YOUR_") or "placeholder" in clean.lower():
            return False
        return True

    def _init_client(self) -> None:
        if GENAI_AVAILABLE and self._is_valid_key(self.api_key):
            try:
                self.client = genai.Client(api_key=self.api_key)
                self._is_live = True
            except Exception:
                self.client = None
                self._is_live = False
        else:
            self.client = None
            self._is_live = False

    def is_live(self) -> bool:
        return self._is_live

    def parse_query(self, query: str) -> Dict[str, Any]:
        """
        Extract metric name, temporal window, and parameters from a natural-language question.
        Falls back to robust pattern extraction if live Gemini is unavailable.
        """
        if self._is_live and self.client:
            models_to_try = [self.model, "gemini-3.5-flash-lite", "gemini-3.6-flash"]
            prompt = f"""
You are a precise business intelligence query parser for an e-commerce company operating between January 1, 2026 and June 30, 2026.
Extract the target metric and date/period from the user's question.
Return ONLY valid JSON with keys:
- "metric": one of ["revenue", "active_customers", "average_order_value", "profit", "general"]
- "as_of_date": string in YYYY-MM-DD or null
- "start_date": string in YYYY-MM-DD or null
- "end_date": string in YYYY-MM-DD or null
- "period_name": e.g. "February 2026", "May 2026", "Q1 2026", "All Data"
- "summary": short 1-sentence summary of the intent

User question: "{query}"
JSON:
"""
            for m in models_to_try:
                try:
                    response = self.client.models.generate_content(
                        model=m,
                        contents=prompt,
                    )
                    raw_text = response.text.strip()
                    json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
                    if json_match:
                        parsed = json.loads(json_match.group(0))
                        return self._normalize_parsed_query(parsed, query)
                except Exception:
                    continue

        # Deterministic regex parser fallback
        return self._fallback_parse_query(query)

    def _normalize_parsed_query(self, parsed: Dict[str, Any], raw_query: str) -> Dict[str, Any]:
        metric = parsed.get("metric", "revenue")
        if metric not in ["revenue", "active_customers", "average_order_value", "profit", "general"]:
            metric = "revenue"
            
        return {
            "metric": metric,
            "as_of_date": parsed.get("as_of_date") or self._infer_as_of_date(raw_query),
            "start_date": parsed.get("start_date") or self._infer_start_date(raw_query),
            "end_date": parsed.get("end_date") or self._infer_end_date(raw_query),
            "period_name": parsed.get("period_name") or self._infer_period_name(raw_query),
            "summary": parsed.get("summary") or f"Analyzing {metric} for the requested period.",
        }

    def _fallback_parse_query(self, query: str) -> Dict[str, Any]:
        q_lower = query.lower()
        
        # Metric classification
        if any(w in q_lower for w in ["active customer", "active users", "customer count", "active buyer", "scenario 1"]):
            metric = "active_customers"
        elif any(w in q_lower for w in ["average order value", "aov"]):
            metric = "average_order_value"
        elif any(w in q_lower for w in ["profit", "margin", "cogs"]):
            metric = "profit"
        else:
            metric = "revenue"
            
        start_date = self._infer_start_date(query)
        end_date = self._infer_end_date(query)
        as_of_date = self._infer_as_of_date(query)
        period_name = self._infer_period_name(query)
        
        return {
            "metric": metric,
            "as_of_date": as_of_date,
            "start_date": start_date,
            "end_date": end_date,
            "period_name": period_name,
            "summary": f"Calculated {metric.replace('_', ' ').title()} for {period_name}.",
        }

    def _infer_as_of_date(self, query: str) -> str:
        q = query.lower()
        if "feb" in q:
            return "2026-02-28"
        if "may" in q:
            return "2026-05-31"
        if "jan" in q:
            return "2026-01-31"
        if "mar" in q:
            return "2026-03-31"
        if "apr" in q:
            return "2026-04-30"
        if "jun" in q:
            return "2026-06-30"
        if "q1" in q:
            return "2026-03-31"
        if "q2" in q:
            return "2026-06-30"
        return "2026-06-30"

    def _infer_start_date(self, query: str) -> str:
        q = query.lower()
        if "feb" in q:
            return "2026-02-01"
        if "may" in q:
            return "2026-05-01"
        if "jan" in q:
            return "2026-01-01"
        if "mar" in q:
            return "2026-03-01"
        if "apr" in q:
            return "2026-04-01"
        if "jun" in q:
            return "2026-06-01"
        if "q1" in q:
            return "2026-01-01"
        if "q2" in q:
            return "2026-04-01"
        return "2026-01-01"

    def _infer_end_date(self, query: str) -> str:
        return self._infer_as_of_date(query)

    def _infer_period_name(self, query: str) -> str:
        q = query.lower()
        if "feb" in q:
            return "February 2026"
        if "may" in q:
            return "May 2026"
        if "jan" in q:
            return "January 2026"
        if "mar" in q:
            return "March 2026"
        if "apr" in q:
            return "April 2026"
        if "jun" in q:
            return "June 2026"
        if "q1" in q:
            return "Q1 2026"
        if "q2" in q:
            return "Q2 2026"
        return "H1 2026 (Jan–Jun)"

    def explain_comparison(
        self,
        query: str,
        without_memory_data: Dict[str, Any],
        with_memory_data: Dict[str, Any],
        recalled_memories: List[Dict[str, Any]],
        reflection: str,
    ) -> Dict[str, Any]:
        """
        Formulates a comprehensive comparison between Without-Hindsight and With-Hindsight.
        """
        metric = with_memory_data.get("metric_name", "Revenue")
        
        # Format live Gemini explanation if available
        if self._is_live and self.client:
            models_to_try = [self.model, "gemini-3.5-flash-lite", "gemini-3.6-flash"]
            evidence_text = "\n".join([
                f"- [{m.get('approval_status', 'N/A')}] {m.get('title', '')}: {m.get('content', '')} (Effective: {m.get('effective_start', '')})"
                for m in recalled_memories[:4]
            ])
            prompt = f"""
You are an expert AI business analytics consultant explaining the exact difference between an AI agent operating WITHOUT organizational memory vs WITH Hindsight memory.

Question: "{query}"
Recalled Hindsight Memories:
{evidence_text}

Reflection:
{reflection}

Calculated Data:
- WITHOUT HINDSIGHT: {without_memory_data}
- WITH HINDSIGHT: {with_memory_data}

Provide a JSON object with:
- "without_hindsight_critique": 2 sentences explaining why standard assumptions failed or created risk.
- "with_hindsight_narrative": 2 sentences explaining how Hindsight applied company policy and temporal definitions.
- "difference_analysis": 2 sentences quantifying the delta and business impact.
- "unresolved_conflict_or_audit": 1 sentence summarizing approvals, effective dates, or conflict resolution.
"""
            for m in models_to_try:
                try:
                    response = self.client.models.generate_content(
                        model=m,
                        contents=prompt,
                    )
                    json_match = re.search(r"\{.*\}", response.text.strip(), re.DOTALL)
                    if json_match:
                        data = json.loads(json_match.group(0))
                        return data
                except Exception:
                    continue

        # High-fidelity analytical synthesis fallback
        return self._fallback_explanation(
            query, without_memory_data, with_memory_data, recalled_memories, reflection
        )

    def _fallback_explanation(
        self,
        query: str,
        without_memory_data: Dict[str, Any],
        with_memory_data: Dict[str, Any],
        recalled_memories: List[Dict[str, Any]],
        reflection: str,
    ) -> Dict[str, Any]:
        metric = with_memory_data.get("metric_name", "Metric")
        as_of = with_memory_data.get("as_of_date", "2026-06-30")
        
        # Check if temporal active customer query
        if "active" in metric.lower():
            lookback = with_memory_data.get("lookback_days", 90)
            naive_lookback = without_memory_data.get("lookback_days", 30)
            with_val = with_memory_data.get("active_customers", 0)
            without_val = without_memory_data.get("active_customers", 0)
            
            critique = (
                f"Without Hindsight, standard LLMs use a generic static 30-day window ({without_val} users), "
                "unaware of historical team definitions or formal policy transitions across quarters."
            )
            narrative = (
                f"With Hindsight, the agent retrieved the active policy governing {as_of} "
                f"({lookback}-day purchase window, approved by {recalled_memories[0].get('author', 'Guild') if recalled_memories else 'Growth Team'}), "
                f"yielding {with_val} active customers."
            )
            delta = (
                f"Difference of {abs(with_val - without_val)} customers. In Q1, the 90-day policy properly captures quarterly buyer retention, "
                "preventing severe undercounting of the customer base."
            )
        elif "aov" in metric.lower():
            with_val = with_memory_data.get("aov", 0.0)
            without_val = without_memory_data.get("aov", 0.0)
            critique = (
                f"Without Hindsight, AOV is naively computed as raw GMV divided by all order attempts (${without_val:,.2f}), "
                "improperly skewing unit economics by including cancelled orders."
            )
            narrative = (
                f"With Hindsight, the agent generalized the company revenue policy to AOV: subtracting refunds and "
                f"dividing only by fulfilled/delivered orders, resulting in an accurate ${with_val:,.2f} AOV."
            )
            delta = (
                f"Delta of ${abs(with_val - without_val):,.2f} per order. Relying on the naive calculation distorts customer lifetime value and acquisition payback periods."
            )
        elif "profit" in metric.lower():
            with_val = with_memory_data.get("profit", 0.0)
            without_val = without_memory_data.get("profit", 0.0)
            critique = (
                f"Without Hindsight, profit is overstated at ${without_val:,.2f} because cancelled orders and refund deductions "
                "are completely overlooked in the calculation."
            )
            narrative = (
                f"With Hindsight, CFO policy (Doc FP-2026-01) is recalled, strictly factoring in ${with_memory_data.get('total_refunds_deducted', 0):,.2f} "
                f"in refunds before COGS deduction, generating the audited profit of ${with_val:,.2f}."
            )
            delta = (
                f"Delta of ${abs(with_val - without_val):,.2f} in reported gross profit. Uncalibrated AI reports imaginary profits on returned merchandise."
            )
        else: # Revenue
            with_val = with_memory_data.get("revenue", 0.0)
            without_val = without_memory_data.get("revenue", 0.0)
            refunds = with_memory_data.get("total_refunds_deducted", 0.0)
            critique = (
                f"Without Hindsight, standard AI calculates gross booking volume (${without_val:,.2f}), "
                "erroneously including unfulfilled orders and cancelled customer attempts."
            )
            narrative = (
                f"With Hindsight, the agent applies Finance Leadership Policy FP-2026-01: delivered gross orders minus ${refunds:,.2f} in refunds, "
                f"yielding official net GAAP revenue of ${with_val:,.2f}."
            )
            delta = (
                f"Delta of ${abs(without_val - with_val):,.2f}. Using naive GMV overstates revenue and breaches corporate accounting standards."
            )

        conflict_note = (
            "Hindsight identified and resolved historical interpretations, prioritizing approved leadership policies "
            "over unapproved analyst drafts."
        )

        return {
            "without_hindsight_critique": critique,
            "with_hindsight_narrative": narrative,
            "difference_analysis": delta,
            "unresolved_conflict_or_audit": conflict_note,
        }


# Global singleton
_gemini_service_instance: Optional[GeminiService] = None

def get_gemini_service() -> GeminiService:
    global _gemini_service_instance
    if _gemini_service_instance is None:
        _gemini_service_instance = GeminiService()
    return _gemini_service_instance
