"""
MetricChronicle Analytics Service
Deterministic e-commerce dataset generation and calculation engine.
Calculates metrics using Pandas based on specified business policies.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sample_orders.csv")


def generate_sample_dataset(filepath: str = DATA_PATH, seed: int = 42) -> pd.DataFrame:
    """Generate a deterministic synthetic e-commerce dataset for Jan 1 - Jun 30, 2026."""
    np.random.seed(seed)
    
    start_date = datetime(2026, 1, 1)
    end_date = datetime(2026, 6, 30)
    total_days = (end_date - start_date).days + 1
    
    num_orders = 350
    customer_ids = [f"CUST-{i:03d}" for i in range(1, 51)]
    regions = ["North America", "Europe", "Asia Pacific", "Latin America"]
    region_weights = [0.45, 0.30, 0.15, 0.10]
    
    records = []
    
    for i in range(1, num_orders + 1):
        order_id = f"ORD-2026-{i:04d}"
        cust_id = np.random.choice(customer_ids)
        days_offset = np.random.randint(0, total_days)
        order_date = start_date + timedelta(days=days_offset)
        region = np.random.choice(regions, p=region_weights)
        
        # Order amount
        gross_amount = round(float(np.random.gamma(shape=3.0, scale=40.0) + 20.0), 2)
        # Cost of goods sold: ~40% - 55% of gross
        cogs_rate = np.random.uniform(0.40, 0.55)
        cogs = round(gross_amount * cogs_rate, 2)
        
        # Status distribution: 75% Delivered, 15% Cancelled, 10% Refunded
        status_choice = np.random.choice(["Delivered", "Cancelled", "Refunded"], p=[0.75, 0.15, 0.10])
        
        refund_amount = 0.0
        if status_choice == "Refunded":
            # Can be full or partial refund
            is_full = np.random.choice([True, False], p=[0.7, 0.3])
            refund_amount = gross_amount if is_full else round(gross_amount * np.random.uniform(0.3, 0.7), 2)
        elif status_choice == "Delivered":
            # Occasional post-delivery partial refund (~5% chance)
            if np.random.rand() < 0.05:
                refund_amount = round(gross_amount * np.random.uniform(0.2, 0.5), 2)
        
        records.append({
            "order_id": order_id,
            "customer_id": cust_id,
            "order_date": order_date.strftime("%Y-%m-%d"),
            "order_status": status_choice,
            "gross_amount": gross_amount,
            "refund_amount": refund_amount,
            "cost_of_goods_sold": cogs,
            "region": region,
        })
        
    df = pd.DataFrame(records)
    # Sort chronologically
    df["order_date_dt"] = pd.to_datetime(df["order_date"])
    df = df.sort_values("order_date_dt").reset_index(drop=True)
    df = df.drop(columns=["order_date_dt"])
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    return df


def load_dataset(filepath: str = DATA_PATH) -> pd.DataFrame:
    """Load dataset, generating it deterministically if not present."""
    if not os.path.exists(filepath):
        return generate_sample_dataset(filepath)
    df = pd.read_csv(filepath)
    return df


def filter_by_date(df: pd.DataFrame, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
    """Filter dataframe by date range."""
    filtered = df.copy()
    if start_date:
        filtered = filtered[filtered["order_date"] >= start_date]
    if end_date:
        filtered = filtered[filtered["order_date"] <= end_date]
    return filtered


def calculate_metrics(
    df: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    as_of_date: Optional[str] = None,
    use_hindsight_policy: bool = True,
    policy_override: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Calculate deterministic business metrics comparing With-Hindsight vs Without-Hindsight.
    
    Policies supported:
    - Company Official (With Hindsight):
        * Revenue: Delivered order gross minus refunds. Cancelled orders excluded.
        * Active Customers:
            - Before 2026-04-01: at least 1 delivered purchase in preceding 90 days.
            - From 2026-04-01 onwards: at least 1 delivered purchase in preceding 30 days.
        * AOV: Net Delivered Revenue / Count of Delivered orders.
        * Profit: Net Delivered Revenue minus COGS (for delivered orders).
    - Naive / Standard (Without Hindsight):
        * Revenue: Sum of all gross_amount (often ignores refunds or includes cancelled).
        * Active Customers: Generic fixed 30-day window or total unique buyers in period.
        * AOV: Total Gross Amount / Total orders count.
        * Profit: Total Gross Amount minus Total COGS.
    """
    # Normalize dates
    if not as_of_date and end_date:
        as_of_date = end_date
    elif not as_of_date:
        as_of_date = "2026-06-30"
        
    date_filtered = filter_by_date(df, start_date, end_date)
    as_of_dt = pd.to_datetime(as_of_date)
    
    if use_hindsight_policy and policy_override != "naive":
        # === WITH HINDSIGHT: COMPANY OFFICIAL POLICY ===
        
        # 1. Revenue
        delivered_orders = date_filtered[date_filtered["order_status"] == "Delivered"]
        gross_delivered = float(delivered_orders["gross_amount"].sum())
        total_refunds = float(date_filtered["refund_amount"].sum())
        net_revenue = round(gross_delivered - total_refunds, 2)
        
        # 2. Active Customers (Temporal Policy Rule)
        # Check cutoff: Jan-Mar = 90 days, Apr+ = 30 days
        is_q1 = as_of_dt < pd.to_datetime("2026-04-01")
        lookback_days = 90 if is_q1 else 30
        window_start_dt = as_of_dt - timedelta(days=lookback_days)
        window_start_str = window_start_dt.strftime("%Y-%m-%d")
        
        # Lookback uses entire history up to as_of_date
        history_window = df[
            (df["order_date"] >= window_start_str) &
            (df["order_date"] <= as_of_date) &
            (df["order_status"] == "Delivered")
        ]
        active_customer_count = int(history_window["customer_id"].nunique())
        
        # 3. Average Order Value (AOV)
        delivered_count = int(len(delivered_orders))
        aov = round(net_revenue / delivered_count, 2) if delivered_count > 0 else 0.0
        
        # 4. Profit
        delivered_cogs = float(delivered_orders["cost_of_goods_sold"].sum())
        profit = round(net_revenue - delivered_cogs, 2)
        
        return {
            "mode": "WITH_HINDSIGHT",
            "policy_name": "Official Approved Company Policy (Hindsight Governed)",
            "period": f"{start_date or 'Start'} to {end_date or as_of_date}",
            "as_of_date": as_of_date,
            "revenue": net_revenue,
            "revenue_definition": "Delivered order value minus refunds. Cancelled orders excluded.",
            "gross_delivered": round(gross_delivered, 2),
            "total_refunds_deducted": round(total_refunds, 2),
            "cancelled_orders_excluded_count": int((date_filtered["order_status"] == "Cancelled").sum()),
            "active_customers": active_customer_count,
            "active_customers_definition": f"{'90-day purchase window (Q1 Jan-Mar Policy)' if is_q1 else '30-day purchase window (Q2+ April Policy Revision)'} for delivered orders as of {as_of_date}.",
            "lookback_days": lookback_days,
            "aov": aov,
            "aov_definition": "Net Delivered Revenue divided by delivered order count ($" + f"{net_revenue:,.2f} / {delivered_count})",
            "delivered_orders_count": delivered_count,
            "profit": profit,
            "profit_definition": "Net Delivered Revenue minus Cost of Goods Sold for delivered orders ($" + f"{net_revenue:,.2f} - ${delivered_cogs:,.2f})",
            "cogs": round(delivered_cogs, 2),
            "total_orders_in_period": int(len(date_filtered)),
        }
    else:
        # === WITHOUT HINDSIGHT: NAIVE / GENERIC BASELINE ===
        all_gross = float(date_filtered["gross_amount"].sum())
        all_orders_count = int(len(date_filtered))
        all_cogs = float(date_filtered["cost_of_goods_sold"].sum())
        
        # Naive active customers: assumes standard generic 30-day window or period buyers
        lookback_days = 30
        window_start_dt = as_of_dt - timedelta(days=lookback_days)
        window_start_str = window_start_dt.strftime("%Y-%m-%d")
        naive_window = df[
            (df["order_date"] >= window_start_str) &
            (df["order_date"] <= as_of_date)
        ]
        naive_active_customers = int(naive_window["customer_id"].nunique())
        
        naive_revenue = round(all_gross, 2)
        naive_aov = round(all_gross / all_orders_count, 2) if all_orders_count > 0 else 0.0
        naive_profit = round(all_gross - all_cogs, 2)
        
        return {
            "mode": "WITHOUT_HINDSIGHT",
            "policy_name": "Generic Standard Assumption (No Organizational Memory)",
            "period": f"{start_date or 'Start'} to {end_date or as_of_date}",
            "as_of_date": as_of_date,
            "revenue": naive_revenue,
            "revenue_definition": "Total gross amount across all orders (refunds and cancellations not tracked or deducted).",
            "gross_delivered": round(float(date_filtered[date_filtered['order_status'] == 'Delivered']['gross_amount'].sum()), 2),
            "total_refunds_deducted": 0.0,
            "cancelled_orders_excluded_count": 0,
            "active_customers": naive_active_customers,
            "active_customers_definition": f"Standard uncalibrated 30-day window across all order statuses as of {as_of_date}.",
            "lookback_days": lookback_days,
            "aov": naive_aov,
            "aov_definition": "Total Gross Amount divided by total raw orders ($" + f"{all_gross:,.2f} / {all_orders_count})",
            "delivered_orders_count": int((date_filtered["order_status"] == "Delivered").sum()),
            "profit": naive_profit,
            "profit_definition": "Total Gross Amount minus Total COGS ($" + f"{all_gross:,.2f} - ${all_cogs:,.2f})",
            "cogs": round(all_cogs, 2),
            "total_orders_in_period": all_orders_count,
        }


def get_dataset_overview(df: pd.DataFrame) -> Dict[str, Any]:
    """Get high-level dataset statistics for the dashboard."""
    return {
        "total_records": len(df),
        "date_min": str(df["order_date"].min()),
        "date_max": str(df["order_date"].max()),
        "unique_customers": int(df["customer_id"].nunique()),
        "total_gross_gmv": round(float(df["gross_amount"].sum()), 2),
        "total_refunds": round(float(df["refund_amount"].sum()), 2),
        "delivered_count": int((df["order_status"] == "Delivered").sum()),
        "cancelled_count": int((df["order_status"] == "Cancelled").sum()),
        "refunded_count": int((df["order_status"] == "Refunded").sum()),
        "regions": sorted(df["region"].unique().tolist()),
    }
