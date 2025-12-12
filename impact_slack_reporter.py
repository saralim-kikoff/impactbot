"""
Impact.com Weekly Performance Reporter → Slack

Metrics tracked:
- Actions (Payment Success)
- Total Cost
- CPC
- Clicks
- Conversion Rate (Payment Success)
- Reversal Rate
- CAC (Total Cost / Payment Success Actions)

Setup:
1. Set environment variables:
   - IMPACT_ACCOUNT_SID
   - IMPACT_AUTH_TOKEN
   - SLACK_WEBHOOK_URL

2. Schedule with cron (every Monday at 9am):
   0 9 * * 1 python3 /path/to/impact_slack_reporter.py
"""

import os
import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import json

# =============================================================================
# CONFIGURATION
# =============================================================================

ACCOUNT_SID = os.getenv("IMPACT_ACCOUNT_SID")
AUTH_TOKEN = os.getenv("IMPACT_AUTH_TOKEN")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
CAMPAIGN_ID = os.getenv("IMPACT_CAMPAIGN_ID", "14994")
BASE_URL = f"https://api.impact.com/Advertisers/{ACCOUNT_SID}"

# Validate required environment variables
def validate_config():
    missing = []
    if not ACCOUNT_SID:
        missing.append("IMPACT_ACCOUNT_SID")
    if not AUTH_TOKEN:
        missing.append("IMPACT_AUTH_TOKEN")
    if not SLACK_WEBHOOK_URL:
        missing.append("SLACK_WEBHOOK_URL")
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

# Payment Success event type configuration
PAYMENT_SUCCESS_EVENT_TYPE_ID = "28113"
PAYMENT_SUCCESS_EVENT_TYPE_NAME = "Payment success"


# =============================================================================
# IMPACT.COM API FUNCTIONS
# =============================================================================

def get_auth() -> HTTPBasicAuth:
    return HTTPBasicAuth(ACCOUNT_SID, AUTH_TOKEN)


def get_week_range(weeks_back: int = 0) -> tuple[str, str]:
    """
    Get Monday-Sunday date range for a given week.
    weeks_back=0 means last complete Mon-Sun week.
    """
    today = datetime.now()
    # Find most recent Sunday
    days_since_sunday = (today.weekday() + 1) % 7
    if days_since_sunday == 0:
        days_since_sunday = 7  # If today is Sunday, go back to last Sunday
    last_sunday = today - timedelta(days=days_since_sunday)
    
    # Adjust for weeks_back
    end_date = last_sunday - timedelta(weeks=weeks_back)
    start_date = end_date - timedelta(days=6)  # Monday
    
    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")


def get_month_range(months_back: int = 1) -> tuple[str, str]:
    """
    Get first-last day date range for a given month.
    months_back=1 means last complete month (default for monthly report).
    months_back=2 means the month before last.
    """
    from calendar import monthrange
    
    today = datetime.now()
    
    # Calculate target month
    year = today.year
    month = today.month - months_back
    
    # Handle year rollover
    while month <= 0:
        month += 12
        year -= 1
    
    # Get first and last day of target month
    first_day = datetime(year, month, 1)
    last_day_num = monthrange(year, month)[1]
    last_day = datetime(year, month, last_day_num)
    
    return first_day.strftime("%Y-%m-%d"), last_day.strftime("%Y-%m-%d")


def fetch_actions(start_date: str, end_date: str) -> list[dict]:
    """Fetch all conversion actions within a date range."""
    actions = []
    page = 1
    page_size = 2000  # API minimum is 2000, max is 20000
    
    while True:
        params = {
            "CampaignId": CAMPAIGN_ID,
            "ActionDateStart": f"{start_date}T00:00:00Z",
            "ActionDateEnd": f"{end_date}T23:59:59Z",
            "PageSize": page_size,
            "Page": page
        }
        
        response = requests.get(
            f"{BASE_URL}/Actions",
            auth=get_auth(),
            params=params,
            headers={"Accept": "application/json"}
        )
        
        # Better error handling - print response for debugging
        if response.status_code != 200:
            print(f"❌ API Error {response.status_code}: {response.text}")
            response.raise_for_status()
        data = response.json()
        
        batch = data.get("Actions", [])
        if not batch:
            break
            
        actions.extend(batch)
        if len(batch) < page_size:
            break
        page += 1
    
    return actions


def fetch_media_partner_stats(start_date: str, end_date: str) -> Dict[str, Dict]:
    """
    Fetch aggregated stats including clicks by partner via ReportExport.
    Uses Performance by Partner report for partner-level breakdown.
    """
    import time
    
    total_clicks = 0
    partner_clicks = {}
    
    # Use Performance by Partner report for partner-level data
    report_id = "att_adv_performance_by_media_pm_only"
    
    params = {
        "START_DATE": start_date,
        "END_DATE": end_date,
        "SUBAID": CAMPAIGN_ID,
    }
    
    try:
        print(f"   🔍 Fetching clicks by partner via ReportExport...")
        response = requests.get(
            f"{BASE_URL}/ReportExport/{report_id}",
            auth=get_auth(),
            params=params,
            headers={"Accept": "application/json"}
        )
        
        if response.status_code != 200:
            print(f"   ⚠️  ReportExport failed: {response.status_code}")
            return {}
        
        data = response.json()
        queued_uri = data.get("QueuedUri")
        
        if not queued_uri:
            print(f"   ⚠️  No QueuedUri returned")
            return {}
        
        # Poll for job completion
        for attempt in range(15):
            status_response = requests.get(
                f"https://api.impact.com{queued_uri}",
                auth=get_auth(),
                headers={"Accept": "application/json"}
            )
            
            if status_response.status_code != 200:
                time.sleep(2)
                continue
            
            job_data = status_response.json()
            job_status = job_data.get("Status", "").upper()
            
            if job_status == "COMPLETED":
                # Download the results
                result_uri = job_data.get("ResultUri")
                if result_uri:
                    dl_response = requests.get(
                        f"https://api.impact.com{result_uri}",
                        auth=get_auth(),
                        headers={"Accept": "application/json"}
                    )
                    
                    if dl_response.status_code == 200:
                        # Parse CSV
                        import csv
                        import io
                        content_type = dl_response.headers.get("Content-Type", "")
                        
                        if "json" in content_type:
                            dl_data = dl_response.json()
                            records = dl_data.get("Records", [])
                        else:
                            reader = csv.DictReader(io.StringIO(dl_response.text))
                            records = list(reader)
                        
                        if records:
                            # Show first record to see field names
                            print(f"   📋 Fields: {list(records[0].keys())}")
                            print(f"   📋 Sample: {records[0]}")
                            
                            for record in records:
                                # Get partner name - try different field names
                                partner = (
                                    record.get("Media") or 
                                    record.get("Partner") or 
                                    record.get("Media_Name") or 
                                    record.get("partner_name") or
                                    "Unknown"
                                )
                                
                                clicks = record.get("Clicks") or record.get("clicks") or 0
                                if clicks and str(clicks).strip():
                                    click_count = int(float(clicks))
                                    total_clicks += click_count
                                    partner_clicks[partner] = {"clicks": click_count, "cost": 0}
                            
                            print(f"   ✅ Total clicks: {total_clicks:,} across {len(partner_clicks)} partners")
                            
                            if total_clicks > 0:
                                return partner_clicks
                break
                
            elif job_status in ["FAILED", "CANCELLED", "ERROR"]:
                print(f"   ❌ Job failed: {job_data.get('StatusMessage')}")
                break
            
            time.sleep(2)
                
    except Exception as e:
        print(f"   ⚠️  Error: {e}")
    
    return {}


# =============================================================================
# DATA PROCESSING
# =============================================================================

def get_top_partners(metrics: Dict, n: int = 10) -> list[str]:
    """Get top N partners by payment success actions."""
    partner_metrics = metrics.get("partner_metrics", {})
    sorted_partners = sorted(
        partner_metrics.items(),
        key=lambda x: x[1].get("payment_success", 0),
        reverse=True
    )
    return [p[0] for p in sorted_partners[:n] if p[1].get("payment_success", 0) > 0]


def identify_new_top_partners(current_top: list[str], historical_tops: list[list[str]]) -> list[str]:
    """
    Identify partners in current top 10 that weren't in top 10 for any of the past weeks.
    """
    # Combine all historical top partners
    historical_set = set()
    for week_top in historical_tops:
        historical_set.update(week_top)
    
    # Find new partners
    new_partners = [p for p in current_top if p not in historical_set]
    return new_partners


def process_metrics(actions: list[dict], partner_stats: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Process raw data into the key metrics we care about.
    """
    # Initialize counters
    payment_success_actions = 0
    reversed_actions = 0
    total_actions = 0
    
    # Partner-level tracking
    partner_metrics = {}
    
    for action in actions:
        partner = action.get("MediaPartnerName", "Unknown")
        status = (action.get("State", "") or "").lower()
        payout = float(action.get("Payout", 0) or 0)
        
        # Initialize partner if needed
        if partner not in partner_metrics:
            partner_metrics[partner] = {
                "payment_success": 0,
                "reversed": 0,
                "total_actions": 0,
                "cost": 0.0
            }
        
        total_actions += 1
        partner_metrics[partner]["total_actions"] += 1
        partner_metrics[partner]["cost"] += payout
        
        # Check if this is a payment success action using Event Type ID
        event_type_id = str(action.get("ActionTrackerId", "") or action.get("EventTypeId", "") or "")
        event_type_name = (action.get("ActionTrackerName", "") or action.get("EventTypeName", "") or "").lower()
        
        is_payment_success = (
            event_type_id == PAYMENT_SUCCESS_EVENT_TYPE_ID or
            PAYMENT_SUCCESS_EVENT_TYPE_NAME.lower() in event_type_name
        )
        
        if is_payment_success:
            payment_success_actions += 1
            partner_metrics[partner]["payment_success"] += 1
            
            # Check for reversals - only count reversed Payment Success actions
            if status in ["reversed", "rejected"]:
                reversed_actions += 1
                partner_metrics[partner]["reversed"] += 1
    
    # Calculate total clicks and cost from partner stats (if available)
    # Exclude _total key to avoid double counting
    total_clicks = sum(p.get("clicks", 0) for k, p in partner_stats.items() if k != "_total") if partner_stats else 0
    total_cost = sum(p.get("cost", 0) for k, p in partner_stats.items() if k != "_total") if partner_stats else 0
    
    # If cost from reports is 0, fall back to payout sum from actions
    if total_cost == 0:
        total_cost = sum(p["cost"] for p in partner_metrics.values())
    
    # Merge click data into partner metrics (if available)
    for partner, stats in partner_stats.items():
        if partner not in partner_metrics:
            partner_metrics[partner] = {
                "payment_success": 0,
                "reversed": 0,
                "total_actions": 0,
                "cost": stats.get("cost", 0)
            }
        partner_metrics[partner]["clicks"] = stats.get("clicks", 0)
    
    # Calculate derived metrics
    # Note: CPC and Conversion Rate require click data from Reports API
    cpc = total_cost / total_clicks if total_clicks > 0 else None
    conversion_rate = (payment_success_actions / total_clicks * 100) if total_clicks > 0 else None
    # Reversal rate = reversed Payment Success actions / total Payment Success actions
    reversal_rate = (reversed_actions / payment_success_actions * 100) if payment_success_actions > 0 else 0
    cac = total_cost / payment_success_actions if payment_success_actions > 0 else None
    
    return {
        "payment_success_actions": payment_success_actions,
        "total_cost": total_cost,
        "clicks": total_clicks,
        "cpc": cpc,
        "conversion_rate": conversion_rate,
        "reversal_rate": reversal_rate,
        "cac": cac,
        "reversed_actions": reversed_actions,
        "total_actions": total_actions,
        "partner_metrics": partner_metrics
    }


def calculate_changes(current: Dict, previous: Dict) -> Dict[str, Any]:
    """Calculate WoW changes for each metric."""
    
    def calc_change(curr: float, prev: float) -> Dict:
        # Handle None values
        if curr is None or prev is None:
            return {"current": curr, "previous": prev, "change_pct": None, "change_abs": None}
        if prev == 0:
            pct = None if curr == 0 else float('inf')
            return {"current": curr, "previous": prev, "change_pct": pct, "change_abs": curr - prev}
        pct = ((curr - prev) / prev) * 100
        return {"current": curr, "previous": prev, "change_pct": pct, "change_abs": curr - prev}
    
    return {
        "payment_success_actions": calc_change(
            current["payment_success_actions"], 
            previous["payment_success_actions"]
        ),
        "total_cost": calc_change(current["total_cost"], previous["total_cost"]),
        "clicks": calc_change(current["clicks"], previous["clicks"]),
        "cpc": calc_change(current["cpc"], previous["cpc"]),
        "conversion_rate": calc_change(current["conversion_rate"], previous["conversion_rate"]),
        "reversal_rate": calc_change(current["reversal_rate"], previous["reversal_rate"]),
        "cac": calc_change(current["cac"], previous["cac"]),
    }


def identify_partner_drivers(
    current_metrics: Dict, 
    previous_metrics: Dict
) -> Dict[str, list]:
    """
    Identify which partners drove the biggest changes in key metrics.
    Returns top movers (positive and negative) for each metric.
    
    CAC Movers: Partners whose cost/action ratio change contributed most to overall CAC change
    CVR Movers: Partners whose click/conversion changes contributed most to overall CVR change
    """
    current_partners = current_metrics["partner_metrics"]
    previous_partners = previous_metrics["partner_metrics"]
    
    # Get all partners
    all_partners = set(current_partners.keys()) | set(previous_partners.keys())
    
    # Calculate overall metrics for context
    total_current_actions = current_metrics.get("payment_success_actions", 0)
    total_prev_actions = previous_metrics.get("payment_success_actions", 0)
    total_current_clicks = current_metrics.get("clicks", 0)
    total_prev_clicks = previous_metrics.get("clicks", 0)
    total_current_cost = current_metrics.get("total_cost", 0)
    total_prev_cost = previous_metrics.get("total_cost", 0)
    
    # Calculate changes per partner
    partner_changes = []
    for partner in all_partners:
        curr = current_partners.get(partner, {"payment_success": 0, "cost": 0, "clicks": 0})
        prev = previous_partners.get(partner, {"payment_success": 0, "cost": 0, "clicks": 0})
        
        curr_actions = curr.get("payment_success", 0)
        prev_actions = prev.get("payment_success", 0)
        curr_cost = curr.get("cost", 0)
        prev_cost = prev.get("cost", 0)
        curr_clicks = curr.get("clicks", 0)
        prev_clicks = prev.get("clicks", 0)
        
        actions_change = curr_actions - prev_actions
        cost_change = curr_cost - prev_cost
        clicks_change = curr_clicks - prev_clicks
        
        # Calculate CAC contribution
        # A partner contributes to CAC increase if their cost increased more than actions
        # or if their cost stayed same but actions decreased
        curr_cac = curr_cost / curr_actions if curr_actions > 0 else 0
        prev_cac = prev_cost / prev_actions if prev_actions > 0 else 0
        cac_change = curr_cac - prev_cac
        
        # Weight CAC change by partner's share of total cost (impact on overall CAC)
        cost_share = curr_cost / total_current_cost if total_current_cost > 0 else 0
        cac_impact = cac_change * cost_share
        
        # Calculate CVR contribution
        # CVR = actions / clicks
        # A partner impacts CVR if their clicks/actions ratio changed significantly
        curr_cvr = (curr_actions / curr_clicks * 100) if curr_clicks > 0 else 0
        prev_cvr = (prev_actions / prev_clicks * 100) if prev_clicks > 0 else 0
        cvr_change = curr_cvr - prev_cvr
        
        # Weight CVR change by partner's share of total clicks (impact on overall CVR)
        clicks_share = curr_clicks / total_current_clicks if total_current_clicks > 0 else 0
        cvr_impact = cvr_change * clicks_share
        
        partner_changes.append({
            "partner": partner,
            "actions_change": actions_change,
            "actions_current": curr_actions,
            "cost_change": cost_change,
            "cost_current": curr_cost,
            "clicks_change": clicks_change,
            "clicks_current": curr_clicks,
            "cac_change": cac_change,
            "cac_impact": cac_impact,
            "curr_cac": curr_cac,
            "prev_cac": prev_cac,
            "cvr_change": cvr_change,
            "cvr_impact": cvr_impact,
            "curr_cvr": curr_cvr,
            "prev_cvr": prev_cvr,
        })
    
    # Sort by absolute change to find biggest movers
    actions_movers = sorted(partner_changes, key=lambda x: abs(x["actions_change"]), reverse=True)[:3]
    
    # CAC movers - sorted by impact on overall CAC (weighted by cost share)
    cac_movers = sorted(partner_changes, key=lambda x: abs(x["cac_impact"]), reverse=True)[:3]
    
    # CVR movers - sorted by impact on overall CVR (weighted by clicks share)  
    cvr_movers = sorted(partner_changes, key=lambda x: abs(x["cvr_impact"]), reverse=True)[:3]
    
    # Filter out zero/negligible changes
    actions_movers = [p for p in actions_movers if p["actions_change"] != 0]
    cac_movers = [p for p in cac_movers if abs(p["cac_impact"]) > 0.01]  # > 1 cent impact
    cvr_movers = [p for p in cvr_movers if abs(p["cvr_impact"]) > 0.01]  # > 0.01% impact
    
    return {
        "actions": actions_movers,
        "cac": cac_movers,
        "cvr": cvr_movers
    }


# =============================================================================
# SLACK FORMATTING
# =============================================================================

def format_currency(amount: float) -> str:
    if amount is None:
        return "N/A"
    return f"${amount:,.2f}"


def format_number(num: float, decimals: int = 0) -> str:
    if num is None:
        return "N/A"
    if decimals == 0:
        return f"{num:,.0f}"
    return f"{num:,.{decimals}f}"


def format_pct(value: float) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2f}%"


def format_trend(change_data: Dict, is_inverse: bool = False) -> str:
    """
    Format a trend with emoji. 
    is_inverse=True means lower is better (e.g., CAC, CPC, Reversal Rate)
    """
    pct = change_data.get("change_pct")
    if pct is None:
        return "—"
    if pct == float('inf'):
        return "🆕"
    
    # Determine if change is good or bad
    is_positive_change = pct > 0
    is_good = is_positive_change if not is_inverse else not is_positive_change
    
    # Use colored circle emojis
    if abs(pct) < 1:
        emoji = ""
    elif is_good:
        emoji = ":large_green_circle:"
    else:
        emoji = ":red_circle:"
    
    sign = "+" if pct > 0 else ""
    return f"{emoji} *{sign}{pct:.1f}%*"


def format_partner_movers(movers: list, metric_type: str) -> str:
    """Format partner movers into readable text."""
    if not movers:
        return "_No significant changes_"
    
    lines = []
    for p in movers:
        if metric_type == "actions":
            change = p["actions_change"]
            sign = "+" if change > 0 else ""
            lines.append(f"• *{p['partner']}*: {sign}{change:,.0f} actions")
        elif metric_type == "cac":
            # Show CAC change with context
            cac_change = p["cac_change"]
            curr_cac = p["curr_cac"]
            prev_cac = p["prev_cac"]
            if cac_change > 0:
                lines.append(f"• *{p['partner']}*: CAC ↑ ${prev_cac:.2f} → ${curr_cac:.2f}")
            else:
                lines.append(f"• *{p['partner']}*: CAC ↓ ${prev_cac:.2f} → ${curr_cac:.2f}")
        elif metric_type == "cvr":
            # Show CVR change with context
            cvr_change = p["cvr_change"]
            curr_cvr = p["curr_cvr"]
            prev_cvr = p["prev_cvr"]
            clicks_change = p["clicks_change"]
            actions_change = p["actions_change"]
            
            if cvr_change > 0:
                arrow = "↑"
            else:
                arrow = "↓"
            
            # Add context about what drove the change
            context = ""
            if abs(clicks_change) > abs(actions_change) * 10:
                # Clicks changed much more than actions
                sign = "+" if clicks_change > 0 else ""
                context = f" ({sign}{clicks_change:,.0f} clicks)"
            elif abs(actions_change) > 0:
                sign = "+" if actions_change > 0 else ""
                context = f" ({sign}{actions_change:,.0f} actions)"
            
            lines.append(f"• *{p['partner']}*: CVR {arrow} {prev_cvr:.2f}% → {curr_cvr:.2f}%{context}")
        elif metric_type == "clicks":
            change = p["clicks_change"]
            sign = "+" if change > 0 else ""
            lines.append(f"• *{p['partner']}*: {sign}{change:,.0f} clicks")
    
    return "\n".join(lines)


def build_slack_message(
    current: Dict,
    changes: Dict,
    partner_drivers: Dict,
    date_range: tuple[str, str],
    prev_date_range: tuple[str, str],
    new_top_partners: list[str] = None,
    report_type: str = "weekly"
) -> Dict[str, Any]:
    """Build Slack Block Kit message."""
    
    start_date, end_date = date_range
    prev_start, prev_end = prev_date_range
    partner_metrics = current.get("partner_metrics", {})
    
    # Set title and comparison label based on report type
    if report_type == "monthly":
        title = "📊 Monthly Impact Affiliate Performance Report"
        comparison_label = "MoM"
    else:
        title = "📊 Weekly Impact Affiliate Performance Report"
        comparison_label = "WoW"
    
    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": title,
                "emoji": True
            }
        },
        {
            "type": "context",
            "elements": [
                {"type": "mrkdwn", "text": f"*{start_date}* to *{end_date}*\n(vs *{prev_start}* to *{prev_end}*)"}
            ]
        },
        {"type": "divider"},
        
        # Key Metrics Section
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": "*🎯 Key Metrics*"}
        },
        # Row 1: Actions & Clicks
        {
            "type": "section",
            "fields": [
                {
                    "type": "mrkdwn",
                    "text": f"*Actions (Payment Success)*\n*{format_number(current['payment_success_actions'])}*\nvs {format_number(changes['payment_success_actions']['previous'])} prev\n{format_trend(changes['payment_success_actions'])} {comparison_label}"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Clicks*\n*{format_number(current['clicks'])}*\nvs {format_number(changes['clicks']['previous'])} prev\n{format_trend(changes['clicks'])} {comparison_label}"
                }
            ]
        },
        # Row 2: Conversion Rate & Total Cost
        {
            "type": "section",
            "fields": [
                {
                    "type": "mrkdwn",
                    "text": f"*Conversion Rate*\n*{format_pct(current['conversion_rate'])}*\nvs {format_pct(changes['conversion_rate']['previous'])} prev\n{format_trend(changes['conversion_rate'])} {comparison_label}"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Total Cost*\n*{format_currency(current['total_cost'])}*\nvs {format_currency(changes['total_cost']['previous'])} prev\n{format_trend(changes['total_cost'], is_inverse=True)} {comparison_label}"
                }
            ]
        },
        # Row 3: CAC & Reversal Rate
        {
            "type": "section",
            "fields": [
                {
                    "type": "mrkdwn",
                    "text": f"*CAC*\n*{format_currency(current['cac'])}*\nvs {format_currency(changes['cac']['previous'])} prev\n{format_trend(changes['cac'], is_inverse=True)} {comparison_label}"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Reversal Rate*\n*{format_pct(current['reversal_rate'])}*\nvs {format_pct(changes['reversal_rate']['previous'])} prev\n{format_trend(changes['reversal_rate'], is_inverse=True)} {comparison_label}"
                }
            ]
        },
        {"type": "divider"},
        
        # Partner Attribution Section
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": "*🔍 What Drove the Changes?*"}
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Actions (Payment Success) Movers:*\n{format_partner_movers(partner_drivers['actions'], 'actions')}"
            }
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*CAC Movers:*\n{format_partner_movers(partner_drivers['cac'], 'cac')}"
            }
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*CVR Movers:*\n{format_partner_movers(partner_drivers['cvr'], 'cvr')}"
            }
        },
    ]
    
    # Add new top partners callout if any
    if new_top_partners:
        partner_list = ", ".join([
            f"*{p}* ({partner_metrics.get(p, {}).get('payment_success', 0):,} actions)" 
            for p in new_top_partners
        ])
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"🆕 *New Top 10 Partner(s):* {partner_list}\n_First time in top 10 for Payment Success in past 4 weeks_"
            }
        })
    
    blocks.extend([
        {"type": "divider"},
        {
            "type": "context",
            "elements": [
                {"type": "mrkdwn", "text": ":large_green_circle: = favorable change | :red_circle: = unfavorable change | Data from Impact.com"}
            ]
        }
    ])
    
    return {"blocks": blocks}


def send_to_slack(message: Dict[str, Any]) -> bool:
    """Send message to Slack via webhook."""
    if not SLACK_WEBHOOK_URL:
        print("⚠️  Slack webhook not configured. Message preview:")
        print(json.dumps(message, indent=2))
        return False
    
    response = requests.post(
        SLACK_WEBHOOK_URL,
        json=message,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        print("✅ Message sent to Slack successfully!")
        return True
    else:
        print(f"❌ Failed to send to Slack: {response.status_code} - {response.text}")
        return False


# =============================================================================
# MAIN
# =============================================================================

def run_weekly_report():
    """Generate and send the weekly report."""
    print("🚀 Starting Impact.com Weekly Report...")
    
    # Validate configuration
    validate_config()
    
    # Get date ranges (Mon-Sun)
    current_start, current_end = get_week_range(weeks_back=0)
    previous_start, previous_end = get_week_range(weeks_back=1)
    
    print(f"📅 Current week: {current_start} to {current_end}")
    print(f"📅 Previous week: {previous_start} to {previous_end}")
    
    # Fetch current week data
    print("📥 Fetching current week data...")
    current_actions = fetch_actions(current_start, current_end)
    current_partner_stats = fetch_media_partner_stats(current_start, current_end)
    print(f"   Found {len(current_actions)} actions")
    
    # Fetch previous week data
    print("📥 Fetching previous week data...")
    previous_actions = fetch_actions(previous_start, previous_end)
    previous_partner_stats = fetch_media_partner_stats(previous_start, previous_end)
    print(f"   Found {len(previous_actions)} actions")
    
    # Fetch historical data (weeks 2-4 back) for new partner detection
    print("📥 Fetching historical data for new partner detection...")
    historical_tops = []
    for weeks_back in range(1, 5):  # weeks 1-4 back
        hist_start, hist_end = get_week_range(weeks_back=weeks_back)
        hist_actions = fetch_actions(hist_start, hist_end)
        hist_metrics = process_metrics(hist_actions, {})
        top_partners = get_top_partners(hist_metrics, n=10)
        historical_tops.append(top_partners)
        print(f"      Week {weeks_back} back ({hist_start}): {top_partners}")
    print(f"   Analyzed {len(historical_tops)} historical weeks")
    
    # Process metrics
    print("🔄 Processing metrics...")
    current_metrics = process_metrics(current_actions, current_partner_stats)
    previous_metrics = process_metrics(previous_actions, previous_partner_stats)
    
    # Identify new top 10 partners
    current_top_10 = get_top_partners(current_metrics, n=10)
    print(f"   Current week top 10: {current_top_10}")
    new_top_partners = identify_new_top_partners(current_top_10, historical_tops)
    if new_top_partners:
        print(f"   🆕 New top 10 partners: {', '.join(new_top_partners)}")
    else:
        print(f"   ℹ️  No new top 10 partners this week")
    
    # Calculate changes
    changes = calculate_changes(current_metrics, previous_metrics)
    partner_drivers = identify_partner_drivers(current_metrics, previous_metrics)
    
    # Print summary to console
    def fmt_pct(val):
        return f"{val:.1f}" if val is not None else "N/A"
    
    def fmt_currency(val):
        return f"${val:,.2f}" if val is not None else "N/A"
    
    def fmt_num(val):
        return f"{val:,}" if val is not None else "N/A"
    
    print("\n📊 Summary:")
    print(f"   Actions (Payment Success): {fmt_num(current_metrics['payment_success_actions'])} ({fmt_pct(changes['payment_success_actions']['change_pct'])}% WoW)")
    print(f"   Total Cost: {fmt_currency(current_metrics['total_cost'])} ({fmt_pct(changes['total_cost']['change_pct'])}% WoW)")
    print(f"   CAC: {fmt_currency(current_metrics['cac'])} ({fmt_pct(changes['cac']['change_pct'])}% WoW)")
    print(f"   Reversal Rate: {fmt_pct(current_metrics['reversal_rate'])}%")
    
    # Build and send Slack message
    print("\n📝 Building Slack message...")
    message = build_slack_message(
        current_metrics,
        changes,
        partner_drivers,
        (current_start, current_end),
        (previous_start, previous_end),
        new_top_partners
    )
    
    send_to_slack(message)
    
    return {
        "period": f"{current_start} to {current_end}",
        "metrics": current_metrics,
        "changes": changes,
        "partner_drivers": partner_drivers
    }


def run_monthly_report():
    """Generate and send the monthly report for the previous completed month."""
    print("🚀 Starting Impact.com Monthly Report...")
    
    # Validate configuration
    validate_config()
    
    # Get date ranges (full months)
    current_start, current_end = get_month_range(months_back=1)  # Last month
    previous_start, previous_end = get_month_range(months_back=2)  # Month before last
    
    print(f"📅 Current month: {current_start} to {current_end}")
    print(f"📅 Previous month: {previous_start} to {previous_end}")
    
    # Fetch current month data
    print("📥 Fetching current month data...")
    current_actions = fetch_actions(current_start, current_end)
    current_partner_stats = fetch_media_partner_stats(current_start, current_end)
    print(f"   Found {len(current_actions)} actions")
    
    # Fetch previous month data
    print("📥 Fetching previous month data...")
    previous_actions = fetch_actions(previous_start, previous_end)
    previous_partner_stats = fetch_media_partner_stats(previous_start, previous_end)
    print(f"   Found {len(previous_actions)} actions")
    
    # Fetch historical data (months 2-4 back) for new partner detection
    print("📥 Fetching historical data for new partner detection...")
    historical_tops = []
    for months_back in range(1, 4):  # months 1-3 back
        hist_start, hist_end = get_month_range(months_back=months_back)
        hist_actions = fetch_actions(hist_start, hist_end)
        hist_metrics = process_metrics(hist_actions, {})
        top_partners = get_top_partners(hist_metrics, n=10)
        historical_tops.append(top_partners)
        print(f"      Month {months_back} back ({hist_start[:7]}): {top_partners[:5]}...")
    print(f"   Analyzed {len(historical_tops)} historical months")
    
    # Process metrics
    print("🔄 Processing metrics...")
    current_metrics = process_metrics(current_actions, current_partner_stats)
    previous_metrics = process_metrics(previous_actions, previous_partner_stats)
    
    # Identify new top 10 partners
    current_top_10 = get_top_partners(current_metrics, n=10)
    print(f"   Current month top 10: {current_top_10}")
    new_top_partners = identify_new_top_partners(current_top_10, historical_tops)
    if new_top_partners:
        print(f"   🆕 New top 10 partners: {', '.join(new_top_partners)}")
    else:
        print(f"   ℹ️  No new top 10 partners this month")
    
    # Calculate changes
    changes = calculate_changes(current_metrics, previous_metrics)
    partner_drivers = identify_partner_drivers(current_metrics, previous_metrics)
    
    # Print summary to console
    def fmt_pct(val):
        return f"{val:.1f}" if val is not None else "N/A"
    
    def fmt_currency(val):
        return f"${val:,.2f}" if val is not None else "N/A"
    
    def fmt_num(val):
        return f"{val:,}" if val is not None else "N/A"
    
    print("\n📊 Summary:")
    print(f"   Actions (Payment Success): {fmt_num(current_metrics['payment_success_actions'])} ({fmt_pct(changes['payment_success_actions']['change_pct'])}% MoM)")
    print(f"   Total Cost: {fmt_currency(current_metrics['total_cost'])} ({fmt_pct(changes['total_cost']['change_pct'])}% MoM)")
    print(f"   CAC: {fmt_currency(current_metrics['cac'])} ({fmt_pct(changes['cac']['change_pct'])}% MoM)")
    print(f"   Reversal Rate: {fmt_pct(current_metrics['reversal_rate'])}%")
    
    # Build and send Slack message
    print("\n📝 Building Slack message...")
    message = build_slack_message(
        current_metrics,
        changes,
        partner_drivers,
        (current_start, current_end),
        (previous_start, previous_end),
        new_top_partners,
        report_type="monthly"
    )
    
    send_to_slack(message)
    
    return {
        "period": f"{current_start} to {current_end}",
        "metrics": current_metrics,
        "changes": changes,
        "partner_drivers": partner_drivers
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "monthly":
        run_monthly_report()
    else:
        run_weekly_report()
