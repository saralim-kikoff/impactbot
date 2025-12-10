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
    Fetch aggregated stats including clicks.
    Uses the Reports endpoint for Performance by Day report.
    """
    total_clicks = 0
    report_id = "att_adv_performance_by_day_pm_only"
    
    # Try different parameter combinations
    param_sets = [
        # With timestamps
        {"START_DATE": f"{start_date}T00:00:00Z", "END_DATE": f"{end_date}T23:59:59Z", "CAMPAIGN_ID": CAMPAIGN_ID},
        # Without campaign filter
        {"START_DATE": start_date, "END_DATE": end_date},
        # Different date format
        {"Start Date": start_date, "End Date": end_date},
    ]
    
    for params in param_sets:
        try:
            print(f"   🔍 Trying params: {list(params.keys())}")
            response = requests.get(
                f"{BASE_URL}/Reports/{report_id}",
                auth=get_auth(),
                params=params,
                headers={"Accept": "application/json"}
            )
            
            if response.status_code != 200:
                print(f"   ⚠️  Status {response.status_code}")
                continue
            
            data = response.json()
            records = data.get("Records", [])
            
            if records:
                # Check if first record has actual data
                sample = records[0]
                clicks_value = sample.get("Clicks", "")
                
                print(f"   📋 Sample Clicks value: '{clicks_value}' (type: {type(clicks_value).__name__})")
                
                if clicks_value and clicks_value != "":
                    for record in records:
                        clicks = record.get("Clicks") or 0
                        if clicks:
                            total_clicks += int(clicks)
                    
                    print(f"   ✅ Total clicks: {total_clicks:,}")
                    return {"_total": {"clicks": total_clicks, "cost": 0}}
                else:
                    print(f"   ⚠️  Records have empty values")
                    
        except Exception as e:
            print(f"   ⚠️  Error: {e}")
    
    print(f"   ⚠️  No click data available from report")
    return {"_total": {"clicks": 0, "cost": 0}}


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
    total_clicks = sum(p.get("clicks", 0) for p in partner_stats.values()) if partner_stats else 0
    total_cost = sum(p.get("cost", 0) for p in partner_stats.values()) if partner_stats else 0
    
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
    """
    current_partners = current_metrics["partner_metrics"]
    previous_partners = previous_metrics["partner_metrics"]
    
    # Get all partners
    all_partners = set(current_partners.keys()) | set(previous_partners.keys())
    
    # Calculate changes per partner
    partner_changes = []
    for partner in all_partners:
        curr = current_partners.get(partner, {"payment_success": 0, "cost": 0, "clicks": 0})
        prev = previous_partners.get(partner, {"payment_success": 0, "cost": 0, "clicks": 0})
        
        partner_changes.append({
            "partner": partner,
            "actions_change": curr.get("payment_success", 0) - prev.get("payment_success", 0),
            "actions_current": curr.get("payment_success", 0),
            "cost_change": curr.get("cost", 0) - prev.get("cost", 0),
            "cost_current": curr.get("cost", 0),
            "clicks_change": curr.get("clicks", 0) - prev.get("clicks", 0),
            "clicks_current": curr.get("clicks", 0),
        })
    
    # Sort by absolute change to find biggest movers
    actions_movers = sorted(partner_changes, key=lambda x: abs(x["actions_change"]), reverse=True)[:3]
    cost_movers = sorted(partner_changes, key=lambda x: abs(x["cost_change"]), reverse=True)[:3]
    clicks_movers = sorted(partner_changes, key=lambda x: abs(x["clicks_change"]), reverse=True)[:3]
    
    # Filter out zero changes
    actions_movers = [p for p in actions_movers if p["actions_change"] != 0]
    cost_movers = [p for p in cost_movers if p["cost_change"] != 0]
    clicks_movers = [p for p in clicks_movers if p["clicks_change"] != 0]
    
    return {
        "actions": actions_movers,
        "cost": cost_movers,
        "clicks": clicks_movers
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
        elif metric_type == "cost":
            change = p["cost_change"]
            sign = "+" if change > 0 else ""
            lines.append(f"• *{p['partner']}*: {sign}${change:,.2f}")
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
    new_top_partners: list[str] = None
) -> Dict[str, Any]:
    """Build Slack Block Kit message."""
    
    start_date, end_date = date_range
    prev_start, prev_end = prev_date_range
    partner_metrics = current.get("partner_metrics", {})
    
    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": "📊 Weekly Impact Affiliate Performance Report",
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
        {
            "type": "section",
            "fields": [
                {
                    "type": "mrkdwn",
                    "text": f"*Actions (Payment Success)*\n*{format_number(current['payment_success_actions'])}*\nvs {format_number(changes['payment_success_actions']['previous'])} prev\n{format_trend(changes['payment_success_actions'])} WoW"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Total Cost*\n*{format_currency(current['total_cost'])}*\nvs {format_currency(changes['total_cost']['previous'])} prev\n{format_trend(changes['total_cost'], is_inverse=True)} WoW"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*CAC*\n*{format_currency(current['cac'])}*\nvs {format_currency(changes['cac']['previous'])} prev\n{format_trend(changes['cac'], is_inverse=True)} WoW"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Reversal Rate*\n*{format_pct(current['reversal_rate'])}*\nvs {format_pct(changes['reversal_rate']['previous'])} prev\n{format_trend(changes['reversal_rate'], is_inverse=True)} WoW"
                }
            ]
        },
        {
            "type": "section",
            "fields": [
                {
                    "type": "mrkdwn",
                    "text": f"*Clicks*\n*{format_number(current['clicks'])}*\nvs {format_number(changes['clicks']['previous'])} prev\n{format_trend(changes['clicks'])} WoW"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Conversion Rate*\n*{format_pct(current['conversion_rate'])}*\nvs {format_pct(changes['conversion_rate']['previous'])} prev\n{format_trend(changes['conversion_rate'])} WoW"
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
                "text": f"*Cost Movers:*\n{format_partner_movers(partner_drivers['cost'], 'cost')}"
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


if __name__ == "__main__":
    run_weekly_report()
