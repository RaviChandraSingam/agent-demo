# Banking domain tools

# ...existing code...
"""
Banking domain tools for the AI agent.
These simulate real banking operations.
"""

import random
from datetime import datetime, timedelta
from typing import Optional
from langchain_core.tools import tool


# ─── Mock Database ─────────────────────────────────────────────────────────────

MOCK_ACCOUNTS = {
    "ACC001": {
        "name": "Rajesh Kumar",
        "balance": 125000.0,
        "account_type": "savings",
        "credit_score": 780,
        "monthly_income": 85000,
        "kyc_verified": True,
        "city": "Bengaluru",
    },
    "ACC002": {
        "name": "Priya Sharma",
        "balance": 32000.0,
        "account_type": "savings",
        "credit_score": 650,
        "monthly_income": 45000,
        "kyc_verified": True,
        "city": "Mumbai",
    },
    "ACC003": {
        "name": "Amit Patel",
        "balance": 500000.0,
        "account_type": "current",
        "credit_score": 820,
        "monthly_income": 200000,
        "kyc_verified": True,
        "city": "Ahmedabad",
    },
}

MOCK_TRANSACTIONS = {
    "ACC001": [
        {"date": "2026-03-18", "amount": -15000, "type": "debit", "desc": "Online Shopping", "merchant": "Amazon"},
        {"date": "2026-03-17", "amount": 85000, "type": "credit", "desc": "Salary Credit", "merchant": "Employer"},
        {"date": "2026-03-15", "amount": -5000, "type": "debit", "desc": "ATM Withdrawal", "merchant": "ATM"},
        {"date": "2026-03-12", "amount": -120000, "type": "debit", "desc": "Night transfer", "merchant": "NEFT", "flagged": True},
        {"date": "2026-03-10", "amount": -8000, "type": "debit", "desc": "Restaurant", "merchant": "Swiggy"},
    ],
    "ACC002": [
        {"date": "2026-03-19", "amount": -200000, "type": "debit", "desc": "International transfer", "merchant": "Wire", "flagged": True},
        {"date": "2026-03-18", "amount": 45000, "type": "credit", "desc": "Salary Credit", "merchant": "Employer"},
    ],
}

ACTIVE_ALERTS = {
    "ACC001": ["Suspicious transaction flagged on 2026-03-12: Rs. 1,20,000 night transfer"],
    "ACC002": ["Suspicious international transfer of Rs. 2,00,000 flagged on 2026-03-19"],
}


# ─── Banking Tools ─────────────────────────────────────────────────────────────

@tool
def get_account_balance(account_id: str) -> str:
    """Get the current account balance and basic account details for a given account ID."""
    if account_id not in MOCK_ACCOUNTS:
        return f"Account {account_id} not found."
    acc = MOCK_ACCOUNTS[account_id]
    return (
        f"Account: {account_id} | Holder: {acc['name']}\n"
        f"Balance: Rs. {acc['balance']:,.2f}\n"
        f"Account Type: {acc['account_type'].title()}\n"
        f"City: {acc['city']}"
    )


@tool
def get_transaction_history(account_id: str, days: int = 30) -> str:
    """Retrieve recent transaction history for an account. Specify number of days to look back."""
    if account_id not in MOCK_TRANSACTIONS:
        return f"No transaction history found for {account_id}."
    txns = MOCK_TRANSACTIONS[account_id]
    result = f"Transaction history for {account_id} (last {days} days):\n"
    for t in txns:
        flag = " ⚠️ FLAGGED" if t.get("flagged") else ""
        result += (
            f"  [{t['date']}] {t['type'].upper():6} | Rs. {abs(t['amount']):>10,.0f}"
            f" | {t['desc']} ({t['merchant']}){flag}\n"
        )
    return result


@tool
def check_loan_eligibility(account_id: str, loan_amount: float, loan_type: str = "personal") -> str:
    """
    Check if a customer is eligible for a loan.
    loan_type can be 'personal' or 'home'.
    loan_amount should be in Rupees.
    """
    if account_id not in MOCK_ACCOUNTS:
        return f"Account {account_id} not found."
    acc = MOCK_ACCOUNTS[account_id]
    
    eligible = True
    reasons = []
    
    if loan_type == "personal":
        min_score = 700
        max_amount = acc["monthly_income"] * 20
        if acc["credit_score"] < min_score:
            eligible = False
            reasons.append(f"Credit score {acc['credit_score']} is below minimum {min_score}")
        if loan_amount > max_amount:
            eligible = False
            reasons.append(f"Requested Rs. {loan_amount:,.0f} exceeds max eligible Rs. {max_amount:,.0f}")
        if acc["monthly_income"] < 25000:
            eligible = False
            reasons.append("Monthly income below Rs. 25,000 minimum")
    elif loan_type == "home":
        if acc["credit_score"] < 750:
            eligible = False
            reasons.append(f"Credit score {acc['credit_score']} below minimum 750 for home loan")

    if eligible:
        rate = 10.5 if acc["credit_score"] >= 780 else (13.5 if acc["credit_score"] >= 720 else 17.5)
        return (
            f"✅ ELIGIBLE for {loan_type.title()} Loan\n"
            f"Customer: {acc['name']} | Credit Score: {acc['credit_score']}\n"
            f"Approved Amount: Rs. {min(loan_amount, acc['monthly_income']*20):,.0f}\n"
            f"Interest Rate: {rate}% p.a.\n"
            f"Max Tenure: {'60' if loan_type == 'personal' else '360'} months"
        )
    else:
        return (
            f"❌ NOT ELIGIBLE for {loan_type.title()} Loan\n"
            f"Customer: {acc['name']} | Reasons:\n"
            + "\n".join(f"  - {r}" for r in reasons)
        )


@tool
def flag_suspicious_transaction(account_id: str, transaction_date: str, reason: str) -> str:
    """Flag a transaction as suspicious and initiate fraud review process."""
    if account_id not in MOCK_ACCOUNTS:
        return f"Account {account_id} not found."
    acc = MOCK_ACCOUNTS[account_id]
    case_id = f"FRAUD-{random.randint(10000, 99999)}"
    return (
        f"🚨 Fraud Alert Raised\n"
        f"Case ID: {case_id}\n"
        f"Account: {account_id} | Customer: {acc['name']}\n"
        f"Transaction Date: {transaction_date}\n"
        f"Reason: {reason}\n"
        f"Status: Under Review — Expected resolution: 24 hours\n"
        f"Temporary account protection: ENABLED"
    )


@tool
def get_active_alerts(account_id: str) -> str:
    """Get any active fraud or security alerts for an account."""
    alerts = ACTIVE_ALERTS.get(account_id, [])
    if not alerts:
        return f"No active alerts for account {account_id}."
    result = f"⚠️ Active Alerts for {account_id}:\n"
    for i, a in enumerate(alerts, 1):
        result += f"  {i}. {a}\n"
    return result


@tool
def calculate_emi(principal: float, annual_rate: float, tenure_months: int) -> str:
    """
    Calculate EMI for a loan.
    principal: loan amount in Rs.
    annual_rate: interest rate per annum (e.g., 10.5 for 10.5%)
    tenure_months: loan duration in months
    """
    r = annual_rate / (12 * 100)
    if r == 0:
        emi = principal / tenure_months
    else:
        emi = principal * r * (1 + r) ** tenure_months / ((1 + r) ** tenure_months - 1)
    total_payment = emi * tenure_months
    total_interest = total_payment - principal
    return (
        f"📊 EMI Calculation\n"
        f"Principal: Rs. {principal:,.2f}\n"
        f"Interest Rate: {annual_rate}% p.a.\n"
        f"Tenure: {tenure_months} months\n"
        f"Monthly EMI: Rs. {emi:,.2f}\n"
        f"Total Payment: Rs. {total_payment:,.2f}\n"
        f"Total Interest: Rs. {total_interest:,.2f}"
    )


@tool
def get_fd_rates(tenure_days: int, amount: float = 100000) -> str:
    """Get Fixed Deposit interest rates for a given tenure in days and calculate maturity amount."""
    if tenure_days < 365:
        rate = 6.5
        category = "Less than 1 year"
    elif tenure_days < 1095:
        rate = 7.0
        category = "1 to 3 years"
    elif tenure_days < 1825:
        rate = 7.5
        category = "3 to 5 years"
    else:
        rate = 7.25
        category = "More than 5 years"

    years = tenure_days / 365
    maturity_amount = amount * (1 + rate / 100) ** years
    interest_earned = maturity_amount - amount

    return (
        f"📈 Fixed Deposit Rates\n"
        f"Tenure: {tenure_days} days ({category})\n"
        f"Interest Rate: {rate}% p.a.\n"
        f"Principal: Rs. {amount:,.2f}\n"
        f"Maturity Amount: Rs. {maturity_amount:,.2f}\n"
        f"Interest Earned: Rs. {interest_earned:,.2f}\n"
        f"Minimum Deposit: Rs. 10,000 | Premature withdrawal penalty: 1% rate reduction"
    )


@tool
def raise_support_ticket(account_id: str, issue_type: str, description: str) -> str:
    """Raise a customer support ticket for an unresolved issue.
    issue_type can be: 'account', 'transaction', 'loan', 'fraud', 'general'.
    """
    if account_id not in MOCK_ACCOUNTS:
        return f"Account {account_id} not found."
    acc = MOCK_ACCOUNTS[account_id]
    ticket_id = f"TKT-{random.randint(100000, 999999)}"
    resolution_days = {
        "fraud": "24 hours",
        "transaction": "7 working days",
        "loan": "5 working days",
        "account": "3 working days",
        "general": "3 working days",
    }.get(issue_type.lower(), "3 working days")
    return (
        f"✅ Support Ticket Raised\n"
        f"Ticket ID: {ticket_id}\n"
        f"Account: {account_id} | Customer: {acc['name']}\n"
        f"Issue Type: {issue_type.title()}\n"
        f"Description: {description}\n"
        f"Expected Resolution: {resolution_days}\n"
        f"You will receive updates via SMS and email."
    )


@tool
def check_upi_limit(account_id: str, amount: float) -> str:
    """Check if a UPI transaction of the given amount is within allowed limits for the account."""
    if account_id not in MOCK_ACCOUNTS:
        return f"Account {account_id} not found."
    acc = MOCK_ACCOUNTS[account_id]
    per_txn_limit = 100_000
    daily_limit = 200_000
    cooling_threshold = 25_000

    result = (
        f"💳 UPI Limit Check for {account_id} ({acc['name']})\n"
        f"Requested Amount: Rs. {amount:,.2f}\n"
        f"Per Transaction Limit: Rs. {per_txn_limit:,.0f}\n"
        f"Daily Limit: Rs. {daily_limit:,.0f}\n\n"
    )
    if amount > per_txn_limit:
        result += f"❌ EXCEEDS per-transaction limit of Rs. {per_txn_limit:,.0f}. Transaction not allowed."
    elif amount > cooling_threshold:
        result += (
            f"⚠️  Amount exceeds Rs. {cooling_threshold:,.0f}. "
            f"A 4-hour cooling period applies for new beneficiaries.\n"
            f"✅ Transaction allowed within daily limit."
        )
    else:
        result += "✅ Transaction allowed. No restrictions apply."
    return result


@tool
def get_customer_profile(account_id: str) -> str:
    """Get full customer profile including credit score, KYC status, and account details."""
    if account_id not in MOCK_ACCOUNTS:
        return f"Account {account_id} not found."
    acc = MOCK_ACCOUNTS[account_id]
    return (
        f"👤 Customer Profile\n"
        f"Account ID:       {account_id}\n"
        f"Name:             {acc['name']}\n"
        f"City:             {acc['city']}\n"
        f"Account Type:     {acc['account_type'].title()}\n"
        f"Monthly Income:   Rs. {acc['monthly_income']:,.0f}\n"
        f"Credit Score:     {acc['credit_score']} (CIBIL)\n"
        f"KYC Verified:     {'Yes ✅' if acc['kyc_verified'] else 'No ❌'}\n"
        f"Current Balance:  Rs. {acc['balance']:,.2f}"
    )
