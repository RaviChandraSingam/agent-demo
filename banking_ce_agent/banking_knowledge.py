# Banking KB for RAG

# ...existing code...
"""
Banking domain knowledge base documents for RAG.
"""

BANKING_DOCS = [
    """
    LOAN ELIGIBILITY POLICY
    Personal Loan Eligibility:
    - Minimum age: 21 years, Maximum age: 60 years at loan maturity
    - Minimum monthly income: Rs. 25,000 (salaried) or Rs. 40,000 (self-employed)
    - Minimum credit score: 700 (CIBIL)
    - Maximum loan amount: 20x monthly salary, up to Rs. 40 lakhs
    - Loan tenure: 12 to 60 months
    - Interest rate: 10.5% to 18% per annum based on credit score
    - Documents required: PAN card, Aadhaar, last 3 months salary slips, 6 months bank statement
    
    Home Loan Eligibility:
    - Minimum credit score: 750
    - Loan-to-Value (LTV) ratio: up to 80% of property value
    - Maximum tenure: 30 years
    - Interest rate: 8.5% to 10.5% per annum
    """,

    """
    FRAUD DETECTION POLICY
    Transaction Monitoring Rules:
    - Flag transactions above Rs. 1,00,000 in a single day from a new device
    - Flag international transactions if customer has no prior international history
    - Flag 3 or more failed PIN attempts within 15 minutes — auto-lock account
    - Flag transactions at odd hours (12 AM - 5 AM) above Rs. 50,000
    - Velocity check: more than 10 transactions per hour triggers review
    
    High-Risk Merchants:
    - Cryptocurrency exchanges
    - Overseas lottery sites
    - Unregistered gambling portals
    
    Customer Alert Protocol:
    - SMS alert for every transaction above Rs. 5,000
    - Email alert for transactions above Rs. 25,000
    - Call verification for transactions above Rs. 1,00,000
    """,

    """
    ACCOUNT TYPES AND FEATURES
    Savings Account:
    - Minimum balance: Rs. 10,000 (urban), Rs. 5,000 (semi-urban), Rs. 1,000 (rural)
    - Interest rate: 3.5% per annum on daily closing balance
    - Free ATM transactions: 5 per month at own ATMs, 3 at other bank ATMs
    - Debit card: Visa Classic (default), upgrade available to Visa Platinum
    
    Current Account:
    - Minimum balance: Rs. 50,000
    - No interest paid on balance
    - Unlimited transactions
    - Free NEFT/RTGS up to 10 transactions per month
    
    Fixed Deposit:
    - Minimum deposit: Rs. 10,000
    - Tenure: 7 days to 10 years
    - Interest rates: 6.5% (< 1 year), 7.0% (1-3 years), 7.5% (3-5 years), 7.25% (> 5 years)
    - Premature withdrawal penalty: 1% reduction in applicable rate
    """,

    """
    CUSTOMER GRIEVANCE AND ESCALATION POLICY
    Resolution Timeline:
    - Account-related queries: resolved within 3 working days
    - Transaction disputes: resolved within 7 working days
    - Loan processing queries: resolved within 5 working days
    - Fraud complaints: urgent — resolved within 24 hours, temporary credit within 10 days
    
    Escalation Levels:
    Level 1: Branch manager (Day 1-3)
    Level 2: Nodal officer (Day 4-7)
    Level 3: Principal Nodal Officer (Day 8-15)
    Level 4: Banking Ombudsman (after 30 days if unresolved)
    
    Compensation Policy:
    - Wrongful debit: Full reversal + Rs. 100 per day delay
    - ATM cash dispense failure: Rs. 100 per day after 5 days
    """,

    """
    DIGITAL BANKING AND UPI GUIDELINES
    UPI Transaction Limits:
    - Per transaction: Rs. 1,00,000
    - Per day: Rs. 2,00,000 (enhanced limit for verified accounts)
    - New beneficiary cooling period: 4 hours for amounts > Rs. 25,000
    
    Internet Banking:
    - NEFT: 24x7, charges waived for amounts < Rs. 10,000
    - RTGS: Available 7 AM to 6 PM, minimum Rs. 2,00,000
    - IMPS: 24x7, up to Rs. 5,00,000 per transaction
    
    Mobile Banking Security:
    - mPIN must be changed every 90 days
    - 3 wrong mPIN attempts locks mobile banking for 24 hours
    - Device binding: only 2 devices can be linked at a time
    """,
]
