# Banking CE Agent

A **Contextual Engineering** demo featuring an AI-powered banking customer-experience agent built with LangGraph and LangChain. The project showcases all four CE strategies in a realistic banking domain.

---

## Contextual Engineering Strategies

| Strategy | Implementation |
|----------|---------------|
| **WRITE** | LangGraph `StateGraph` scratchpad (`BankingState`) + `InMemoryStore` for cross-session long-term memory |
| **SELECT** | RAG over an internal banking knowledge base (loan policies, fraud rules, UPI limits, grievance procedures) |
| **COMPRESS** | On-the-fly conversation summarisation every 3 turns; trimmed message history to prevent context overflow |
| **ISOLATE** | Supervisor multi-agent system with three specialist agents (`fraud_agent`, `loan_agent`, `support_agent`), each with its own isolated context window and domain-scoped tool set |

---

## Project Structure

```
banking_ce_agent/
├── banking_agent.py        # Main application — graphs, nodes, demo runner
├── banking_tools.py        # Banking domain tools (balance, loans, fraud, UPI, FD, support)
├── banking_knowledge.py    # Banking knowledge base documents for RAG
├── requirements.txt        # Python dependencies
└── README.md
```

---

## Tools Available to the Agent

| Tool | Description |
|------|-------------|
| `get_account_balance` | Current balance and account details |
| `get_transaction_history` | Recent transactions with fraud flags |
| `check_loan_eligibility` | Personal / home loan eligibility check |
| `calculate_emi` | EMI, total payment, and total interest |
| `get_fd_rates` | FD interest rates and maturity calculation |
| `flag_suspicious_transaction` | Raise a fraud case with case ID |
| `get_active_alerts` | Active fraud/security alerts on an account |
| `raise_support_ticket` | Create a support ticket with SLA |
| `check_upi_limit` | UPI per-transaction and daily limit validation |
| `get_customer_profile` | Full profile — credit score, KYC, income |
| `search_banking_policy` | RAG retriever over internal policy documents |

---

## Setup

### Prerequisites
- Python 3.10+
- An OpenAI API key

### Installation

```bash
cd banking_ce_agent
pip install -r requirements.txt
```

Create a `.env` file in `banking_ce_agent/`:

```
OPENAI_API_KEY=sk-...
```

---

## Usage

```bash
# Single agent demo (WRITE · SELECT · COMPRESS)
python banking_agent.py

# Supervisor multi-agent demo (ISOLATE)
python banking_agent.py supervisor

# Run both demos sequentially
python banking_agent.py both
```

---

## Architecture

### Single Agent (default mode)

```
START → [llm node] ──tool calls?──► [tool executor] ──► [llm node]
                  ↘ every 3 turns ► [compress node] ──► END
                  ↘ otherwise     ──────────────────────► END
```

- `banking_llm_node` — binds all tools + RAG retriever, reads scratchpad from state
- `tool_executor_node` — runs tool calls, writes fraud flags / loan context back to state
- `compress_node` — summarises conversation, persists to `InMemoryStore`, trims message history

### Supervisor Multi-Agent (supervisor mode)

```
START → [supervisor] ──routes──► [fraud_agent]   ──► END
                               ► [loan_agent]    ──► END
                               ► [support_agent] ──► END
```

Each specialist agent receives only the tools relevant to its domain, ensuring true context isolation.

---

## Mock Data

The demo uses three pre-loaded accounts:

| Account | Customer | Notable Data |
|---------|----------|--------------|
| ACC001 | Rajesh Kumar | Flagged ₹1,20,000 night transfer (2026-03-12) |
| ACC002 | Priya Sharma | Flagged ₹2,00,000 international wire (2026-03-19) |
| ACC003 | Amit Patel | High-income current account, credit score 820 |