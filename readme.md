# Banking CE Agent

A **Contextual Engineering** demo featuring an AI-powered banking customer-experience agent built with LangGraph and LangChain. The project showcases all four CE strategies in a realistic banking domain.

https://blog.langchain.com/context-engineering-for-agents/



---

## Contextual Engineering Strategies

| Strategy | Implementation |
|----------|---------------|
| **WRITE** | LangGraph `StateGraph` scratchpad (`BankingState`) + `InMemoryStore` for cross-session long-term memory |
| **SELECT** | RAG over an internal banking knowledge base (loan policies, fraud rules, UPI limits, grievance procedures) |
| **COMPRESS** | On-the-fly conversation summarisation every 3 turns (single agent) / every 3 queries per account (supervisor); message history trimmed to last 6 to prevent context overflow; summaries persisted to `InMemoryStore` |
| **ISOLATE** | Supervisor multi-agent system with three specialist agents (`fraud_agent`, `loan_agent`, `support_agent`), each with its own isolated context window and domain-scoped tool set; prior account memory is also injected into each sub-agent (SELECT) |

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
# Interactive chat — supervisor mode (default)
python banking_agent.py
python banking_agent.py chat

# Interactive chat — single-agent mode
python banking_agent.py chat single

# Scripted single-agent demo (WRITE · SELECT · COMPRESS)
python banking_agent.py single

# Scripted supervisor multi-agent demo (ISOLATE · WRITE · COMPRESS · SELECT)
python banking_agent.py supervisor

# Run both scripted demos sequentially
python banking_agent.py both
```

### Interactive Chat

The default entry point launches an interactive chat session. Type any banking query at the prompt and press Enter. The agent will route your message, call relevant tools, and respond in real time.

```
You> My account is ACC001. Check my balance.
You> I see a suspicious transfer — please investigate.
You> exit
```

- In **supervisor mode** (default), queries are routed to the appropriate specialist (`fraud_agent`, `loan_agent`, or `support_agent`). If your message contains an account ID (`ACC001` etc.) it is extracted automatically; otherwise the shell prompts you for one.
- In **single-agent mode**, all queries are handled by a single graph with full tool access.
- Memory is displayed before each query — a **MEMORY HIT** panel shows prior context being injected; a **MEMORY MISS** panel confirms a fresh start.
- Type `exit`, `quit`, or `q` to end the session.

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
- `compress_node` — summarises conversation every 3 interaction turns, persists to `InMemoryStore`, trims message history to last 6 messages

### Supervisor Multi-Agent (supervisor mode)

```
START → [supervisor] ──routes──► [fraud_agent]   ──► END
                               ► [loan_agent]    ──► END
                               ► [support_agent] ──► END
```

Each specialist agent receives only the tools relevant to its domain, ensuring true context isolation. Prior account memory (summary + last 5 interactions) is also injected into each sub-agent as a `SystemMessage`, applying the SELECT strategy within isolated contexts. Compression runs every 3 queries per account and resets the interaction log.

---

## Mock Data

The demo uses three pre-loaded accounts:

| Account | Customer | Notable Data |
|---------|----------|--------------|
| ACC001 | Rajesh Kumar | Flagged ₹1,20,000 night transfer (2026-03-12) |
| ACC002 | Priya Sharma | Flagged ₹2,00,000 international wire (2026-03-19) |
| ACC003 | Amit Patel | High-income current account, credit score 820 |