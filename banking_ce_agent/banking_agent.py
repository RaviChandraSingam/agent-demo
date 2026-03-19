"""
banking_agent.py
─────────────────────────────────────────────────────────────────────────────
Banking Domain AI Agent demonstrating all four Contextual Engineering
strategies from the guide:

  WRITE    → LangGraph StateGraph scratchpad + InMemoryStore long-term memory
  SELECT   → RAG over banking knowledge base + memory-aware context injection
  COMPRESS → On-the-fly tool-output summarisation + conversation summary node
  ISOLATE  → Supervisor multi-agent (fraud_agent / loan_agent / support_agent)
             each with its own isolated context window

Usage:
    pip install -r requirements.txt
    cp .env.example .env        # paste your OpenAI API key
    python banking_agent.py
─────────────────────────────────────────────────────────────────────────────
"""

import os
import uuid
from typing import Literal, Annotated

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.markdown import Markdown
from rich.text import Text


# LangChain v1+ imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.tools.retriever import create_retriever_tool

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore
from langgraph.prebuilt import create_react_agent
from typing_extensions import TypedDict

from banking_tools import (
    get_account_balance, get_transaction_history, check_loan_eligibility,
    calculate_emi, get_fd_rates, flag_suspicious_transaction,
    get_active_alerts, raise_support_ticket, check_upi_limit,
    get_customer_profile,
)
from banking_knowledge import BANKING_DOCS

load_dotenv()
console = Console()


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def build_llm(model: str = "openai/gpt-4o", temperature: float = 0):
    # Use ChatOpenAI for OpenAI models; update as needed for Anthropic/Gemini
    return ChatOpenAI(model=model, temperature=temperature)


def section(title: str):
    console.print()
    console.rule(f"[bold cyan]{title}[/bold cyan]")


def print_messages(messages: list):
    """Pretty-print a list of LangChain messages."""
    for msg in messages:
        if isinstance(msg, HumanMessage):
            console.print(Panel(str(msg.content), title="[bold green]🧑 User[/bold green]",
                                border_style="green"))
        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    console.print(Panel(
                        f"Tool: [bold yellow]{tc['name']}[/bold yellow]\nArgs: {tc['args']}",
                        title="[bold yellow]🔧 Tool Call[/bold yellow]", border_style="yellow"))
            if msg.content:
                console.print(Panel(str(msg.content), title="[bold blue]🤖 Assistant[/bold blue]",
                                    border_style="blue"))
        elif isinstance(msg, ToolMessage):
            console.print(Panel(str(msg.content)[:600] + ("..." if len(str(msg.content)) > 600 else ""),
                                title="[bold magenta]⚙️  Tool Result[/bold magenta]",
                                border_style="magenta"))


# ═══════════════════════════════════════════════════════════════════════════════
# 1. SELECT STRATEGY — RAG over Banking Knowledge Base
# ═══════════════════════════════════════════════════════════════════════════════

def build_rag_retriever(llm):
    """
    SELECT strategy:
    Index all banking policy docs into an in-memory vector store.
    Returns a retriever tool the agent can call to fetch relevant policy chunks.
    """
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=300, chunk_overlap=30
    )
    docs = [Document(page_content=d) for d in BANKING_DOCS]
    splits = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    # Use FAISS for in-memory vectorstore (langchain_community)
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    retriever_tool = create_retriever_tool(
        retriever,
        name="search_banking_policy",
        description=(
            "Search the bank's internal policy documents. Use this to answer questions "
            "about loan eligibility criteria, fraud rules, account features, UPI limits, "
            "interest rates, grievance procedures, and KYC requirements."
        ),
    )
    return retriever_tool


# ═══════════════════════════════════════════════════════════════════════════════
# 2. WRITE STRATEGY — StateGraph scratchpad + InMemoryStore long-term memory
# ═══════════════════════════════════════════════════════════════════════════════

class BankingState(TypedDict):
    """
    WRITE strategy: shared scratchpad passed between all graph nodes.
    Each field acts as isolated working memory for a specific concern.
    """
    messages:         Annotated[list, lambda x, y: x + y]   # conversation history
    current_account:  str                                     # active account in session
    fraud_flags:      list                                    # flagged transactions this session
    loan_context:     dict                                    # loan evaluation scratchpad
    session_summary:  str                                     # COMPRESS: rolling summary
    interaction_count: int                                    # turn counter


BANKING_SYSTEM_PROMPT = """You are a knowledgeable and empathetic Banking Assistant for IndiaFirst Bank.

Your responsibilities:
- Answer customer queries about accounts, loans, FDs, UPI, fraud, and support.
- Use available tools to fetch live account data and perform calculations.
- Search banking policy docs when you need rules/limits/rates.
- Be concise, professional, and customer-friendly.
- Always cite policy when making eligibility or limit decisions.
- If fraud is suspected, immediately flag it and reassure the customer.

Current session context will be provided at the start of each turn.
"""

COMPRESS_PROMPT = """You are a banking assistant session summariser.
Given the full conversation so far, write a concise 3-5 sentence summary capturing:
- The customer's account ID (if mentioned)
- Key topics discussed
- Any actions taken (loans checked, fraud flagged, tickets raised)
- Outstanding issues or next steps
Keep it factual and brief."""


# ── Node functions ─────────────────────────────────────────────────────────────

def context_injector(state: BankingState, store: BaseStore) -> dict:
    """
    WRITE + SELECT strategy:
    Inject long-term memory from InMemoryStore into the conversation context
    before the LLM sees the messages. Acts as the scratchpad selection step.
    """
    namespace = ("banking_sessions", state.get("current_account", "anonymous"))
    memories = list(store.search(namespace))

    context_parts = []

    # Inject prior session memory
    if memories:
        prior = memories[0].value.get("summary", "")
        if prior:
            context_parts.append(f"[Prior session summary]\n{prior}")

    # Inject current session scratchpad state
    if state.get("fraud_flags"):
        context_parts.append(f"[Fraud flags this session] {state['fraud_flags']}")
    if state.get("loan_context"):
        context_parts.append(f"[Loan evaluation in progress] {state['loan_context']}")
    if state.get("session_summary"):
        context_parts.append(f"[Session so far] {state['session_summary']}")

    if context_parts:
        context_msg = SystemMessage(content="\n\n".join(context_parts))
        # Prepend context into messages (SELECT: only inject what's relevant)
        updated_messages = [context_msg] + state["messages"]
        return {"messages": updated_messages[len(state["messages"]):]}  # only return the new system msg delta

    return {}


def banking_llm_node(state: BankingState, store: BaseStore) -> dict:
    """
    Core LLM node with all banking tools + RAG retriever bound.
    Implements WRITE (reads state scratchpad) and SELECT (can call RAG tool).
    """
    llm = build_llm()
    all_tools = [
        get_account_balance, get_transaction_history, check_loan_eligibility,
        calculate_emi, get_fd_rates, flag_suspicious_transaction,
        get_active_alerts, raise_support_ticket, check_upi_limit,
        get_customer_profile,
    ]
    # Add RAG tool (SELECT strategy)
    rag_tool = build_rag_retriever(llm)
    all_tools.append(rag_tool)

    llm_with_tools = llm.bind_tools(all_tools)

    # Build messages: system prompt + prior context + conversation
    namespace = ("banking_sessions", state.get("current_account", "anonymous"))
    memories = list(store.search(namespace))
    prior_summary = memories[0].value.get("summary", "") if memories else ""

    system_parts = [BANKING_SYSTEM_PROMPT]
    if prior_summary:
        system_parts.append(f"\n[Memory from prior sessions]\n{prior_summary}")
    if state.get("session_summary"):
        system_parts.append(f"\n[Current session context]\n{state['session_summary']}")
    if state.get("fraud_flags"):
        system_parts.append(f"\n[Active fraud flags this session]\n{state['fraud_flags']}")

    messages_to_send = [SystemMessage(content="\n".join(system_parts))] + state["messages"]

    response = llm_with_tools.invoke(messages_to_send)

    # Detect account ID mentions to update scratchpad (WRITE)
    updates: dict = {"messages": [response], "interaction_count": state.get("interaction_count", 0) + 1}
    for acc_id in ["ACC001", "ACC002", "ACC003"]:
        # Check all user messages for account mentions
        for msg in state["messages"]:
            if hasattr(msg, "content") and acc_id in str(msg.content):
                updates["current_account"] = acc_id
                break

    return updates


def tool_executor_node(state: BankingState) -> dict:
    """
    Execute tool calls from the last AI message.
    COMPRESS strategy: summarise large tool outputs before storing in state.
    """
    last_message = state["messages"][-1]
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return {}

    all_tools_map = {
        t.name: t for t in [
            get_account_balance, get_transaction_history, check_loan_eligibility,
            calculate_emi, get_fd_rates, flag_suspicious_transaction,
            get_active_alerts, raise_support_ticket, check_upi_limit,
            get_customer_profile,
        ]
    }

    tool_messages = []
    fraud_flags = list(state.get("fraud_flags", []))
    loan_context = dict(state.get("loan_context", {}))

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        # Handle RAG tool separately (it's not in our tools map)
        if tool_name == "search_banking_policy":
            # This will be executed by the LLM's bound tools — skip manual execution
            continue

        tool_fn = all_tools_map.get(tool_name)
        if not tool_fn:
            result = f"Tool '{tool_name}' not found."
        else:
            try:
                result = tool_fn.invoke(tool_args)
            except Exception as e:
                result = f"Tool error: {e}"

        # COMPRESS: Update scratchpad with important state (WRITE strategy)
        if tool_name == "flag_suspicious_transaction":
            fraud_flags.append(f"{tool_args.get('transaction_date')}: {tool_args.get('reason')}")
        if tool_name == "check_loan_eligibility":
            loan_context.update({
                "account": tool_args.get("account_id"),
                "amount": tool_args.get("loan_amount"),
                "type": tool_args.get("loan_type"),
                "result_snippet": str(result)[:100],
            })

        tool_messages.append(
            ToolMessage(content=str(result), tool_call_id=tool_call["id"])
        )

    updates: dict = {"messages": tool_messages}
    if fraud_flags != state.get("fraud_flags", []):
        updates["fraud_flags"] = fraud_flags
    if loan_context != state.get("loan_context", {}):
        updates["loan_context"] = loan_context

    return updates


def should_continue(state: BankingState) -> Literal["tools", "compress", "__end__"]:
    """
    Routing logic:
    - If last AI message has tool calls → execute tools
    - Every 3 turns → run compression node to summarise context
    - Otherwise → end turn
    """
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    if state.get("interaction_count", 0) % 3 == 0 and state.get("interaction_count", 0) > 0:
        return "compress"
    return "__end__"


def compress_node(state: BankingState, store: BaseStore) -> dict:
    """
    COMPRESS strategy:
    Summarise the entire conversation history every N turns.
    Saves summary to InMemoryStore for cross-session retrieval (WRITE strategy).
    Also trims messages to last 6 to prevent context overflow.
    """
    llm = build_llm()
    messages_for_summary = [SystemMessage(content=COMPRESS_PROMPT)] + state["messages"]
    summary_response = llm.invoke(messages_for_summary)
    new_summary = summary_response.content

    # WRITE: Persist summary to long-term store (cross-session memory)
    namespace = ("banking_sessions", state.get("current_account", "anonymous"))
    store.put(namespace, "summary", {"summary": new_summary})

    # Trim messages to last 6 (keep recent context, discard old turns)
    trimmed_messages = state["messages"][-6:]

    console.print(Panel(
        f"[italic]{new_summary}[/italic]",
        title="[bold green]🗜️  Context Compressed & Saved to Memory[/bold green]",
        border_style="green"
    ))

    return {
        "session_summary": new_summary,
        "messages": trimmed_messages,
    }


def build_single_agent_graph(store: InMemoryStore, checkpointer: InMemorySaver):
    """
    Build the main single-agent graph with WRITE + SELECT + COMPRESS strategies.
    """
    graph = StateGraph(BankingState)

    graph.add_node("llm",      banking_llm_node)
    graph.add_node("tools",    tool_executor_node)
    graph.add_node("compress", compress_node)

    graph.add_edge(START, "llm")
    graph.add_conditional_edges("llm", should_continue, {
        "tools":    "tools",
        "compress": "compress",
        "__end__":  END,
    })
    graph.add_edge("tools",    "llm")
    graph.add_edge("compress", END)

    return graph.compile(checkpointer=checkpointer, store=store)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ISOLATE STRATEGY — Supervisor multi-agent system
# ═══════════════════════════════════════════════════════════════════════════════

FRAUD_AGENT_PROMPT = """You are a Fraud Detection Specialist at IndiaFirst Bank.
Your ONLY job is to:
- Review transaction history for suspicious patterns
- Flag fraudulent transactions using the flag_suspicious_transaction tool
- Check active alerts on accounts
- Advise on account protection measures
Do NOT handle loans, FDs, or general support. Stay strictly in fraud domain."""

LOAN_AGENT_PROMPT = """You are a Loan & Investment Advisor at IndiaFirst Bank.
Your ONLY job is to:
- Check loan eligibility for customers
- Calculate EMIs for loan proposals
- Provide Fixed Deposit rate information and maturity calculations
- Explain loan products and investment options
Do NOT handle fraud investigations or customer complaints. Stay in loans/investment domain."""

SUPPORT_AGENT_PROMPT = """You are a Customer Support Specialist at IndiaFirst Bank.
Your ONLY job is to:
- Answer general banking queries using the banking policy search tool
- Raise support tickets for unresolved issues
- Check UPI limits and transaction permissions
- Retrieve account balances and customer profiles
Do NOT handle fraud investigations or loan processing. Stay in general support domain."""

SUPERVISOR_PROMPT = """You are the Banking Operations Supervisor at IndiaFirst Bank.
You manage a team of three specialist agents. Route each customer query to the right agent:

- fraud_agent  : Suspicious transactions, fraud alerts, account security concerns
- loan_agent   : Loan eligibility, EMI calculations, FD rates, investment queries
- support_agent: General queries, account balance, UPI limits, policy questions, tickets

Analyse the customer's message carefully and delegate to the most appropriate specialist.
If a query spans multiple domains, break it down and delegate each part separately.
Always provide a final consolidated response after agents complete their work."""


def build_supervisor_graph(llm):
    """
    ISOLATE strategy:
    Three specialist agents each with their own isolated context window
    and domain-specific tool sets. A supervisor routes queries.
    """
    # Each agent only gets domain-relevant tools → context isolation
    fraud_agent = create_react_agent(
        model=llm,
        tools=[get_transaction_history, flag_suspicious_transaction, get_active_alerts],
        name="fraud_agent",
        prompt=FRAUD_AGENT_PROMPT,
    )

    loan_agent = create_react_agent(
        model=llm,
        tools=[check_loan_eligibility, calculate_emi, get_fd_rates, get_customer_profile],
        name="loan_agent",
        prompt=LOAN_AGENT_PROMPT,
    )

    support_agent = create_react_agent(
        model=llm,
        tools=[get_account_balance, raise_support_ticket, check_upi_limit,
               get_customer_profile, build_rag_retriever(llm)],
        name="support_agent",
        prompt=SUPPORT_AGENT_PROMPT,
    )

    # Build supervisor using LangGraph MessagesState + conditional routing
    from langgraph.graph import MessagesState as MS

    class SupervisorState(MS):
        next_agent: str

    def supervisor_node(state: SupervisorState) -> dict:
        supervisor_llm = llm.bind_tools([
            {"type": "function", "function": {
                "name": "route",
                "description": "Route query to specialist agent",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent": {"type": "string", "enum": ["fraud_agent", "loan_agent", "support_agent", "FINISH"]},
                        "reason": {"type": "string"},
                    },
                    "required": ["agent", "reason"],
                },
            }}
        ], tool_choice={"type": "function", "function": {"name": "route"}})

        response = supervisor_llm.invoke(
            [SystemMessage(content=SUPERVISOR_PROMPT)] + state["messages"]
        )
        tool_call = response.tool_calls[0]["args"] if response.tool_calls else {"agent": "FINISH", "reason": "done"}
        return {"next_agent": tool_call["agent"], "messages": [response]}

    def route_to_agent(state: SupervisorState) -> str:
        agent = state.get("next_agent", "FINISH")
        return agent if agent != "FINISH" else END

    def run_fraud_agent(state: SupervisorState) -> dict:
        result = fraud_agent.invoke({"messages": state["messages"]})
        return {"messages": result["messages"][-1:], "next_agent": "FINISH"}

    def run_loan_agent(state: SupervisorState) -> dict:
        result = loan_agent.invoke({"messages": state["messages"]})
        return {"messages": result["messages"][-1:], "next_agent": "FINISH"}

    def run_support_agent(state: SupervisorState) -> dict:
        result = support_agent.invoke({"messages": state["messages"]})
        return {"messages": result["messages"][-1:], "next_agent": "FINISH"}

    builder = StateGraph(SupervisorState)
    builder.add_node("supervisor",    supervisor_node)
    builder.add_node("fraud_agent",   run_fraud_agent)
    builder.add_node("loan_agent",    run_loan_agent)
    builder.add_node("support_agent", run_support_agent)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges("supervisor", route_to_agent, {
        "fraud_agent":   "fraud_agent",
        "loan_agent":    "loan_agent",
        "support_agent": "support_agent",
        END:             END,
    })
    builder.add_edge("fraud_agent",   END)
    builder.add_edge("loan_agent",    END)
    builder.add_edge("support_agent", END)

    return builder.compile()


# ═══════════════════════════════════════════════════════════════════════════════
# 4. DEMO RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

SINGLE_AGENT_DEMO_QUERIES = [
    ("ACC001", "Hi! Can you check my account balance and recent transactions?"),
    ("ACC001", "I see a suspicious night transfer of Rs.1,20,000 on 2026-03-12. Please investigate."),
    ("ACC001", "I want to apply for a personal loan of Rs.10,00,000. Am I eligible?"),
    ("ACC001", "If I get the loan at 10.5% for 48 months, what will my EMI be?"),
    ("ACC001", "What are the UPI daily limits? I want to transfer Rs.90,000 to a new contact."),
    ("ACC002", "Check my account and tell me if there are any alerts."),
]

SUPERVISOR_DEMO_QUERIES = [
    ("ACC001", "I noticed a transfer of Rs.1,20,000 at 2 AM on 2026-03-12. Can you check if it's fraudulent?"),
    ("ACC001", "Also, can you tell me if I'm eligible for a home loan of Rs.50 lakhs?"),
    ("ACC002", "I want to know my account balance and if there are any active fraud alerts."),
    ("ACC002", "Can you also check the eligibility criteria for a personal loan?"),
    ("ACC003", "I want to invest in a Fixed Deposit. What are the current rates and tenure options?"),
    ("ACC003", "I also have a question about UPI limits for my account. Can you help?"),
    ("ACC003", "I noticed a suspicious transaction on my account. Can you investigate?"),
    ("ACC003", "I want to apply for a personal loan of Rs.5,00,000. Am I eligible?"),
    ("ACC003", "If I get the loan at 10.5% for 48 months, what will my EMI be?"),
    ("ACC003", "Can you also check if there are any active fraud alerts on my account?"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# 5. DEMO RUNNERS
# ═══════════════════════════════════════════════════════════════════════════════

def run_single_agent_demo():
    """Run the single-agent graph demo showcasing WRITE + SELECT + COMPRESS."""
    section("SINGLE AGENT DEMO  (WRITE · SELECT · COMPRESS · ISOLATE-lite)")
    console.print(
        "[bold]LangGraph agent with StateGraph scratchpad, RAG, and context compression.[/bold]\n"
    )

    store = InMemoryStore()
    checkpointer = InMemorySaver()
    graph = build_single_agent_graph(store, checkpointer)
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # Maintain running state across turns in the same thread
    current_state: dict = {
        "messages": [],
        "current_account": "",
        "fraud_flags": [],
        "loan_context": {},
        "session_summary": "",
        "interaction_count": 0,
    }

    for account_id, query in SINGLE_AGENT_DEMO_QUERIES:
        section(f"Turn  [{account_id}]")
        console.print(Panel(query, title="[bold green]🧑 User[/bold green]", border_style="green"))

        current_state["messages"] = [HumanMessage(content=query)]
        current_state["current_account"] = account_id

        result = graph.invoke(current_state, config=config)
        print_messages(result["messages"][-3:])

        # Carry over mutable scratchpad fields between turns
        current_state["fraud_flags"]      = result.get("fraud_flags", [])
        current_state["loan_context"]     = result.get("loan_context", {})
        current_state["session_summary"]  = result.get("session_summary", "")
        current_state["interaction_count"] = result.get("interaction_count", 0)


def run_supervisor_demo():
    """Run the supervisor multi-agent demo showcasing ISOLATE strategy."""
    section("SUPERVISOR MULTI-AGENT DEMO  (ISOLATE strategy)")
    console.print(
        "[bold]Supervisor routes each query to the right specialist agent "
        "(fraud / loan / support).[/bold]\n"
    )

    llm = build_llm()
    supervisor_graph = build_supervisor_graph(llm)

    for account_id, query in SUPERVISOR_DEMO_QUERIES:
        section(f"Supervisor Query  [{account_id}]")
        tagged_query = f"[Account: {account_id}] {query}"
        console.print(Panel(tagged_query, title="[bold green]🧑 User[/bold green]", border_style="green"))

        result = supervisor_graph.invoke({"messages": [HumanMessage(content=tagged_query)]})
        print_messages(result["messages"][-3:])


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    console.print(Rule("[bold cyan]Banking CE Agent — Contextual Engineering Demo[/bold cyan]"))
    console.print(
        "[dim]Demonstrating WRITE, SELECT, COMPRESS, and ISOLATE strategies "
        "in a banking customer-experience agent.[/dim]\n"
    )

    mode = sys.argv[1] if len(sys.argv) > 1 else "single"

    if mode == "supervisor":
        run_supervisor_demo()
    elif mode == "both":
        run_single_agent_demo()
        run_supervisor_demo()
    else:
        run_single_agent_demo()