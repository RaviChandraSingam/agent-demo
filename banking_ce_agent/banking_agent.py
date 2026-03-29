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

───────────────────────────────
 Prompt for Testing:
───────────────────────────────
Hi! My account ID is ACC001. Can you check my balance and recent transactions?
I see a suspicious night transfer of Rs.1,20,000 on 2026-03-12. Please investigate.
I want to apply for a personal loan of Rs.10,00,000. Am I eligible?
If I get the loan at 10.5% for 48 months, what will my EMI be?
What are the UPI daily limits? I want to transfer Rs.90,000 to a new contact.
Check my account and tell me if there are any alerts.

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
from langchain_core.tools import Tool

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

def build_llm(model: str = "gpt-4o-mini", temperature: float = 0):
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


def print_memory_status(account_id: str, store: InMemoryStore, namespace_prefix: str = "supervisor_sessions"):
    """
    Print what memory (if any) exists for this account before the query is processed.
    Clarifies whether the agent will be informed by prior context or starting fresh.
    Memory provides CONTEXT to the LLM — tools are always called for live data.
    """
    namespace = (namespace_prefix, account_id)
    memories = list(store.search(namespace))
    if memories:
        mem = memories[0].value
        summary = mem.get("summary", "")
        interactions = mem.get("interactions", [])
        lines = []
        if summary:
            lines.append(f"[bold]Stored summary:[/bold] {summary[:250]}{'...' if len(summary) > 250 else ''}")
        if interactions:
            lines.append(f"[bold]Stored interactions:[/bold] {len(interactions)} entry/entries cached")
        lines.append("")
        lines.append("[italic dim]→ This context is INJECTED into the LLM prompt.[/italic dim]")
        lines.append("[italic dim]→ Tools will still be called for fresh, live data — memory does NOT replace tool calls.[/italic dim]")
        console.print(Panel(
            "\n".join(lines),
            title=f"[bold cyan]🧠 MEMORY HIT — Prior context found for {account_id}[/bold cyan]",
            border_style="cyan",
        ))
    else:
        console.print(Panel(
            f"[dim]No prior memory for {account_id}. Starting fresh — all data will be fetched via tools.[/dim]",
            title=f"[bold dim]🧠 MEMORY MISS — No prior context for {account_id}[/bold dim]",
            border_style="dim",
        ))


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

    def _search_banking_policy(query: str) -> str:
        docs = retriever.get_relevant_documents(query)
        return "\n\n".join(d.page_content for d in docs)

    retriever_tool = Tool(
        name="search_banking_policy",
        description=(
            "Search the bank's internal policy documents. Use this to answer questions "
            "about loan eligibility criteria, fraud rules, account features, UPI limits, "
            "interest rates, grievance procedures, and KYC requirements."
        ),
        func=_search_banking_policy,
        args_schema=None,
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

SUPERVISOR_COMPRESS_PROMPT = """You are a banking supervisor session summariser.
Given the prior summary and recent interactions for a specific customer account, write a concise 3-5 sentence summary capturing:
- The customer's account ID
- Key topics across all their queries (fraud, loans, support, UPI, etc.)
- Actions taken (fraud flagged, loan eligibility checked, tickets raised, etc.)
- Outstanding issues or pending follow-ups
Keep it factual, brief, and strictly scoped to this customer."""


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
    # If we already have a current_account, keep it
    current_account = state.get("current_account")
    # Search for account IDs in user messages if not already set
    if not current_account:
        for msg in state["messages"]:
            if hasattr(msg, "content"):
                content = str(msg.content)
                # Look for any pattern like ACC followed by 3+ digits
                import re
                match = re.search(r"ACC\d{3,}", content)
                if match:
                    current_account = match.group(0)
                    break
    if current_account:
        updates["current_account"] = current_account
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
        tool_call_id = tool_call["id"]

        # List of tools that require account_id
        tools_require_account = {
            "get_account_balance", "get_transaction_history", "check_loan_eligibility",
            "flag_suspicious_transaction", "get_active_alerts", "raise_support_ticket",
            "check_upi_limit", "get_customer_profile"
        }

        # If the tool requires account_id and it's missing, prompt the user only now
        if tool_name in tools_require_account and ("account_id" not in tool_args or not tool_args.get("account_id")):
            tool_messages.append(
                ToolMessage(
                    content="To proceed, please provide your account ID (e.g., ACC001).",
                    tool_call_id=tool_call_id
                )
            )
            continue

        # Handle RAG tool (search_banking_policy) by invoking via LLM's bound tools
        if tool_name == "search_banking_policy":
            llm = build_llm()
            rag_tool = build_rag_retriever(llm)
            try:
                result = rag_tool.invoke(tool_args)
            except Exception as e:
                result = f"Tool error: {e}"
            tool_messages.append(
                ToolMessage(content=str(result), tool_call_id=tool_call_id)
            )
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
            ToolMessage(content=str(result), tool_call_id=tool_call_id)
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


def build_supervisor_graph(llm, store: InMemoryStore):
    """
    ISOLATE + WRITE + COMPRESS strategy:
    - Three specialist agents with isolated, domain-scoped tool sets (ISOLATE)
    - Per-account InMemoryStore for cross-query memory, keyed by account ID (WRITE)
    - Every 3 queries per account, summarise and persist to store (COMPRESS)
    - RAG retriever available to support_agent (SELECT)
    """
    import re as _re

    # Each agent only gets domain-relevant tools → context isolation (ISOLATE)
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

    from langgraph.graph import MessagesState as MS

    class SupervisorState(MS):
        next_agent: str
        current_account: str   # WRITE: scopes store namespace per user
        query_count: int       # WRITE: tracks queries for COMPRESS trigger
        session_summary: str   # COMPRESS: rolling per-account summary

    def _extract_account(messages: list) -> str:
        """Extract account ID from the most recent HumanMessage."""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                m = _re.search(r"ACC\d{3,}", str(msg.content))
                if m:
                    return m.group(0)
        return "anonymous"

    def _get_account_memory(account_id: str) -> dict:
        """SELECT: Retrieve this account's stored memory from the store."""
        namespace = ("supervisor_sessions", account_id)
        memories = list(store.search(namespace))
        return memories[0].value if memories else {}

    def _save_interaction(account_id: str, query: str, response: str):
        """WRITE: Append the latest interaction to per-account store."""
        namespace = ("supervisor_sessions", account_id)
        memory = _get_account_memory(account_id)
        interactions = memory.get("interactions", [])
        interactions.append({"query": query[:200], "response": response[:300]})
        store.put(namespace, "context", {
            "summary": memory.get("summary", ""),
            "interactions": interactions[-5:],  # retain last 5 interactions
        })
        console.print(Panel(
            f"Saved interaction #{len(interactions)} for [bold]{account_id}[/bold] to memory store.\n"
            f"[dim]Q: {query[:120]}[/dim]",
            title="[bold yellow]✍️  WRITE — Interaction Saved to Memory[/bold yellow]",
            border_style="yellow",
        ))

    def _maybe_compress(account_id: str, query_count: int) -> str:
        """COMPRESS: Every 3 queries, summarise all interactions and persist."""
        if query_count > 0 and query_count % 3 == 0:
            memory = _get_account_memory(account_id)
            prior_summary = memory.get("summary", "")
            interactions = memory.get("interactions", [])
            interaction_text = "\n".join(
                f"Q: {i['query']}\nA: {i['response']}" for i in interactions
            )
            summary_response = llm.invoke([
                SystemMessage(content=SUPERVISOR_COMPRESS_PROMPT),
                HumanMessage(content=(
                    f"Account: {account_id}\n"
                    f"Prior summary:\n{prior_summary}\n\n"
                    f"Recent interactions:\n{interaction_text}"
                )),
            ])
            new_summary = summary_response.content
            namespace = ("supervisor_sessions", account_id)
            store.put(namespace, "context", {"summary": new_summary, "interactions": []})
            console.print(Panel(
                f"[italic]{new_summary}[/italic]",
                title=f"[bold green]🗜️  Memory Compressed for [{account_id}][/bold green]",
                border_style="green",
            ))
            return new_summary
        return ""

    def supervisor_node(state: SupervisorState) -> dict:
        # WRITE: determine and persist current account for this query
        account_id = state.get("current_account") or _extract_account(state["messages"])

        # SELECT: inject per-account memory into supervisor prompt
        memory = _get_account_memory(account_id)
        prior_summary = memory.get("summary", "")
        system_content = SUPERVISOR_PROMPT
        if prior_summary:
            system_content += f"\n\n[Prior session memory for account {account_id}]\n{prior_summary}"
        if state.get("session_summary"):
            system_content += f"\n\n[Current session summary for account {account_id}]\n{state['session_summary']}"

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

        # Show SELECT signal if memory is being injected
        if prior_summary:
            console.print(Panel(
                f"[dim]Injecting prior memory for [{account_id}] into supervisor context.[/dim]\n"
                f"[italic]{prior_summary[:300]}{'...' if len(prior_summary) > 300 else ''}[/italic]",
                title="[bold cyan]🔍 SELECT — Prior Memory Injected[/bold cyan]",
                border_style="cyan",
            ))

        response = supervisor_llm.invoke(
            [SystemMessage(content=system_content)] + state["messages"]
        )
        tool_call = response.tool_calls[0]["args"] if response.tool_calls else {"agent": "FINISH", "reason": "done"}

        # Show ISOLATE signal — which specialist was selected and why
        agent_selected = tool_call['agent']
        console.print(Panel(
            f"Routing to [bold]{agent_selected}[/bold]\nReason: [italic]{tool_call.get('reason', '')}[/italic]",
            title="[bold magenta]🔀 ISOLATE — Supervisor Routing Decision[/bold magenta]",
            border_style="magenta",
        ))

        return {
            "next_agent": agent_selected,
            "current_account": account_id,
            "query_count": state.get("query_count", 0) + 1,
        }

    def route_to_agent(state: SupervisorState) -> str:
        agent = state.get("next_agent", "FINISH")
        return agent if agent != "FINISH" else END

    def _human_messages_with_context(state: SupervisorState) -> list:
        """Pass only HumanMessages to sub-agents + inject account memory as SystemMessage."""
        account_id = state.get("current_account", "anonymous")
        memory = _get_account_memory(account_id)
        prior_summary = memory.get("summary", "")
        human_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        if prior_summary:
            console.print(Panel(
                f"[dim]Injecting stored memory for [{account_id}] into sub-agent context.[/dim]\n"
                f"[italic]{prior_summary[:300]}{'...' if len(prior_summary) > 300 else ''}[/italic]",
                title="[bold cyan]🔍 SELECT — Memory Injected into Sub-Agent[/bold cyan]",
                border_style="cyan",
            ))
            return [
                SystemMessage(content=f"[Prior context for account {account_id}]\n{prior_summary}")
            ] + human_msgs
        return human_msgs

    def _last_human_query(state: SupervisorState) -> str:
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                return str(msg.content)
        return ""

    def _print_agent_internals(agent_name: str, messages: list):
        """Print tool calls and tool results from inside a sub-agent's execution."""
        internal = [m for m in messages if not isinstance(m, HumanMessage)]
        if not internal:
            return
        # Show header so it's clear these are sub-agent internals
        console.print(f"[dim]  ┌─ {agent_name} internal steps ({'→'.join(type(m).__name__ for m in internal)})[/dim]")
        for msg in internal[:-1]:  # all but final response (printed separately)
            if isinstance(msg, AIMessage):
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        console.print(Panel(
                            f"Tool: [bold yellow]{tc['name']}[/bold yellow]\nArgs: {tc['args']}",
                            title=f"[bold yellow]🔧 {agent_name} → Tool Call[/bold yellow]",
                            border_style="yellow",
                        ))
            elif isinstance(msg, ToolMessage):
                console.print(Panel(
                    str(msg.content)[:600] + ("..." if len(str(msg.content)) > 600 else ""),
                    title=f"[bold magenta]⚙️  {agent_name} → Tool Result[/bold magenta]",
                    border_style="magenta",
                ))

    def run_fraud_agent(state: SupervisorState) -> dict:
        result = fraud_agent.invoke({"messages": _human_messages_with_context(state)})
        response_text = str(result["messages"][-1].content) if result["messages"] else ""
        account_id = state.get("current_account", "anonymous")
        _print_agent_internals("fraud_agent", result["messages"])
        _save_interaction(account_id, _last_human_query(state), response_text)
        new_summary = _maybe_compress(account_id, state.get("query_count", 0))
        updates = {"messages": result["messages"][-1:], "next_agent": "FINISH"}
        if new_summary:
            updates["session_summary"] = new_summary
        return updates

    def run_loan_agent(state: SupervisorState) -> dict:
        result = loan_agent.invoke({"messages": _human_messages_with_context(state)})
        response_text = str(result["messages"][-1].content) if result["messages"] else ""
        account_id = state.get("current_account", "anonymous")
        _print_agent_internals("loan_agent", result["messages"])
        _save_interaction(account_id, _last_human_query(state), response_text)
        new_summary = _maybe_compress(account_id, state.get("query_count", 0))
        updates = {"messages": result["messages"][-1:], "next_agent": "FINISH"}
        if new_summary:
            updates["session_summary"] = new_summary
        return updates

    def run_support_agent(state: SupervisorState) -> dict:
        result = support_agent.invoke({"messages": _human_messages_with_context(state)})
        response_text = str(result["messages"][-1].content) if result["messages"] else ""
        account_id = state.get("current_account", "anonymous")
        _print_agent_internals("support_agent", result["messages"])
        _save_interaction(account_id, _last_human_query(state), response_text)
        new_summary = _maybe_compress(account_id, state.get("query_count", 0))
        updates = {"messages": result["messages"][-1:], "next_agent": "FINISH"}
        if new_summary:
            updates["session_summary"] = new_summary
        return updates

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
    ("ACC001", "What  UPI daily limits? are theI want to transfer Rs.90,000 to a new contact."),
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


        # Always set current_account from the demo tuple
        current_state["current_account"] = account_id
        # Append account_id to the message content for tool context
        user_message = f"[Account: {account_id}] {query}"
        current_state["messages"] = [HumanMessage(content=user_message)]

        result = graph.invoke(current_state, config=config)
        print_messages(result["messages"][-3:])

        # Carry over mutable scratchpad fields between turns
        current_state["fraud_flags"]      = result.get("fraud_flags", [])
        current_state["loan_context"]     = result.get("loan_context", {})
        current_state["session_summary"]  = result.get("session_summary", "")
        current_state["interaction_count"] = result.get("interaction_count", 0)


def run_supervisor_demo():
    """Run the supervisor multi-agent demo showcasing ISOLATE + WRITE + COMPRESS strategies."""
    section("SUPERVISOR MULTI-AGENT DEMO  (ISOLATE · WRITE · COMPRESS · SELECT)")
    console.print(
        "[bold]Supervisor routes each query to the right specialist agent "
        "(fraud / loan / support). Per-account memory is stored and compressed across queries.[/bold]\n"
    )

    llm = build_llm()
    # WRITE: shared store, keyed per account — different accounts never share memory
    store = InMemoryStore()
    supervisor_graph = build_supervisor_graph(llm, store)

    # Track per-account query counts and session summaries across queries
    account_query_counts: dict = {}
    account_summaries: dict = {}

    for account_id, query in SUPERVISOR_DEMO_QUERIES:
        section(f"Supervisor Query  [{account_id}]")
        tagged_query = f"[Account: {account_id}] {query}"
        console.print(Panel(tagged_query, title="[bold green]🧑 User[/bold green]", border_style="green"))

        account_query_counts[account_id] = account_query_counts.get(account_id, 0)
        result = supervisor_graph.invoke({
            "messages": [HumanMessage(content=tagged_query)],
            "current_account": account_id,
            "query_count": account_query_counts[account_id],
            "session_summary": account_summaries.get(account_id, ""),
            "next_agent": "",
        })
        print_messages(result["messages"][-3:])

        # Carry over per-account state for next query
        account_query_counts[account_id] = result.get("query_count", account_query_counts[account_id])
        if result.get("session_summary"):
            account_summaries[account_id] = result["session_summary"]


# ═══════════════════════════════════════════════════════════════════════════════
# 6. INTERACTIVE CHAT
# ═══════════════════════════════════════════════════════════════════════════════

def run_chat(mode: str = "supervisor"):
    """
    Interactive chat loop — type queries and get live responses.
    Supports 'single' (single-agent) and 'supervisor' (multi-agent) modes.
    Type 'exit' or 'quit' to end the session.
    """
    section(f"INTERACTIVE CHAT  (mode: {mode})")
    console.print(
        "[bold]Type your banking query and press Enter. "
        "Type [red]exit[/red] or [red]quit[/red] to end the session.[/bold]\n"
    )

    if mode == "supervisor":
        llm = build_llm()
        store = InMemoryStore()
        graph = build_supervisor_graph(llm, store)

        account_query_counts: dict = {}
        account_summaries: dict = {}

        while True:
            try:
                raw = console.input("[bold green]You>[/bold green] ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Session ended.[/dim]")
                break

            if raw.lower() in {"exit", "quit", "q"}:
                console.print("[dim]Goodbye![/dim]")
                break
            if not raw:
                continue

            # Extract or prompt for account ID
            import re as _re
            m = _re.search(r"ACC\d{3,}", raw)
            if m:
                account_id = m.group(0)
            else:
                account_id = console.input(
                    "[yellow]Account ID not found in message. Enter account ID (e.g. ACC001): [/yellow]"
                ).strip() or "ACCGUEST"
                raw = f"[Account: {account_id}] {raw}"

            account_query_counts[account_id] = account_query_counts.get(account_id, 0)
            print_memory_status(account_id, store, "supervisor_sessions")
            result = graph.invoke({
                "messages": [HumanMessage(content=raw)],
                "current_account": account_id,
                "query_count": account_query_counts[account_id],
                "session_summary": account_summaries.get(account_id, ""),
                "next_agent": "",
            })
            print_messages(result["messages"][-3:])

            account_query_counts[account_id] = result.get("query_count", account_query_counts[account_id])
            if result.get("session_summary"):
                account_summaries[account_id] = result["session_summary"]

    else:  # single-agent mode
        store = InMemoryStore()
        checkpointer = InMemorySaver()
        graph = build_single_agent_graph(store, checkpointer)
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        current_state: dict = {
            "messages": [],
            "current_account": "",
            "fraud_flags": [],
            "loan_context": {},
            "session_summary": "",
            "interaction_count": 0,
        }

        while True:
            try:
                raw = console.input("[bold green]You>[/bold green] ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Session ended.[/dim]")
                break

            if raw.lower() in {"exit", "quit", "q"}:
                console.print("[dim]Goodbye![/dim]")
                break
            if not raw:
                continue

            import re as _re
            m = _re.search(r"ACC\d{3,}", raw)
            account_id = m.group(0) if m else current_state.get("current_account") or "ACCGUEST"

            current_state["current_account"] = account_id
            current_state["messages"] = [HumanMessage(content=f"[Account: {account_id}] {raw}")]

            print_memory_status(account_id, store, "banking_sessions")
            result = graph.invoke(current_state, config=config)
            # Show all messages except the echoed HumanMessage (tool calls + results + final)
            non_human = [m for m in result["messages"] if not isinstance(m, HumanMessage)]
            print_messages(non_human)

            current_state["fraud_flags"]       = result.get("fraud_flags", [])
            current_state["loan_context"]      = result.get("loan_context", {})
            current_state["session_summary"]   = result.get("session_summary", "")
            current_state["interaction_count"] = result.get("interaction_count", 0)


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

    mode = sys.argv[1] if len(sys.argv) > 1 else "chat"

    if mode == "chat":
        agent_mode = sys.argv[2] if len(sys.argv) > 2 else "supervisor"
        run_chat(agent_mode)
    elif mode == "supervisor":
        run_supervisor_demo()
    elif mode == "single":
        run_single_agent_demo()
    elif mode == "both":
        run_single_agent_demo()
        run_supervisor_demo()
    else:
        run_chat("supervisor")
        run_single_agent_demo()