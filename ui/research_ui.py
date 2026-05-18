"""Live research tab — web + news → agent debate fuel."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import streamlit as st

from utils.web_research import (
    ResearchBrief,
    normalize_research_brief,
    parse_research_command,
    run_research,
)


def init_research_session_state() -> None:
    defaults = {
        "research_brief": None,
        "research_topic": "",
        "auto_research_on_start": True,
        "auto_research_no_scenario": True,
        "freeform_topic": "",
        "last_research_error": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    if st.session_state.get("research_brief") is not None:
        st.session_state.research_brief = normalize_research_brief(st.session_state.research_brief)


def get_research_brief() -> Optional[ResearchBrief]:
    b = st.session_state.get("research_brief")
    if b is None:
        return None
    normalized = normalize_research_brief(b)
    st.session_state.research_brief = normalized
    return normalized


def inject_research_into_conversation(brief: ResearchBrief, as_breaking: bool = False) -> None:
    brief = normalize_research_brief(brief) or brief
    prefix = "📡 BREAKING" if as_breaking else "📡 LIVE RESEARCH"
    if brief.article_hit:
        h = brief.article_hit
        msg = (
            f"{prefix}: **{h.title}**\n\n"
            f"{h.snippet[:500]}\n\n"
            f"_Agents: debate this story using the facts above — do not read URLs aloud._"
        )
    else:
        headlines = [h.title for h in (brief.news_hits or brief.web_hits)[:3]]
        hl = "; ".join(headlines) if headlines else "see Research tab"
        msg = (
            f"{prefix} on **{brief.topic}** — agents have live context. "
            f"Related: {hl}."
        )
    st.session_state.setdefault("conversation_history", []).append(
        {
            "speaker": "System",
            "message": msg,
            "timestamp": datetime.now(),
            "turn": st.session_state.get("turn_count", 0),
            "research_injected": True,
        }
    )
    st.session_state.research_brief = brief
    _seed_agent_memories(brief)


def _seed_agent_memories(brief: ResearchBrief) -> None:
    block = brief.to_context_block(max_chars=1500)
    for agent in (st.session_state.get("agents") or {}).values():
        try:
            agent.memory_system.store_knowledge(block, category="live_research", confidence=0.95)
        except Exception:
            pass


def all_agents_research_topic(topic: str) -> Optional[ResearchBrief]:
    """Run web/news research on a plain topic (no URL) and brief every agent."""
    topic = (topic or "").strip()
    if not topic:
        return None
    brief = run_research(topic)
    brief = normalize_research_brief(brief)
    st.session_state.research_brief = brief
    st.session_state.research_topic = topic
    _seed_agent_memories(brief)
    inject_research_into_conversation(brief, as_breaking=True)
    agents = list((st.session_state.get("agents") or {}).keys())
    if agents:
        lead = agents[0]
        agent_research_request(lead, topic, skip_run=True, brief=brief)
    return brief


def agent_research_request(
    agent_name: str,
    topic: str,
    *,
    skip_run: bool = False,
    brief: Optional[ResearchBrief] = None,
) -> Optional[ResearchBrief]:
    """One agent 'researches' a topic and shares findings in chat."""
    topic = (topic or "").strip()
    if not topic or agent_name not in (st.session_state.get("agents") or {}):
        return None
    if not skip_run:
        brief = run_research(topic)
        brief = normalize_research_brief(brief)
        st.session_state.research_brief = brief
        _seed_agent_memories(brief)
    elif brief is not None:
        brief = normalize_research_brief(brief)
    if brief is None:
        return None
    h = brief.article_hit
    if h:
        share = f"I looked this up: **{h.title}** — {h.snippet[:350]}"
    else:
        top = (brief.news_hits or brief.web_hits or brief.wiki_hits)[:1]
        share = (
            f"I researched **{brief.topic}**. "
            + (f"{top[0].title}: {top[0].snippet[:200]}" if top else "Found background context.")
        )
    st.session_state.conversation_history.append(
        {
            "speaker": agent_name,
            "message": share,
            "timestamp": datetime.now(),
            "turn": st.session_state.get("turn_count", 0),
            "is_research_share": True,
        }
    )
    return brief


def handle_moderator_research_command(message: str) -> bool:
    """If moderator says 'research war' etc., agents run live research."""
    topic = parse_research_command(message)
    if not topic:
        return False
    all_agents_research_topic(topic)
    return True


def run_auto_research_for_start(freeform_topic: str = "", force: bool = False) -> bool:
    """Research on sim start — required when no scenario unless brief already loaded."""
    if not force and get_research_brief() and not freeform_topic:
        return False
    topic = (freeform_topic or st.session_state.get("research_topic") or "").strip()
    ctx = st.session_state.scenario_manager.get_current_context()
    if not topic and ctx:
        topic = (ctx.get("scenario_description") or "")[:120]
    if not topic:
        topic = (st.session_state.get("freeform_topic") or "").strip()
    if not topic:
        return False
    brief = normalize_research_brief(run_research(topic))
    st.session_state.research_brief = brief
    st.session_state.research_topic = topic
    inject_research_into_conversation(brief, as_breaking=bool(not ctx))
    return True


def render_research_tab() -> None:
    init_research_session_state()
    st.header("🌐 Live Research")
    st.markdown(
        "Paste a **topic** or a **full article URL** (BBC, Reuters, etc.). "
        "We fetch the article title & summary, then find related news — not random RSS from the URL string."
    )

    c1, c2 = st.columns([3, 1])
    with c1:
        topic = st.text_input(
            "Research topic or article URL",
            value=st.session_state.get("research_topic", ""),
            placeholder="e.g. climate tipping points — or https://www.bbc.com/news/articles/...",
        )
    with c2:
        st.write("")
        st.write("")
        go = st.button("🔍 Research now", type="primary", use_container_width=True)

    include_news = st.checkbox("Include related news headlines", value=True)
    include_web = st.checkbox("Include web search (pip install duckduckgo-search)", value=True)
    st.session_state.auto_research_on_start = st.checkbox(
        "Auto-research when starting simulation (if topic set)",
        value=st.session_state.get("auto_research_on_start", True),
    )
    st.session_state.auto_research_no_scenario = st.checkbox(
        "When **no scenario** is selected, always research on start (uses free-form topic below)",
        value=st.session_state.get("auto_research_no_scenario", True),
    )
    st.session_state.freeform_topic = st.text_input(
        "Free-form debate topic (used when no scenario)",
        value=st.session_state.get("freeform_topic", ""),
        placeholder="e.g. Should we regulate AI in healthcare?",
    )

    if go and topic.strip():
        st.session_state.research_topic = topic.strip()
        with st.spinner("Fetching article / web / news…"):
            try:
                brief = run_research(topic.strip(), include_news=include_news, include_web=include_web)
                st.session_state.research_brief = brief
                st.session_state.last_research_error = ""
                n = len(brief.news_hits) + len(brief.web_hits) + len(brief.wiki_hits)
                extra = " + primary article" if brief.article_hit else ""
                st.success(f"Ready: {brief.topic} ({n} related hits{extra}).")
            except Exception as e:
                st.session_state.last_research_error = str(e)
                st.error(f"Research failed: {e}")

        brief = get_research_brief()
        if brief:
            _render_brief_card(brief)

    else:
        brief = get_research_brief()
        if brief:
            _render_brief_card(brief)

    st.divider()
    st.subheader("Send research into the conversation")
    b1, b2, b3 = st.columns(3)
    with b1:
        brief = get_research_brief()
        if st.button("💬 Inject brief", disabled=brief is None):
            inject_research_into_conversation(brief, as_breaking=False)
            st.success("Injected.")
    with b2:
        brief = get_research_brief()
        if st.button("🚨 Inject breaking", disabled=brief is None):
            inject_research_into_conversation(brief, as_breaking=True)
            st.success("Breaking context added.")
    with b3:
        if st.button("🗑️ Clear research"):
            st.session_state.research_brief = None
            st.session_state.research_topic = ""
            st.rerun()

    st.divider()
    st.subheader("Agents research a topic (no URL needed)")
    agents = list((st.session_state.get("agents") or {}).keys())
    team_topic = st.text_input(
        "Topic for all agents",
        key="team_research_topic",
        placeholder="e.g. war in Ukraine, AI regulation, climate tipping points",
    )
    if agents:
        if st.button("🌐 All agents research this topic now", type="primary") and team_topic.strip():
            with st.spinner("Searching web & news…"):
                all_agents_research_topic(team_topic.strip())
            st.success("Research complete — injected into chat and agent memory.")
        ar_agent = st.selectbox("Or one agent", agents, key="research_agent_pick")
        if st.button("🔍 Single agent researches & shares") and team_topic.strip():
            with st.spinner(f"{ar_agent} is researching…"):
                agent_research_request(ar_agent, team_topic.strip())
            st.success(f"{ar_agent} shared findings.")
        st.caption('Moderator tip: type **"research war"** or **"look up climate change"** in Manual Intervention.')
    else:
        st.caption("Create agents in Simulation first.")

    st.divider()
    st.subheader("During simulation")
    if st.session_state.get("conversation_history"):
        last = [e for e in st.session_state.conversation_history if e.get("speaker") not in ("System", "Moderator")]
        if last and st.button("🔎 Quick research (last message)"):
            from utils.web_research import research_snippet_for_message

            line = last[-1].get("message", "")
            snippet = research_snippet_for_message(line)
            if snippet:
                st.session_state.conversation_history.append(
                    {
                        "speaker": "System",
                        "message": f"📡 QUICK RESEARCH:\n\n{snippet}",
                        "timestamp": datetime.now(),
                        "turn": st.session_state.get("turn_count", 0),
                    }
                )
                st.success("Added.")
            else:
                st.warning("No results.")
    else:
        st.info("Start a conversation first.")


def render_sidebar_agent_research() -> None:
    """Compact agent-research controls in Simulation sidebar."""
    agents = list((st.session_state.get("agents") or {}).keys())
    if not agents:
        return
    st.subheader("Agent research")
    topic = st.text_input(
        "Topic (e.g. war, elections, AI)",
        key="sb_research_topic",
        placeholder="Plain words — agents search the web",
    )
    if st.button("🌐 All agents research", use_container_width=True) and topic.strip():
        with st.spinner("Researching…"):
            all_agents_research_topic(topic.strip())
        st.rerun()
    agent = st.selectbox("One agent shares findings", agents, key="sb_research_agent")
    if st.button("🔍 One agent researches", use_container_width=True) and topic.strip():
        with st.spinner("Researching…"):
            agent_research_request(agent, topic.strip())
        st.rerun()


def _render_brief_card(brief: ResearchBrief) -> None:
    brief = normalize_research_brief(brief) or brief
    st.markdown(f"### {brief.topic}")
    source_url = getattr(brief, "source_url", "") or ""
    if source_url:
        st.caption(f"Source: {source_url}")
    st.caption(f"Updated {brief.fetched_at}")
    with st.expander("Preview — what agents see", expanded=True):
        st.text(brief.to_context_block())
    with st.expander("Sources (links)"):
        st.markdown(brief.sources_markdown())
