"""Reset conversation / agents from the sidebar."""

from __future__ import annotations

import streamlit as st


def clear_conversation(keep_research: bool = True) -> None:
    st.session_state.conversation_history = []
    st.session_state.turn_count = 0
    st.session_state.simulation_running = False
    st.session_state.continuous_mode = False
    st.session_state.emotion_timeline = []
    if not keep_research:
        st.session_state.research_brief = None
        st.session_state.research_topic = ""


def clear_agents() -> None:
    st.session_state.agents = {}
    st.session_state.agent_persona_snapshots = {}
    clear_conversation(keep_research=True)


def render_reset_buttons() -> None:
    st.subheader("Reset")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🗑️ Clear chat", use_container_width=True, help="Wipe transcript; keep agents & research"):
            clear_conversation(keep_research=True)
            st.rerun()
    with c2:
        if st.button("🤖 Clear agents", use_container_width=True, help="Remove all agents and chat"):
            clear_agents()
            st.rerun()
