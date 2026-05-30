"""
Extended Streamlit dashboard: transcript tools, lab, library, settings, analytics helpers.
"""

from __future__ import annotations

import copy
import csv
import io
import json
import random
import zipfile
from datetime import datetime
from typing import Any, Dict, List, Optional

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from models.remote_llm import (
    estimate_anthropic_cost_usd,
    estimate_groq_cost_usd,
    estimate_openai_cost_usd,
)


def init_extended_session_state() -> None:
    defaults: Dict[str, Any] = {
        "run_label": "",
        "run_seed": None,
        "next_speaker_mode": "round_robin",
        "manual_next_speaker": None,
        "muted_agents": [],
        "solo_agent": None,
        "checkpoints": [],
        "emotion_timeline": [],
        "run_logs": [],
        "scenario_draft_json": "",
        "scenario_draft_name": "My custom scenario",
        "win_goals": [],
        "gallery_runs": [],
        "compare_slot_a": 0,
        "compare_slot_b": 0,
        "agent_persona_snapshots": {},
        "prompt_lab_system": "",
        "prompt_lab_style": "",
        "ui_theme": "default",
        "font_scale": 1.0,
        "reduce_motion": False,
        "llm_backend": "local",
        "openai_api_key": "",
        "anthropic_api_key": "",
        "huggingface_token": "",
        "openai_model": "gpt-4o-mini",
        "anthropic_model": "claude-3-5-haiku-20241022",
        "groq_api_key": "",
        "groq_model": "llama-3.3-70b-versatile",
        "openrouter_api_key": "",
        "openrouter_model": "openai/gpt-4o-mini",
        "openrouter_site_url": "",
        "openrouter_site_name": "Multi-Agent Dialogue Simulator",
        "local_model_name": "teknium/OpenHermes-2.5-Mistral-7B",
        "show_transcript_technical": False,
        "transcript_search": "",
        "persona_snapshot_a": None,
        "persona_snapshot_b": None,
        "total_cost_usd_session": 0.0,
        "use_fixed_seed": False,
        "run_seed_value": 42,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def log_run(message: str, level: str = "info") -> None:
    logs: List[Dict[str, Any]] = st.session_state.setdefault("run_logs", [])
    logs.append({"time": datetime.now().isoformat(), "level": level, "message": message})
    st.session_state["run_logs"] = logs[-300:]


def render_accessibility_css() -> None:
    theme = st.session_state.get("ui_theme", "default")
    scale = float(st.session_state.get("font_scale", 1.0))
    reduce = st.session_state.get("reduce_motion", False)
    bg = "#0e1117" if theme == "high_contrast" else "transparent"
    fg = "#ffffff" if theme == "high_contrast" else "inherit"
    motion = "none" if reduce else "auto"
    st.markdown(
        f"""
        <style>
        html, body, [class*="css"]  {{
            font-size: {14 * scale}px !important;
        }}
        .block-container {{
            background: {bg};
            color: {fg};
        }}
        * {{
            scroll-behavior: {motion};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def pick_next_speaker(
    agent_names: List[str],
    turn_count: int,
    history: List[Dict[str, Any]],
    mode: str,
    manual_name: Optional[str],
    muted: List[str],
    solo: Optional[str],
) -> str:
    active = [a for a in agent_names if a not in (muted or [])]
    if not active:
        active = list(agent_names)
    if solo and solo in active:
        return solo
    if mode == "manual" and manual_name and manual_name in active:
        return manual_name
    if mode == "reply_chain" and history:
        last_spk = history[-1].get("speaker")
        others = [a for a in active if a != last_spk]
        if others:
            return others[turn_count % len(others)]
    if mode == "balance" and history:
        counts: Dict[str, int] = {a: 0 for a in active}
        for e in history:
            sp = e.get("speaker")
            if sp in counts and sp not in ("System", "Moderator"):
                counts[sp] += 1
        return min(active, key=lambda x: counts.get(x, 0))
    if mode == "random":
        return random.choice(active)
    return active[turn_count % len(active)]


def append_emotion_timeline(turn: int, agents: Dict[str, Any]) -> None:
    snap: Dict[str, Any] = {"turn": turn}
    for name, agent in agents.items():
        summ = agent.get_agent_summary()
        es = summ.get("emotional_state", {})
        snap[name] = {"emotion": es.get("emotion"), "intensity": es.get("intensity")}
    tl: List[Dict[str, Any]] = st.session_state.setdefault("emotion_timeline", [])
    tl.append(snap)
    st.session_state["emotion_timeline"] = tl[-500:]


def serialize_history(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for e in history:
        row = dict(e)
        ts = row.get("timestamp")
        if hasattr(ts, "isoformat"):
            row["timestamp"] = ts.isoformat()
        out.append(row)
    return out


def deserialize_history(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        r = dict(row)
        if isinstance(r.get("timestamp"), str):
            try:
                r["timestamp"] = datetime.fromisoformat(r["timestamp"])
            except ValueError:
                r["timestamp"] = datetime.now()
        out.append(r)
    return out


def render_run_controls_sidebar() -> None:
    st.session_state.run_label = st.text_input(
        "Run label",
        value=st.session_state.get("run_label", ""),
        key="run_label_sidebar",
    )
    st.session_state.use_fixed_seed = st.checkbox(
        "Use fixed RNG seed",
        value=st.session_state.get("use_fixed_seed", False),
    )
    st.session_state.run_seed_value = st.number_input(
        "Seed",
        min_value=0,
        max_value=2**31 - 1,
        value=int(st.session_state.get("run_seed_value", 42)),
        step=1,
    )


def render_transcript_tab() -> None:
    st.header("Transcript and search")
    hist = st.session_state.get("conversation_history") or []
    q = st.text_input("Search transcript", value=st.session_state.get("transcript_search", ""))
    st.session_state.transcript_search = q
    if not hist:
        st.info("No messages yet.")
        return
    filtered = hist
    if q.strip():
        ql = q.lower()
        filtered = [e for e in hist if ql in (e.get("message") or "").lower() or ql in (e.get("speaker") or "").lower()]
    st.metric("Messages (filtered / total)", f"{len(filtered)} / {len(hist)}")
    show_tech = st.checkbox(
        "Show technical details (latency / tokens)",
        value=st.session_state.get("show_transcript_technical", False),
    )
    st.session_state.show_transcript_technical = show_tech
    for entry in filtered:
        ts = entry.get("timestamp")
        ts_s = ts.strftime("%H:%M:%S") if hasattr(ts, "strftime") else str(ts)
        sp = entry.get("speaker", "")
        msg = entry.get("message", "")
        st.markdown(f"**[{ts_s}] {sp}:** {msg}")
        if show_tech:
            meta = []
            if entry.get("latency_s") is not None:
                meta.append(f"{entry['latency_s']:.2f}s")
            if entry.get("input_tokens") is not None and entry.get("output_tokens") is not None:
                meta.append(f"tokens {entry['input_tokens']}+{entry['output_tokens']}")
            if entry.get("model_id"):
                meta.append(str(entry["model_id"]))
            if meta:
                st.caption(" · ".join(meta))


def render_lab_tab(scenario_manager) -> None:
    st.header("Lab: scenarios, goals, events, memory, prompt tweaks")
    t1, t2, t3, t4 = st.tabs(["Scenario JSON", "Win goals & events", "Memory inspector", "Prompt lab"])

    with t1:
        st.subheader("Register a custom scenario")
        name = st.text_input("Scenario name", value=st.session_state.get("scenario_draft_name", "My scenario"))
        st.session_state.scenario_draft_name = name
        default_json = json.dumps(
            {
                "description": "Custom scenario",
                "context": "Context here",
                "phases": ["Intro", "Debate", "Wrap-up"],
                "suggested_agents": [],
                "initial_prompt": "Begin the discussion.",
            },
            indent=2,
        )
        if not st.session_state.get("scenario_draft_json"):
            st.session_state.scenario_draft_json = default_json
        body = st.text_area("Scenario JSON (same keys as built-ins)", value=st.session_state.scenario_draft_json, height=260)
        st.session_state.scenario_draft_json = body
        if st.button("Register scenario"):
            try:
                data = json.loads(body)
                scenario_manager.register_custom_scenario(name, data)
                log_run(f"Registered scenario: {name}")
                st.success("Scenario registered — select it in the Simulation sidebar.")
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")

    with t2:
        st.subheader("Win conditions (manual checklist)")
        new_goal = st.text_input("Add goal")
        if st.button("Add goal") and new_goal.strip():
            g = st.session_state.setdefault("win_goals", [])
            g.append({"text": new_goal.strip(), "done": False})
        for i, g in enumerate(st.session_state.get("win_goals", [])):
            c1, c2 = st.columns([4, 1])
            with c1:
                st.write(g["text"])
            with c2:
                if st.checkbox("Done", value=g.get("done", False), key=f"wg_{i}"):
                    g["done"] = True
        st.subheader("Inject world events (append as System)")
        ev1, ev2, ev3 = st.columns(3)
        with ev1:
            if st.button("Phone rings"):
                _inject_system("A phone rings loudly in the room.")
        with ev2:
            if st.button("Urgent news"):
                _inject_system("Breaking news interrupts the meeting.")
        with ev3:
            if st.button("Deadline moved"):
                _inject_system("Leadership just moved the deadline up by 24 hours.")

    with t3:
        st.subheader("Per-agent Chroma memory (read / delete)")
        if not st.session_state.get("agents"):
            st.info("Create agents first.")
        else:
            agent_name = st.selectbox("Agent", list(st.session_state.agents.keys()))
            agent = st.session_state.agents[agent_name]
            kind = st.selectbox("Memory kind", ["conversation", "episode", "knowledge", "emotional"])
            entries = agent.memory_system.list_memory_entries(kind, limit=80)
            st.write(f"Showing up to {len(entries)} entries")
            selected_ids = []
            for e in entries[:40]:
                cb = st.checkbox(f"{e['id'][:16]}… {e['document'][:80]}", key=f"mem_{e['id']}")
                if cb:
                    selected_ids.append(e["id"])
            if st.button("Delete selected IDs") and selected_ids:
                n = agent.memory_system.delete_memory_entries(kind, selected_ids)
                log_run(f"Deleted {n} memory rows for {agent_name}/{kind}")
                st.success(f"Deleted {n} entries")

    with t4:
        st.subheader("Prompt lab (applies on next agent turns)")
        st.session_state.prompt_lab_system = st.text_area(
            "Director / system note injected into agent prompt",
            value=st.session_state.get("prompt_lab_system", ""),
        )
        st.session_state.prompt_lab_style = st.text_area(
            "Speaking style note",
            value=st.session_state.get("prompt_lab_style", ""),
        )

    st.divider()
    st.subheader("Persona diff (snapshots when agents were created)")
    if st.session_state.get("agent_persona_snapshots"):
        names = list(st.session_state.agent_persona_snapshots.keys())
        a = st.selectbox("Snapshot A", names, key="pd_a")
        b = st.selectbox("Snapshot B", names, key="pd_b")
        sa = st.session_state.agent_persona_snapshots.get(a)
        sb = st.session_state.agent_persona_snapshots.get(b)
        if sa and sb:
            c1, c2 = st.columns(2)
            with c1:
                st.json(sa)
            with c2:
                st.json(sb)
    else:
        st.caption("Snapshots appear after you click Create Agents in Simulation.")


def _inject_system(text: str) -> None:
    st.session_state.setdefault("conversation_history", []).append(
        {
            "speaker": "System",
            "message": text,
            "timestamp": datetime.now(),
            "turn": st.session_state.get("turn_count", 0),
        }
    )
    log_run(f"Injected system event: {text[:80]}")


def render_library_tab() -> None:
    st.header("Library: checkpoints, gallery, export, compare")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Checkpoints")
        label = st.text_input("Checkpoint label", key="cp_label")
        if st.button("Save checkpoint from current run"):
            cp = {
                "label": label or f"cp-{datetime.now().isoformat()}",
                "history": serialize_history(st.session_state.get("conversation_history", [])),
                "turn": st.session_state.get("turn_count", 0),
                "phase": st.session_state.scenario_manager.current_phase,
                "emotion_timeline": copy.deepcopy(st.session_state.get("emotion_timeline", [])),
            }
            st.session_state.setdefault("checkpoints", []).append(cp)
            log_run(f"Saved checkpoint: {cp['label']}")
            st.success("Checkpoint saved")
        for i, cp in enumerate(st.session_state.get("checkpoints", [])):
            if st.button(f"Restore: {cp['label']}", key=f"r_cp_{i}"):
                st.session_state.conversation_history = deserialize_history(cp["history"])
                st.session_state.turn_count = cp.get("turn", 0)
                st.session_state.scenario_manager.current_phase = cp.get("phase", 0)
                st.session_state.emotion_timeline = cp.get("emotion_timeline", [])
                log_run(f"Restored checkpoint {cp['label']}")
                st.rerun()

    with c2:
        st.subheader("Gallery (save transcript)")
        title = st.text_input("Saved run title", key="gal_title")
        if st.button("Save current run to gallery"):
            entry = {
                "title": title or f"Run {datetime.now().isoformat()}",
                "saved_at": datetime.now().isoformat(),
                "label": st.session_state.get("run_label", ""),
                "history": serialize_history(st.session_state.get("conversation_history", [])),
            }
            st.session_state.setdefault("gallery_runs", []).insert(0, entry)
            log_run(f"Gallery save: {entry['title']}")
            st.success("Saved to gallery")

    st.subheader("Compare two gallery runs")
    gal = st.session_state.get("gallery_runs", [])
    if len(gal) < 2:
        st.info("Save at least two runs to gallery to compare.")
    else:
        a = st.selectbox("Run A", range(len(gal)), format_func=lambda i: gal[i]["title"], key="cmp_a")
        b = st.selectbox("Run B", range(len(gal)), format_func=lambda i: gal[i]["title"], key="cmp_b")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**A**")
            for e in gal[a]["history"][-12:]:
                st.write(f"**{e['speaker']}:** {e['message'][:200]}")
        with c2:
            st.markdown("**B**")
            for e in gal[b]["history"][-12:]:
                st.write(f"**{e['speaker']}:** {e['message'][:200]}")

    st.subheader("Export pack (ZIP: JSON + CSV + Markdown)")
    if st.button("Build export ZIP"):
        buf = io.BytesIO()
        hist = st.session_state.get("conversation_history", [])
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("transcript.json", json.dumps(serialize_history(hist), indent=2))
            md_lines = ["# Transcript\n"]
            for e in hist:
                md_lines.append(f"**{e.get('speaker')}:** {e.get('message')}\n")
            zf.writestr("transcript.md", "\n".join(md_lines))
            sio = io.StringIO()
            w = csv.writer(sio)
            w.writerow(["timestamp", "speaker", "message", "turn"])
            for e in hist:
                ts = e.get("timestamp")
                ts_s = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
                w.writerow([ts_s, e.get("speaker"), e.get("message"), e.get("turn")])
            zf.writestr("transcript.csv", sio.getvalue())
        buf.seek(0)
        st.download_button(
            "Download ZIP",
            data=buf.getvalue(),
            file_name=f"dialogue_export_{datetime.now().strftime('%Y%m%d_%H%M')}.zip",
            mime="application/zip",
        )


def render_settings_tab() -> None:
    st.header("Settings: API keys and model routing")
    st.warning(
        "Keys you type here stay in this browser session only (Streamlit session_state). "
        "For unattended deploys, use environment variables or `.streamlit/secrets.toml` instead."
    )
    opts = ["local", "openai", "anthropic", "groq", "openrouter"]
    cur = st.session_state.get("llm_backend", "local")
    idx = opts.index(cur) if cur in opts else 0
    st.session_state.llm_backend = st.selectbox("LLM backend", opts, index=idx)

    st.session_state.openai_api_key = st.text_input(
        "OpenAI API key",
        value=st.session_state.get("openai_api_key", ""),
        type="password",
        help="Used when backend is OpenAI. Not stored on disk by this app.",
    )
    st.session_state.openai_model = st.text_input("OpenAI model id", value=st.session_state.get("openai_model", "gpt-4o-mini"))

    st.session_state.anthropic_api_key = st.text_input(
        "Anthropic API key",
        value=st.session_state.get("anthropic_api_key", ""),
        type="password",
    )
    st.session_state.anthropic_model = st.text_input(
        "Anthropic model id", value=st.session_state.get("anthropic_model", "claude-3-5-haiku-20241022")
    )

    st.session_state.groq_api_key = st.text_input(
        "Groq API key",
        value=st.session_state.get("groq_api_key", ""),
        type="password",
        help="From https://console.groq.com/keys — also GROQ_API_KEY env var.",
    )
    st.session_state.groq_model = st.text_input(
        "Groq model id",
        value=st.session_state.get("groq_model", "llama-3.3-70b-versatile"),
        help="e.g. llama-3.3-70b-versatile, llama-3.1-8b-instant, mixtral-8x7b-32768",
    )

    st.session_state.openrouter_api_key = st.text_input(
        "OpenRouter API key",
        value=st.session_state.get("openrouter_api_key", ""),
        type="password",
        help="From https://openrouter.ai/keys — also OPENROUTER_API_KEY env var.",
    )
    st.session_state.openrouter_model = st.text_input(
        "OpenRouter model slug",
        value=st.session_state.get("openrouter_model", "openai/gpt-4o-mini"),
        help="e.g. openai/gpt-4o-mini, anthropic/claude-3.5-haiku, meta-llama/llama-3.1-8b-instruct",
    )
    st.session_state.openrouter_site_url = st.text_input(
        "OpenRouter HTTP-Referer (optional)",
        value=st.session_state.get("openrouter_site_url", ""),
    )

    st.session_state.huggingface_token = st.text_input(
        "Hugging Face token (optional, for gated models / higher rate limits)",
        value=st.session_state.get("huggingface_token", ""),
        type="password",
    )
    if st.session_state.huggingface_token.strip():
        import os

        os.environ["HUGGING_FACE_HUB_TOKEN"] = st.session_state.huggingface_token.strip()

    st.session_state.local_model_name = st.text_input(
        "Local Hugging Face model id",
        value=st.session_state.get("local_model_name", "teknium/OpenHermes-2.5-Mistral-7B"),
    )

    st.subheader("Accessibility")
    theme_idx = 1 if st.session_state.get("ui_theme") == "high_contrast" else 0
    st.session_state.ui_theme = st.selectbox("Theme", ["default", "high_contrast"], index=theme_idx)
    st.session_state.font_scale = st.slider("Font scale", 0.85, 1.4, float(st.session_state.get("font_scale", 1.0)))
    st.session_state.reduce_motion = st.checkbox("Reduce motion (continuous mode may still refresh)", value=st.session_state.get("reduce_motion", False))


def render_enhanced_analytics() -> None:
    st.subheader("Relationship graph (trust between agents)")
    agents = st.session_state.get("agents") or {}
    if len(agents) < 2:
        st.caption("Need at least two active agents.")
        return
    names = list(agents.keys())
    G = nx.DiGraph()
    for n in agents:
        G.add_node(n)
    for a in names:
        summ = agents[a].get_agent_summary()
        rel = summ.get("relationships") or {}
        for b, trust in rel.items():
            if b in agents:
                G.add_edge(a, b, weight=float(trust))
    if len(G.edges) == 0:
        st.caption("No relationship edges recorded yet.")
        return
    pos = nx.spring_layout(G.to_undirected(), seed=42)
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    traces = []
    if edge_x:
        traces.append(
            go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=2, color="#888"), hoverinfo="none")
        )
    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    traces.append(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=list(G.nodes()),
            textposition="top center",
            marker=dict(size=20, color="#636efa"),
            hovertext=[f"{n}: {agents[n].get_agent_summary()['emotional_state']}" for n in G.nodes()],
            hoverinfo="text",
        )
    )
    fig = go.Figure(traces)
    fig.update_layout(showlegend=False, height=420, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Emotion over time")
    tl = st.session_state.get("emotion_timeline") or []
    if not tl:
        st.caption("Timeline fills as turns complete.")
    else:
        rows = []
        for row in tl:
            turn = row.get("turn")
            for k, v in row.items():
                if k == "turn":
                    continue
                if isinstance(v, dict):
                    rows.append({"turn": turn, "agent": k, "emotion": v.get("emotion"), "intensity": v.get("intensity")})
        if rows:
            df = pd.DataFrame(rows)
            fig2 = go.Figure()
            for agent in df["agent"].unique():
                sub = df[df["agent"] == agent]
                fig2.add_trace(go.Scatter(x=sub["turn"], y=sub["intensity"], mode="lines+markers", name=agent))
            fig2.update_layout(title="Emotional intensity by turn", height=400)
            st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Simple topic / sentiment buckets")
    hist = [e for e in st.session_state.get("conversation_history", []) if e.get("speaker") not in ("System", "Moderator")]
    pos_w = {"good", "great", "yes", "agree", "hope", "thanks", "excellent"}
    neg_w = {"no", "bad", "wrong", "never", "angry", "worry", "fail"}
    pos_c = neg_c = 0
    for e in hist:
        words = set((e.get("message") or "").lower().split())
        pos_c += len(words & pos_w)
        neg_c += len(words & neg_w)
    st.write(f"Positive keyword hits: **{pos_c}**, negative: **{neg_c}** (very rough heuristic).")

    st.subheader("Fast-reply / overlap heuristic")
    fast = 0
    for i in range(2, len(hist)):
        prev_len = len(hist[i - 1].get("message") or "")
        cur = hist[i]
        if cur.get("speaker") == hist[i - 1].get("speaker"):
            continue
        if prev_len > 120 and len(cur.get("message") or "") < 40:
            fast += 1
    st.metric("Short replies after long neighbour message", fast)


def render_run_logs_drawer() -> None:
    with st.expander("Run log / recent errors", expanded=False):
        for row in reversed((st.session_state.get("run_logs") or [])[-40:]):
            st.text(f"[{row['time']}] ({row['level']}) {row['message']}")


def render_cost_sidebar_note() -> None:
    mm = st.session_state.get("model_manager")
    if not mm:
        return
    info = mm.get_model_info()
    backend = info.get("backend") or st.session_state.get("llm_backend", "local")
    if backend == "openai":
        st.caption("OpenAI pricing is estimated from last response token counts when available.")
    elif backend == "anthropic":
        st.caption("Anthropic pricing is estimated from last response token counts when available.")
    elif backend == "groq":
        st.caption("Groq pricing is estimated from last response token counts when available.")


def add_cost_from_last_response() -> None:
    mm = st.session_state.get("model_manager")
    if not mm or not hasattr(mm, "last_stats"):
        return
    stats = getattr(mm, "last_stats", {}) or {}
    inp = int(stats.get("input_tokens") or 0)
    out = int(stats.get("output_tokens") or 0)
    backend = st.session_state.get("llm_backend", "local")
    cost = None
    if backend == "openai":
        cost = estimate_openai_cost_usd(st.session_state.get("openai_model", ""), inp, out)
    elif backend == "anthropic":
        cost = estimate_anthropic_cost_usd(st.session_state.get("anthropic_model", ""), inp, out)
    elif backend == "groq":
        cost = estimate_groq_cost_usd(st.session_state.get("groq_model", ""), inp, out)
    if cost is not None:
        st.session_state.total_cost_usd_session = float(st.session_state.get("total_cost_usd_session", 0.0)) + cost
