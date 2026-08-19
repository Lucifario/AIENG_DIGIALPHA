import asyncio
import uuid
import pandas as pd
import plotly.express as px
import streamlit as st
from assis import travel_graph
st.set_page_config(page_title="Multi-Modal Travel Assistant", page_icon="✈️", layout="wide")
st.title("Multi-Modal Travel Assistant")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "turns" not in st.session_state:
    st.session_state.turns = []
if "checkpoint_override" not in st.session_state:
    st.session_state.checkpoint_override = None

def rung(query: str):
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    if st.session_state.checkpoint_override is not None:
        config = st.session_state.checkpoint_override
        st.session_state.checkpoint_override = None
    input_payload = {
        "user_query": query,
    }
    return asyncio.run(travel_graph.ainvoke(input_payload, config=config))

with st.sidebar:
    st.header("Indexed Vector Store Cities")
    st.markdown("- **Tokyo**\n- **Paris**\n- **New York**")
    st.info("Any other city (e.g. Kyoto, Snohomish) dynamically triggers the web search route.")
    st.divider()
    st.header("Time Travel")
    st.caption("Rewind to an earlier point in this conversation and branch from there.")
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    history = list(travel_graph.get_state_history(config))
    checkpoints_with_city = [
        h for h in history if h.values.get("target_city")
    ]
    if not checkpoints_with_city:
        st.caption("No checkpoints yet — ask about a city first.")
    else:
        for h in checkpoints_with_city[:8]:
            city = h.values.get("target_city", "?")
            route = h.values.get("route_used", "")
            label = f"{city} ({route or 'follow-up'})"
            if st.button(f"↩︎ Rewind to: {label}", key=h.config["configurable"]["checkpoint_id"]):
                st.session_state.checkpoint_override = h.config
                st.session_state.pending_rewind_label = label
                st.rerun()
    if st.button("New conversation"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.turns = []
        st.session_state.checkpoint_override = None
        st.rerun()

placeholder = "e.g. 'Tell me about Tokyo' — then try 'What about next week?'"
query = st.text_input("Where would you like to explore?", placeholder=placeholder)
if st.session_state.get("pending_rewind_label"):
    st.success(f"Rewound to checkpoint: **{st.session_state.pending_rewind_label}**. Ask your next question.")
    st.session_state.pending_rewind_label = None
if st.button("Search", type="primary") and query:
    with st.spinner("Running the agent graph (routing → parallel fetch → structuring)..."):
        result = rung(query)
        output = result.get("final_output", {})
        route_used = result.get("route_used") or ("Follow-up (reused context)" if result.get("is_followup") else "—")
        st.session_state.turns.append({"query": query, "output": output, "route": route_used})

for turn in reversed(st.session_state.turns):
    output = turn["output"]
    st.markdown("---")
    st.success(f"Query: **{turn['query']}** | Target City: **{output.get('city_name', '—')}** | Route: **{turn['route']}**")
    st.subheader("City Overview")
    st.write(output.get("city_summary", "No summary available."))
    st.subheader("Temperature Forecast (°C)")
    weather_data = output.get("weather_forecast", [])
    if weather_data:
        df = pd.DataFrame(weather_data)
        if "condition" in df.columns:
            fig = px.line(df, x="day", y="temp_c", markers=True, hover_data=["condition"])
            fig.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(df.set_index("day"))
    else:
        st.warning("Weather forecast data unavailable.")
    st.subheader("Gallery")
    images = output.get("image_urls", [])
    if images:
        cols = st.columns(len(images))
        for col, img_url in zip(cols, images):
            with col:
                st.image(img_url, use_container_width=True)
    else:
        st.warning("No images available for this location.")