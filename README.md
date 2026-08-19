# ✈️ Multi-Modal Travel Assistant

An autonomous, agentic travel assistant built with **LangGraph**, **Streamlit**, and **Ollama**. This system aggregates multi-modal data (text, weather forecasts, and images) and renders it into a rich, interactive UI. It runs 100% locally using open-source models.

## 🧠 Architecture & Distinctions

This project successfully implements the core requirements alongside all three "Extreme" distinction criteria:

*   **Intelligent Routing (The "Switch"):** Dynamically decides between fetching internal knowledge from a pre-populated **ChromaDB** vector store (for indexed cities like Tokyo, Paris, New York) or falling back to a simulated Web Search route for unindexed locations.
*   **Distinction 1 (Manual Transmission):** Bypasses high-level wrappers like `ToolNode`. The agent manually binds raw tool schemas, parses the `tool_calls` payload, executes the mock APIs, and appends `ToolMessage` objects back to the state matrix.
*   **Distinction 2 (Parallel Fan-Out):** Utilizes `asyncio` to fetch weather data and image assets in parallel. The graph schedules both independent tool nodes concurrently in the same superstep, drastically reducing latency.
*   **Distinction 3 (Time Travel & Memory):** Integrates LangGraph's `MemorySaver` checkpointer keyed by `thread_id`. The agent retains contextual memory across turns (e.g., following up a city query with "What about next week?"). The UI also features a Time Travel sidebar, allowing users to rewind the state history and branch the conversation from prior checkpoints.
*   **Structured Output:** Guarantees strict JSON schema compliance using Pydantic, ensuring the Streamlit frontend always receives deterministic data to render the UI components.

## 🛠️ Tech Stack

*   **Orchestration:** LangGraph, LangChain
*   **Frontend:** Streamlit, Pandas, Plotly Express
*   **Local LLM:** Ollama (`llama3.1`)
*   **Vector Database:** ChromaDB
*   **Embeddings:** HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)

## 🚀 Setup & Execution

### 1. Prerequisites
Ensure you have [Ollama](https://ollama.com/) installed and running on your system, then pull the required model:
```bash
ollama pull llama3.1
```

### 2. Install Dependencies
Set up your virtual environment and install the required packages:
```bash
pip install -r requirements.txt
```

### 3. Generate Graph Visualization
Run the backend module to generate the `graph.png` topology file:
```bash
python assis.py
```

### 4. Launch the App
Start the Streamlit server:
```bash
streamlit run app_2.py
```
*(Note: If your Streamlit file is named `app.py`, run `streamlit run app.py` instead).*