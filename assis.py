import json
import asyncio
import random
from typing import List, Dict, Any, TypedDict, Annotated
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
load_dotenv()

class WeatherDay(BaseModel):
    day: str = Field(description="Day name or date (e.g. Day 1, Mon)")
    temp_c: float = Field(description="Temperature in Celsius")
    condition: str = Field(description="Short weather condition, e.g. 'Sunny'")

class TravelAssistantResponse(BaseModel):
    city_name: str = Field(description="Name of the city")
    city_summary: str = Field(description="Structured, well-written summary of the city")
    weather_forecast: List[WeatherDay] = Field(description="5 to 7 day forecast")
    image_urls: List[str] = Field(description="High-quality image URLs")

KNOWN_CITIES_DOCS = [
    Document(
        page_content=(
            "Tokyo is the bustling capital of Japan, renowned for historic temples, "
            "modern skyscrapers, vibrant food culture, and world-class public transit."
        ),
        metadata={"city": "tokyo"},
    ),
    Document(
        page_content=(
            "Paris is France's capital, known for art museums like the Louvre, iconic "
            "architecture like the Eiffel Tower, high fashion, and culinary excellence."
        ),
        metadata={"city": "paris"},
    ),
    Document(
        page_content=(
            "New York City is a major global hub in the USA, famous for Times Square, "
            "Central Park, Broadway theater, and diverse cultural neighborhoods."
        ),
        metadata={"city": "new york"},
    ),
]
_embeddings = None
_vector_store = None

def getvec() -> Chroma:
    global _embeddings, _vector_store
    if _vector_store is None:
        _embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        _vector_store = Chroma.from_documents(
            documents=KNOWN_CITIES_DOCS,
            embedding=_embeddings,
            collection_name="city_knowledge",
        )
    return _vector_store

def get_llm() -> ChatOllama:
    return ChatOllama(model="llama3.1", temperature=0)

_CONDITIONS = ["Sunny", "Partly Cloudy", "Cloudy", "Light Rain", "Clear"]

async def mock_weather_api(city: str, date_range: str = "this week") -> List[Dict[str, Any]]:
    await asyncio.sleep(1.0)
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    if "next" in date_range.lower():
        days = [f"Next {d}" for d in days[:5]]
    base_temp = 22.0 if "tokyo" in city.lower() else (18.0 if "paris" in city.lower() else 20.0)
    rng = random.Random(city.lower() + date_range.lower())
    return [
        {
            "day": day,
            "temp_c": round(base_temp + (i * 1.1) - (i % 2 * 1.8) + rng.uniform(-1, 1), 1),
            "condition": rng.choice(_CONDITIONS),
        }
        for i, day in enumerate(days)
    ]

async def mock_image_search_api(city: str) -> List[str]:
    await asyncio.sleep(1.0)
    if "tokyo" in city.lower():
        seeds = ["1540959733332-e3427110c62f", "1503899036084-c55cdd92da26", "1536098561742-ca998e48cbcc"]
    elif "paris" in city.lower():
        seeds = ["1499856871958-5b9627545d1a", "1431274172761-fce41d57ce8e", "1511739001489-186b5b5c9071"]
    else:
        seeds = ["1493976040374-85c8e12f0c0e", "1534447677768-be436bb09401", "1464822759023-fed622ff2c3b"]
    return [f"https://images.unsplash.com/photo-{s}?auto=format&fit=crop&w=800&q=80" for s in seeds]

_TOOL_IMPLEMENTATIONS = {
    "get_weather_forecast": mock_weather_api,
    "get_city_images": lambda city, **_: mock_image_search_api(city),
}

_WEATHER_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_weather_forecast",
        "description": "Get a multi-day weather forecast for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "date_range": {"type": "string", "description": "e.g. 'this week' or 'next week'"},
            },
            "required": ["city"],
        },
    },
}

_IMAGE_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_city_images",
        "description": "Get high-quality image URLs for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}

async def _execute_tool_calls_manually(llm_response: AIMessage) -> List[ToolMessage]:
    tool_messages: List[ToolMessage] = []
    calls = getattr(llm_response, "tool_calls", None) or []
    for call in calls:
        name = call["name"]
        args = call.get("args", {})
        call_id = call.get("id", str(random.randint(1000, 9999)))
        impl = _TOOL_IMPLEMENTATIONS.get(name)
        if impl is None:
            result_payload = {"error": f"Unknown tool '{name}'"}
        else:
            try:
                result = await impl(**args)
                result_payload = result
            except Exception as exc:
                result_payload = {"error": str(exc)}
        tool_messages.append(
            ToolMessage(
                content=json.dumps(result_payload),
                tool_call_id=call_id,
                name=name,
            )
        )
    return tool_messages

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_query: str
    target_city: str
    date_range: str
    is_in_vectorstore: bool
    is_followup: bool
    context: str
    weather: List[Dict[str, Any]]
    images: List[str]
    final_output: Dict[str, Any]
    route_used: str

def router_node(state: AgentState) -> Dict[str, Any]:
    llm = get_llm()
    query = state["user_query"]
    prev_city = state.get("target_city", "")

    classify_prompt = f"""
You are routing a travel assistant request.
Previous target city on record: "{prev_city or "none"}"
New user message: "{query}"

Decide:
1. is_followup: true ONLY if the new message is a refinement of the SAME city already on record (e.g. asking about a different time range like "what about next week", "and tomorrow?") and does not name a new city.
2. city: the city the response should now be about. If is_followup is true, keep the previous target city.
3. date_range: "this week" or "next week" based on what's being asked (default "this week").

Respond with ONLY compact JSON: {{"is_followup": bool, "city": str, "date_range": str}}
"""
    raw = llm.invoke(classify_prompt).content.strip()
    import re
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        raw = match.group(0)
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = {"is_followup": False, "city": query, "date_range": "this week"}
    city = parsed.get("city") or prev_city or query
    is_followup = bool(parsed.get("is_followup")) and bool(prev_city)
    date_range = parsed.get("date_range", "this week")
    is_known = False
    if not is_followup:
        results = getvec().similarity_search_with_score(city, k=1)
        if results:
            doc, score = results[0]
            if score < 0.6 or doc.metadata.get("city", "").lower() in city.lower():
                is_known = True
    else:
        is_known = state.get("is_in_vectorstore", False)
    return {
        "target_city": city,
        "is_in_vectorstore": is_known,
        "is_followup": is_followup,
        "date_range": date_range,
        "messages": [HumanMessage(content=query)],
    }

def vectorstore_node(state: AgentState) -> Dict[str, Any]:
    docs = getvec().similarity_search(state["target_city"], k=1)
    context = docs[0].page_content if docs else "No internal details found."
    return {"context": f"[Retrieved from VectorStore]: {context}", "route_used": "Vector Store"}

def web_search_node(state: AgentState) -> Dict[str, Any]:
    city = state["target_city"]
    context = f"[Retrieved from Web Search]: {city} is a renowned global destination with rich culture and scenic sights."
    return {"context": context, "route_used": "Web Search"}

async def weather_tool_node(state: AgentState) -> Dict[str, Any]:
    llm = get_llm()
    bound = llm.bind_tools([_WEATHER_TOOL_SCHEMA])
    prompt = f"Call get_weather_forecast for city='{state['target_city']}' and date_range='{state.get('date_range', 'this week')}'."
    response = await bound.ainvoke(prompt)
    tool_messages = await _execute_tool_calls_manually(response)
    weather_payload: List[Dict[str, Any]] = []
    for tm in tool_messages:
        if tm.name == "get_weather_forecast":
            data = json.loads(tm.content)
            if isinstance(data, list):
                weather_payload = data
    if not weather_payload:
        weather_payload = await mock_weather_api(state["target_city"], state.get("date_range", "this week"))
    return {"weather": weather_payload, "messages": [response, *tool_messages]}

async def image_tool_node(state: AgentState) -> Dict[str, Any]:
    llm = get_llm()
    bound = llm.bind_tools([_IMAGE_TOOL_SCHEMA])
    prompt = f"Call get_city_images for city='{state['target_city']}'."
    response = await bound.ainvoke(prompt)
    tool_messages = await _execute_tool_calls_manually(response)
    images_payload: List[str] = []
    for tm in tool_messages:
        if tm.name == "get_city_images":
            data = json.loads(tm.content)
            if isinstance(data, list):
                images_payload = data
    if not images_payload:
        images_payload = await mock_image_search_api(state["target_city"])
    return {"images": images_payload, "messages": [response, *tool_messages]}

def structured_output_node(state: AgentState) -> Dict[str, Any]:
    llm = get_llm()
    structured_llm = llm.with_structured_output(TravelAssistantResponse)
    prompt = f"""
City: {state['target_city']}
Information Context: {state['context']}
Weather data already fetched: {json.dumps(state.get('weather', []))}
Image URLs already fetched: {json.dumps(state.get('images', []))}
Generate a short, informative travel summary of the city using the context above.
"""
    summary_obj: TravelAssistantResponse = structured_llm.invoke(prompt)
    summary_text = summary_obj.city_summary
    final_payload = {
        "city_name": state["target_city"],
        "city_summary": summary_text,
        "weather_forecast": state.get("weather", []),
        "image_urls": state.get("images", []),
    }
    return {
        "final_output": final_payload,
        "messages": [AIMessage(content=summary_text)],
    }

def route_after_router(state: AgentState) -> str:
    if state.get("is_followup"):
        return "weather_only"
    if state.get("is_in_vectorstore"):
        return "vectorstore_node"
    return "web_search_node"

builder = StateGraph(AgentState)
builder.add_node("router_node", router_node)
builder.add_node("vectorstore_node", vectorstore_node)
builder.add_node("web_search_node", web_search_node)
builder.add_node("weather_tool_node", weather_tool_node)
builder.add_node("image_tool_node", image_tool_node)
builder.add_node("structured_output_node", structured_output_node)
builder.add_edge(START, "router_node")
builder.add_conditional_edges(
    "router_node",
    route_after_router,
    {
        "vectorstore_node": "vectorstore_node",
        "web_search_node": "web_search_node",
        "weather_only": "weather_tool_node",
    },
)

builder.add_edge("vectorstore_node", "weather_tool_node")
builder.add_edge("vectorstore_node", "image_tool_node")
builder.add_edge("web_search_node", "weather_tool_node")
builder.add_edge("web_search_node", "image_tool_node")
builder.add_edge("weather_tool_node", "structured_output_node")
builder.add_edge("image_tool_node", "structured_output_node")
builder.add_edge("structured_output_node", END)
checkpointer = MemorySaver()
travel_graph = builder.compile(checkpointer=checkpointer)

def save_graph_image(path: str = "graph.png") -> None:
    png_data = travel_graph.get_graph().draw_mermaid_png()
    with open(path, "wb") as f:
        f.write(png_data)
        print(f"{path} generated successfully.")

if __name__ == "__main__":
    save_graph_image()