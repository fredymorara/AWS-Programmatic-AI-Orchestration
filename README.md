# Local Autonomous Travel Planner - AWS Bedrock Converse API

A ReAct (Reason → Act → Observe) travel planning agent built entirely in local Python, with no managed agent framework in the loop. This project steps *down* the stack from AgentCore Gateways/Harnesses to raw `boto3` and Bedrock's unified `converse` API, to see what the same ReAct pattern looks like when you own the orchestration loop yourself.

**Portfolio Link:** [Programmatic AI Orchestration](https://freddymorara.tech/work/programmatic-ai)

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [`invoke_model` vs. `converse`](#invoke_model-vs-converse)
- [Technical Challenges & Solutions](#technical-challenges--solutions)
- [Tool Engineering & Schemas](#tool-engineering--schemas)
- [Full Implementation](#full-implementation)
- [Verification & Execution Traces](#verification--execution-traces)
- [Tech Stack](#tech-stack)
- [Key Takeaways](#key-takeaways)

---

## Overview

The prior two builds (the AgentCore travel assistant and the restaurant recommendation agent) both used managed AWS orchestration - AgentCore Harnesses, MCP Gateways, and remote Lambda tool targets. This project asks a different question: what does the same ReAct loop look like with **no managed framework at all** - just a Python script, `boto3`, and Bedrock's `converse` API?

**Core requirements:**

- **Zero-hallucination guardrail:** the agent is explicitly forbidden from answering from its own pretrained knowledge - every factual claim has to come from a tool call.
- **Strict tool orchestration:** detect required parameters, call local Python tools (`get_weather`, `get_top_attractions`), and synthesize a recommendation only from their output.
- **Dynamic contextual filtering:** the same attraction list has to be filtered two different ways - weather-aware (rain vs. sun) and demographic-aware (family-friendly vs. adult nightlife) - depending on who's asking and when.

---

## Architecture

Instead of routing through an AgentCore Harness and remote Lambda functions, the entire orchestration loop runs client-side, inside the Python process itself:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Local Python Runtime                            │
│                                                                        │
│   User Prompt ──► [ messages ] ──► Bedrock Converse API                │
│                         ▲                     │                        │
│                         │ (toolResult)        ▼ (stopReason: tool_use) │
│                         │                Tool Dispatcher               │
│                         │                     │                        │
│                         │       ┌─────────────┴─────────────┐          │
│                         │       ▼                           ▼          │
│                         └─ get_weather()           get_top_attractions │
│                                                                        │
│   Final Output ◄── [ messages ] ◄── Bedrock Converse API               │
│                                     (stopReason: end_turn)             │
└────────────────────────────────────────────────────────────────────────┘
```

**Component breakdown:**

- **Inference Engine:** Amazon Nova 2 Lite (`amazon.nova-lite-v1:0`), Amazon Bedrock Runtime, `us-east-1`
- **Client SDK:** Python (3.14 / 3.12), `boto3` v1.42.54, `botocore` v1.42.54
- **Orchestrator:** a `while True` loop that inspects `stopReason` (`tool_use` vs. `end_turn`), dispatches to the matching local Python function, and re-injects the result as a `toolResult` block
- **Local Data Layer:** in-memory dictionaries keyed on normalized composite tuples (`city.lower(), date`), with safe `.get()` fallbacks instead of direct indexing

---

## `invoke_model` vs. `converse`

Building this by hand made the difference between Bedrock's two invocation APIs very concrete:

| Feature / Dimension | `invoke_model` API | `converse` API |
|---|---|---|
| **Payload Format** | Model-specific raw JSON (differs across Nova, Claude, Llama, Mistral) | Standardized, cross-model schema (`system`, `messages`, `inferenceConfig`) |
| **Tool Calling Support** | Manual formatting; developer parses raw strings or model-specific tool markup | Native `toolConfig` schema support |
| **Conversation State** | Manual serialization of prior turns into a proprietary prompt format | Standardized message history (`user`, `assistant`, `toolResult` roles) |
| **Tool Turn Management** | Developer manually tracks tool-call markers in raw output | Structured `stopReason: "tool_use"` and `toolUseId` fields |
| **Portability** | Low - swapping models means rewriting the payload parser | High - swap `modelId` without touching the message schema |

The `converse` API is effectively what made this project tractable as a "small script" rather than a per-model integration exercise.

---

## Technical Challenges & Solutions

### 1. CLI Session Authentication & Credential Isolation

**Problem:** Running the script locally in VS Code failed immediately:

```
botocore.exceptions.NoCredentialsError: Unable to locate credentials
```

**Root cause:** Unlike the browser-based cloud terminal used on prior projects, the local VS Code PowerShell session had no persistent AWS IAM credentials or profile configured.

**Fix:** Injected temporary session credentials directly into the local environment:

```powershell
$env:AWS_ACCESS_KEY_ID="<ACCESS_KEY>"
$env:AWS_SECRET_ACCESS_KEY="<SECRET_KEY>"
$env:AWS_SESSION_TOKEN="<SESSION_TOKEN>"
$env:AWS_DEFAULT_REGION="us-east-1"
```

### 2. Handling Underspecified Prompts & Schema Defaults

**Problem:** With an open-ended prompt like *"I am planning on visiting london soon"*, the model filled the required `date` field in `get_weather` with an unprompted placeholder date (`2023-10-01`) - because the tool schema marked `date` as required, so the model had to supply *something* to make a valid call.

**Fix:** The tool implementation used safe `.get()` lookups instead of direct dictionary indexing:

```python
WEATHER_DATA.get((city.lower(), date), {"city": city, "date": date, "condition": "No data available"})
```

Instead of crashing with a `KeyError`, the tool returned a graceful "no data" observation. The model read that observation, reasoned about the gap in its `<thinking>` output, and adjusted its recommendation instead of failing outright.

### 3. Bidirectional ReAct Message History Flow

**Problem:** The Converse API throws validation exceptions if a tool response payload doesn't exactly mirror the `toolUseId` of the corresponding request.

**Fix:** An accumulator loop extracts each `toolUseId`, wraps the corresponding tool output in a matching `toolResult` container, and appends it back into the conversation history under the `user` role:

```python
tool_results.append({
    "toolResult": {
        "toolUseId": tool_use_id,
        "content": [{"json": result}],
    }
})
```

---

## Tool Engineering & Schemas

### System Prompt

The zero-hallucination guardrail lives entirely in the system prompt:

> *"You are a travel planning assistant. Your task is to help users plan visits to cities by providing information about weather conditions and top-rated attractions. You must not answer questions from memory or personal knowledge. Instead, you should always use the provided tools to retrieve information. Base all recommendations and responses solely on the results obtained from these tools."*

### `toolConfig`

```python
TOOLS = [
    {
        "toolSpec": {
            "name": "get_weather",
            "description": "Returns current weather conditions and forecast for a given city and date.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "The city for which to retrieve weather information"},
                        "date": {"type": "string", "description": "The date for which to retrieve weather information"},
                    },
                    "required": ["city", "date"],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "get_top_attractions",
            "description": "Returns a list of top-rated attractions in a given city.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "The city for which to retrieve attraction information"},
                    },
                    "required": ["city"],
                }
            },
        }
    },
]
```

---

## Full Implementation

```python
import boto3

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
MODEL_ID = "amazon.nova-lite-v1:0"

SYSTEM_PROMPT = """\
You are a travel planning assistant. Your task is to help users plan visits to cities by providing information about weather conditions and top-rated attractions.
You must not answer questions from memory or personal knowledge. Instead, you should always use the provided tools to retrieve information. Base all recommendations and responses solely on the results obtained from these tools.
"""

WEATHER_DATA = {
    ("london", "2026-03-14"): {
        "city": "London",
        "date": "2026-03-14",
        "condition": "Light rain in the morning, clearing to partly cloudy by afternoon",
        "temperature_celsius": 11,
        "wind_mph": 12,
        "recommendation": "Bring a light jacket and umbrella for the morning",
    },
    ("london", "2026-03-15"): {
        "city": "London",
        "date": "2026-03-15",
        "condition": "Clear and sunny throughout the day",
        "temperature_celsius": 14,
        "wind_mph": 8,
        "recommendation": "Great day to spend time outdoors",
    },
}

ATTRACTIONS_DATA = {
    "london": {
        "city": "London",
        "attractions": [
            {"name": "British Museum",        "type": "indoor",          "family_friendly": True, "avg_visit_hours": 2.0},
            {"name": "Tower of London",       "type": "outdoor/indoor",  "family_friendly": True, "avg_visit_hours": 2.5},
            {"name": "Natural History Museum","type": "indoor",          "family_friendly": True, "avg_visit_hours": 2.0},
            {"name": "Hyde Park",             "type": "outdoor",         "family_friendly": True, "avg_visit_hours": 1.5},
            {"name": "Covent Garden",         "type": "outdoor/indoor",  "family_friendly": True, "avg_visit_hours": 1.0},
            {"name": "The Comedy Store",      "type": "indoor",          "family_friendly": False, "avg_visit_hours": 2.0},
            {"name": "Soho Nightlife",        "type": "outdoor/indoor",  "family_friendly": False, "avg_visit_hours": 3.0},
            {"name": "Shoreditch Bar Crawl",  "type": "outdoor/indoor",  "family_friendly": False, "avg_visit_hours": 4.0},
        ],
    }
}

TOOLS = [
    {
        "toolSpec": {
            "name": "get_weather",
            "description": "Returns current weather conditions and forecast for a given city and date.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "The city for which to retrieve weather information"},
                        "date": {"type": "string", "description": "The date for which to retrieve weather information"},
                    },
                    "required": ["city", "date"],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "get_top_attractions",
            "description": "Returns a list of top-rated attractions in a given city.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "The city for which to retrieve attraction information"},
                    },
                    "required": ["city"],
                }
            },
        }
    },
]

def get_weather(city: str, date: str) -> dict:
    return WEATHER_DATA.get((city.lower(), date), {"city": city, "date": date, "condition": "No data available"})

def get_top_attractions(city: str) -> dict:
    return ATTRACTIONS_DATA.get(city.lower(), {"city": city, "attractions": []})

def execute_tool(name: str, tool_input: dict) -> dict:
    if name == "get_weather":
        return get_weather(tool_input["city"], tool_input["date"])
    elif name == "get_top_attractions":
        return get_top_attractions(tool_input["city"])
    return {"error": f"Unknown tool: {name}"}

def run_chat() -> None:
    messages = []
    print("Travel Planner\n" + "=" * 40 + "\nAsk me to help plan your visit to a city.\n")
    user_input = input("You: ").strip()
    if not user_input:
        return
    messages.append({"role": "user", "content": [{"text": user_input}]})

    while True:
        response = bedrock.converse(
            modelId=MODEL_ID,
            system=[{"text": SYSTEM_PROMPT}],
            messages=messages,
            toolConfig={"tools": TOOLS},
        )
        stop_reason = response["stopReason"]
        output_message = response["output"]["message"]
        messages.append(output_message)

        if stop_reason == "end_turn":
            for block in output_message["content"]:
                if "text" in block:
                    print(f"\nAssistant: {block['text']}\n")
            break
        elif stop_reason == "tool_use":
            tool_results = []
            for block in output_message["content"]:
                if "toolUse" in block:
                    tool_name = block["toolUse"]["name"]
                    tool_input = block["toolUse"]["input"]
                    tool_use_id = block["toolUse"]["toolUseId"]
                    print(f"  [tool call] {tool_name}({tool_input})")
                    result = execute_tool(tool_name, tool_input)
                    print(f"  [tool result] {result}")
                    tool_results.append({
                        "toolResult": {
                            "toolUseId": tool_use_id,
                            "content": [{"json": result}],
                        }
                    })
            messages.append({"role": "user", "content": tool_results})
```

---

## Verification & Execution Traces

### Trace 1 - Underspecified Date (Graceful Degradation)

**Input:** *"I am planning on visiting london soon"*

**Tool calls:**
- `get_weather({'date': '2023-10-01', 'city': 'London'})` → `{'condition': 'No data available'}`
- `get_top_attractions({'city': 'London'})` → 8 attractions returned

**Agent output:**

```
Assistant: <thinking>The weather data for London is not available, but I have a list of
top-rated attractions. I should let the User know about the attractions and suggest they
check the weather closer to their travel date.</thinking>

Hi there! Here are some top-rated attractions in London that you might want to visit:
1. British Museum - Indoor, Avg. visit hours: 2, Family-friendly: Yes
...
Please note that the weather data for London is currently not available. I recommend
checking the weather closer to your travel date to plan your activities accordingly.
```

The model correctly reasoned about the missing data rather than inventing a forecast, and adjusted its recommendation to compensate.

### Trace 2 - Rainy Day with Family (Multi-Constraint Filtering)

**Input:** *"I'll be in London on 2026-03-14 with my family."*

**Tool calls:**
- `get_weather({'date': '2026-03-14', 'city': 'London'})` → light rain in the morning, 11°C, "bring a light jacket and umbrella"
- `get_top_attractions({'city': 'London'})` → 8 attractions returned

**Result:** The agent filtered out all three adult-nightlife options and prioritized family-friendly indoor/mixed venues (British Museum, Tower of London, Natural History Museum), explicitly folding in the morning-rain warning and gear recommendation.

### Trace 3 - Adult Night Out (Demographic Inversion)

**Input:** *"I'm in London on 2026-03-14 for a night out with friends. What do you suggest?"*

**Tool calls:** identical tools, same date, called in parallel.

**Agent output:**

```
Assistant: <thinking>Now that I have the weather and top-rated attractions for London on
2026-03-14, I can suggest some activities based on the User's request for a night out
with friends... I will suggest some night-friendly attractions from the list of
top-rated attractions.</thinking>

Here's what you can do for a night out in London on 2026-03-14:
1. The Comedy Store (Indoor comedy venue, adult-focused)
2. Soho Nightlife (Mix of indoor and outdoor venues)
3. Shoreditch Bar Crawl (4-hour tour across popular venues)
```

Same date, same underlying data, same tools - but the filter inverted completely based on stated audience, confirming the agent is reasoning over the data rather than returning a fixed answer per city.

---

## Tech Stack

- **Cloud Provider:** AWS
- **Inference Runtime:** Amazon Bedrock Runtime (`us-east-1`)
- **Foundation Model:** Amazon Nova 2 Lite (`amazon.nova-lite-v1:0`)
- **API:** Bedrock `converse` (unified, cross-model tool-calling API)
- **Client:** Python 3.14 / 3.12, `boto3` v1.42.54, `botocore` v1.42.54
- **Orchestration:** Hand-rolled ReAct `while True` state machine (no managed agent framework)
- **Data Layer:** In-memory dictionaries, composite-tuple keys, safe `.get()` fallbacks

---

## Key Takeaways

- **ReAct logic can live anywhere.** Whether it's orchestrated through an AWS Bedrock AgentCore Gateway over MCP, or a ~100-line Python script calling `boto3` directly, the core pattern is identical: strict tool contracts, an explicit guardrail prompt, and a conversational feedback loop between model and tools.
- **The `converse` API is what makes this tractable as a small script.** It standardizes multi-turn tool calling across Nova, Claude, and Llama, so there's no need to hand-write a model-specific prompt/response parser the way `invoke_model` would require.
- **Defensive tool coding is not optional.** LLMs will extrapolate missing parameters when a schema marks them required - here, a hallucinated placeholder date. Returning a structured "no data" dict instead of raising a `KeyError` turned a potential crash into a handled, reasoned-about edge case.
- **Same data, different lens.** Trace 2 and Trace 3 use the exact same city, date, and attraction list, but produce inverted recommendations purely based on stated audience - good evidence the agent is filtering contextually rather than pattern-matching a canned response.
- **Credential handling changes shape once you leave the browser lab environment.** Every "local dev" pivot across these three projects has needed the same fix: explicitly injecting temporary AWS credentials into the local shell session, since nothing is pre-authenticated outside the managed lab terminal.

