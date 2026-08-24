import boto3

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

MODEL_ID = "amazon.nova-lite-v1:0"

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

# TODO (Task 1): Write the system prompt.
# The assistant should:
# - Help users plan visits to cities
# - NOT answer from memory — always use tools first
# - Base recommendations only on tool results
SYSTEM_PROMPT = """\
You are a travel planning assistant. Your task is to help users plan visits to cities by providing information about weather conditions and top-rated attractions.
You must not answer questions from memory or personal knowledge. Instead, you should always use the provided tools to retrieve information. Base all recommendations and responses solely on the results obtained from these tools.
...\
"""

# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------
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
            {"name": "Covent Garden",         "type": "outdoor/indoor",  "family_friendly": True,  "avg_visit_hours": 1.0},
            {"name": "The Comedy Store",      "type": "indoor",          "family_friendly": False, "avg_visit_hours": 2.0},
            {"name": "Soho Nightlife",        "type": "outdoor/indoor",  "family_friendly": False, "avg_visit_hours": 3.0},
            {"name": "Shoreditch Bar Crawl",  "type": "outdoor/indoor",  "family_friendly": False, "avg_visit_hours": 4.0},
        ],
    }
}

# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "toolSpec": {
            "name": "get_weather",
            "description": "Returns current weather conditions and forecast for a given city and date.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        # TODO (Task 2): Define the two input properties — city and date
                        "city": { "type": "string", "description": "The city for which to retrieve weather information" },
                        "date": { "type": "string", "description": "The date for which to retrieve weather information" },
                    },
                    "required": [
                        # TODO (Task 2): List the required fields
                        "city",
                        "date"
                    ],
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
                        # TODO (Task 2): Define the one input property — city
                        "city": { "type": "string", "description": "The city for which to retrieve attraction information" },
                    },
                    "required": [
                        # TODO (Task 2): List the required fields
                        "city"
                    ],
                }
            },
        }
    },
]

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------
def get_weather(city: str, date: str) -> dict:
    # TODO (Task 3): Look up (city.lower(), date) in WEATHER_DATA.
    # Return the matching dict, or {"city": city, "date": date, "condition": "No data available"} if not found.
    return WEATHER_DATA.get((city.lower(), date), {"city": city, "date": date, "condition": "No data available"})
    pass


def get_top_attractions(city: str) -> dict:
    # TODO (Task 3): Look up city.lower() in ATTRACTIONS_DATA.
    # Return the matching dict, or {"city": city, "attractions": []} if not found.
    return ATTRACTIONS_DATA.get(city.lower(), {"city": city, "attractions": []})
    pass


def execute_tool(name: str, tool_input: dict) -> dict:
    if name == "get_weather":
        return get_weather(tool_input["city"], tool_input["date"])
    elif name == "get_top_attractions":
        return get_top_attractions(tool_input["city"])
    return {"error": f"Unknown tool: {name}"}


# ---------------------------------------------------------------------------
# Converse loop
# ---------------------------------------------------------------------------
def run_chat() -> None:
    messages = []

    print("Travel Planner")
    print("=" * 40)
    print("Ask me to help plan your visit to a city.\n")

    try:
        user_input = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        return

    if not user_input:
        print("No input provided.")
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


if __name__ == "__main__":
    run_chat()
