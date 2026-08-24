# Exercise – Travel Planner

## Overview

In this exercise you'll use the Amazon Bedrock Converse API to build a travel planning assistant. The assistant must call tools to look up weather and attractions before making any recommendation — it must not answer from memory.

**Scenario:** A user is planning a family day in London and asks: *"I'll be in London this Saturday with my family. What should we do?"* Your assistant must call `get_weather` and `get_top_attractions` first, then produce a grounded recommendation based on the results.

---

## What You'll Build

A Python script (`travel_planner.py`) that:

1. Sends the user's question to Claude via the Bedrock Converse API
2. Defines and exposes two tool schemas: `get_weather` and `get_top_attractions`
3. Executes each tool locally when the model invokes it
4. Passes tool results back to the model
5. Prints the final grounded recommendation

---

## Tasks

### Task 1 – Write the System Prompt

Write a `SYSTEM_PROMPT` that instructs the assistant to:

- Help users plan city visits
- Never answer from memory — always call the available tools first
- Base all recommendations on tool results only

---

### Task 2 – Complete the Tool Schemas

Fill in the `properties` and `required` fields for both tools:

**`get_weather`** takes two inputs:

| Field | Type | Description |
|-------|------|-------------|
| `city` | string | The city to get weather for |
| `date` | string | The date in YYYY-MM-DD format |

**`get_top_attractions`** takes one input:

| Field | Type | Description |
|-------|------|-------------|
| `city` | string | The city to get attractions for |

---

### Task 3 – Implement the Tool Functions

Complete `get_weather` and `get_top_attractions`:

- `get_weather(city, date)` — look up `(city.lower(), date)` in `WEATHER_DATA`. Return the matching dict, or `{"city": city, "date": date, "condition": "No data available"}` if not found.
- `get_top_attractions(city)` — look up `city.lower()` in `ATTRACTIONS_DATA`. Return the matching dict, or `{"city": city, "attractions": []}` if not found.


> **Note – mock data scope:** The sample data in `WEATHER_DATA` and `ATTRACTIONS_DATA` only covers London. The tool interface, however, accepts any city string — extending the data to support additional cities requires only adding new entries to those dictionaries.

> **Why use a tool instead of relying on the model's knowledge?** The model may already know popular attractions, but a tool lets you serve *current* information: a newly opened attraction, a venue that is temporarily closed, or a special event happening this weekend. Grounding recommendations in tool results keeps them accurate regardless of what the model was trained on.

---

## Running the Script

```bash
python travel_planner.py
```

When prompted, enter a travel planning question. Try the prompts below to exercise different combinations of weather and group type:

| Scenario | Prompt |
|----------|--------|
| Rainy day, family | `I'll be in London on 2026-03-14 with my family. What should we do?` |
| Sunny day, family | `I'll be in London on 2026-03-15 with my kids. What should we do?` |
| Rainy day, adults | `I'm in London on 2026-03-14 for a night out with friends. What do you suggest?` |

The assistant will call `get_weather` and `get_top_attractions`, then produce a recommendation grounded in the tool results.

---

## Expected Output

```
Travel Planner
========================================
Ask me to help plan your visit to a city.

You: I'll be in London this Saturday with my family. What should we do?
  [tool call] get_weather({'city': 'London', 'date': '2026-03-14'})
  [tool result] {'city': 'London', 'condition': 'Light rain in the morning...', ...}
  [tool call] get_top_attractions({'city': 'London'})
  [tool result] {'city': 'London', 'attractions': [...]}