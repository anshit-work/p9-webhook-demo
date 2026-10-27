"""Travel research agent: looks up a destination and reports its weather."""

import requests
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient

SYSTEM_PROMPT = """You are a travel research assistant.

Call search_web first for background on the destination, then geocode_city to
resolve it to coordinates, then get_current_weather for the forecast.

Never guess coordinates. Always cite the source URL for anything you report.
If a tool fails, say so plainly rather than inventing a result.
"""


@tool
def search_web(query: str) -> dict:
    """Search the web for background on a destination."""
    response = requests.post(
        "https://api.tavily.com/search", json={"query": query}, timeout=20
    )
    response.raise_for_status()
    return response.json()


@tool
def geocode_city(city: str) -> dict:
    """Resolve a city name to latitude and longitude."""
    response = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


@tool
def get_current_weather(latitude: float, longitude: float) -> dict:
    """Return the current weather for a coordinate pair."""
    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,wind_speed_10m",
        },
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


mcp_client = MultiServerMCPClient(
    {
        "filesystem": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "./notes"],
            "transport": "stdio",
        },
        "github": {
            "url": "https://api.githubcopilot.com/mcp/",
            "transport": "http",
        },
    }
)

agent = create_agent(
    model="anthropic:claude-sonnet-5",
    tools=[search_web, geocode_city, get_current_weather],
    system_prompt=SYSTEM_PROMPT,
)
