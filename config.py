"""
Configuration management using pydantic-settings.
Reads from environment variables / .env file.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    # LLM
    google_api_key: str | None = Field(default=None, description="Google AI Studio API key")
    google_model: str = Field(
        default="gemini-2.0-flash",
        description="Google Gemini model to use for agents",
    )

    # MCP server (Ticket Management System)
    mcp_server_url: str = Field(
        default="http://localhost:8001/mcp",
        description="HTTP URL of the MCP server endpoint",
    )

    # FastAPI server
    agent_server_port: int = Field(default=8000)
    cors_origins: list[str] = Field(
        default=["http://localhost:5173", "http://localhost:3000"],
        description="Allowed CORS origins",
    )

    # Session lifecycle (in-process store)
    idle_timeout_seconds: int = Field(
        default=1800,
        description="Seconds of inactivity before a session is expired (default: 30 min)",
    )
    session_sweep_interval_seconds: int = Field(
        default=300,
        description="How often the background sweep evicts idle/closed sessions (default: 5 min)",
    )
    max_sessions: int = Field(
        default=1000,
        description="Safety cap on concurrent in-memory sessions; oldest evicted first",
    )

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_file_encoding="utf-8")


settings = Settings()