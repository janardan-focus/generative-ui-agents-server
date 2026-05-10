"""
Configuration management using pydantic-settings.
Reads from environment variables / .env file.
"""

from langchain_core import env
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

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_file_encoding="utf-8")


settings = Settings()