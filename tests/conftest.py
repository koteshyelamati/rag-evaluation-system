import os

# Set before any app modules are imported so pydantic-settings can read it
os.environ.setdefault("GEMINI_API_KEY", "test-api-key-12345")
