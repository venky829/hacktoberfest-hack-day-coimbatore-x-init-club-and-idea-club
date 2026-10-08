import os
from typing import Type
from PIL import Image
from pydantic import BaseModel
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()


class GeminiGemmaClient:
    """
    Dedicated client for calling Google Gemma and Gemini multimodal models
    via the official Google GenAI SDK.
    """

    def __init__(self, api_key: str | None = None, model_name: str | None = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Please add it to your .env file or environment variables."
            )

        # Default model identifier for Gemma multimodal via Gemini API
        self.model_name = model_name or os.getenv("GEMINI_GEMMA_MODEL", "gemma-4-31b-it")
        self.client = genai.Client(api_key=self.api_key)

    def extract_structured_data(
        self,
        image_path: str,
        prompt: str,
        schema: Type[BaseModel],
        temperature: float = 0.1,
    ) -> BaseModel:
        """
        Submits an image and prompt to the model and forces output to match
        the provided Pydantic schema using structured outputs.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image = Image.open(image_path)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[prompt, image],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
                temperature=temperature,
            ),
        )

        if not response.text:
            raise RuntimeError("Received empty response from Gemini API.")

        return schema.model_validate_json(response.text)

    def test_connection(self) -> bool:
        """Simple ping to verify API credentials during startup."""
        try:
            res = self.client.models.generate_content(
                model=self.model_name,
                contents="Ping",
            )
            return bool(res.text)
        except Exception:
            return False
