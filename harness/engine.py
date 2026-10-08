import os
import json
import base64
import requests
from PIL import Image
from pydantic import BaseModel, Field, ValidationError
from google import genai
from google.genai import types
from .gemini_client import GeminiGemmaClient
from dotenv import load_dotenv

load_dotenv()

# Strict schema for our extracted food label data
class ExtractedNutritionData(BaseModel):
    brand_or_product: str = Field(default="Unknown")
    calories: float = Field(default=0.0)
    protein_g: float = Field(default=0.0)
    carbs_g: float = Field(default=0.0)
    fiber_g: float = Field(default=0.0)
    fat_g: float = Field(default=0.0)
    ingredients: list[str] = Field(default_factory=list)

class HybridInferenceHarness:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.cloud_model = os.getenv("GEMINI_GEMMA_MODEL", "gemma-4-31b-it")
        self.local_model = os.getenv("LOCAL_OLLAMA_MODEL", "gemma2:2b")
        self.ollama_url = "http://localhost:11434/api/generate"

        # Initialize the Google GenAI SDK client
        if self.api_key:
            self.gemini_client = genai.Client(api_key=self.api_key)
        else:
            self.gemini_client = None

    def _call_local_ollama(self, image_path: str) -> ExtractedNutritionData:
        """Runs locally on your RTX 3050 (6GB VRAM) via Ollama."""
        with open(image_path, "rb") as f:
            b64_image = base64.b64encode(f.read()).decode("utf-8")

        prompt = """
        Extract the nutrition table and ingredient list from this packaging image.
        Return strictly valid JSON with these exact keys:
        {
          "brand_or_product": "string",
          "calories": float,
          "protein_g": float,
          "carbs_g": float,
          "fiber_g": float,
          "fat_g": float,
          "ingredients": ["string"]
        }
        """

        response = requests.post(
            self.ollama_url,
            json={
                "model": self.local_model,
                "prompt": prompt,
                "images": [b64_image],
                "format": "json",
                "stream": False
            },
            timeout=25
        )

        if response.status_code != 200:
            raise RuntimeError(f"Ollama server error: {response.text}")

        raw_json_str = response.json().get("response", "{}")
        parsed = json.loads(raw_json_str)
        return ExtractedNutritionData.model_validate(parsed)

    def _call_gemini_cloud_api(self, image_path: str) -> ExtractedNutritionData:
        """Calls Gemma 4 through the official Gemini API."""
        if not self.gemini_client:
            raise ValueError("GEMINI_API_KEY is missing from environment (.env).")

        img = Image.open(image_path)
        prompt = (
            "Extract the product name, calories, protein_g, carbs_g, fiber_g, fat_g, "
            "and all listed ingredients from this packaging label image into structured JSON."
        )

        response = self.gemini_client.models.generate_content(
            model=self.cloud_model,
            contents=[prompt, img],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ExtractedNutritionData,
                temperature=0.1
            )
        )
        return ExtractedNutritionData.model_validate_json(response.text)

    # ==============================================================
    # THE SWITCHING MECHANISM
    # ==============================================================
    def process_image(self, image_path: str, force_cloud: bool = False) -> tuple[dict, str]:
        """
        Routes the task intelligently:
        1. If force_cloud is True -> routes straight to Gemini API.
        2. Otherwise -> executes on local GPU (RTX 3050).
        3. If local execution fails or yields invalid schema -> automatically falls back to Gemini API.
        """
        # Manual Override Route
        if force_cloud:
            print("🌐 [Route: Cloud API] User forced Cloud Gemma 4 execution.")
            result = self._call_gemini_cloud_api(image_path)
            return result.model_dump(), "Cloud (Gemma 4 Gemini API)"

        # Local First Route
        try:
            print(f"⚡ [Route: Edge GPU] Attempting local inference via Ollama ({self.local_model})...")
            result = self._call_local_ollama(image_path)
            return result.model_dump(), "Edge (RTX 3050 Local Ollama)"

        except (requests.exceptions.RequestException, json.JSONDecodeError, ValidationError, RuntimeError) as err:
            # Fallback Route
            print(f"⚠️ [Switching Triggered] Local inference failed or schema incomplete: {err}")
            print(f"🚀 [Route: Failover] Escalating to Cloud Gemma 4 via Gemini API...")
            result = self._call_gemini_cloud_api(image_path)
            return result.model_dump(), "Cloud Fallback (Gemma 4 Gemini API)"