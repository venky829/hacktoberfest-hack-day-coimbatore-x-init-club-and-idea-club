import os
import json
import time
from pathlib import Path
from PIL import Image
from pydantic import BaseModel, Field
import requests
from google import genai
from google.genai import types
from google.genai.errors import ServerError
from dotenv import load_dotenv

# Ensure .env is explicitly loaded from the root directory
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


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
        # Define preferred and fallback models
        self.candidate_models = [
            os.getenv("GEMINI_GEMMA_MODEL", "gemini-3.8-flash"),
            "gemini-3.5-flash-lite",
        ]
        self.local_model = os.getenv("LOCAL_OLLAMA_MODEL", "gemma2:2b")
        self.ollama_url = "http://localhost:11434/api/generate"

        if not self.api_key:
            print("⚠️ WARNING: GEMINI_API_KEY is not set in environment or .env!", flush=True)

        self.gemini_client = genai.Client(api_key=self.api_key) if self.api_key else None

    def _call_gemini_multimodal(self, image_path: str) -> ExtractedNutritionData:
        """Uses Multimodal API to parse the food label into structured JSON with 503 retry & fallback."""
        if not self.gemini_client:
            raise ValueError("GEMINI_API_KEY is missing or invalid. Check your .env file.")

        img = Image.open(image_path)
        prompt = (
            "Analyze this food packaging or nutrition label carefully. "
            "Extract: product name, calories, protein_g, carbs_g, fiber_g, fat_g, "
            "and all listed ingredients."
        )

        last_error = None
        for model_name in self.candidate_models:
            for attempt in range(2):
                try:
                    print(f"☁️ [API Call] Calling {model_name} (Attempt {attempt + 1})...", flush=True)
                    response = self.gemini_client.models.generate_content(
                        model=model_name,
                        contents=[prompt, img],
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=ExtractedNutritionData,
                            temperature=0.1,
                        ),
                    )
                    return ExtractedNutritionData.model_validate_json(response.text)
                except ServerError as err:
                    print(f"⚠️ {model_name} high demand (503): {err}. Retrying...", flush=True)
                    last_error = err
                    time.sleep(1.5)
                except Exception as err:
                    print(f"⚠️ {model_name} error: {err}. Cascading to next candidate...", flush=True)
                    last_error = err
                    break

        raise RuntimeError(f"All multimodal endpoints unavailable: {last_error}")

    def _call_local_gemma_audit(self, extracted: dict) -> dict:
        """Runs offline text analysis on local RTX 3050 using Gemma 2B."""
        prompt = f"""
        Audit these nutritional facts:
        {json.dumps(extracted)}

        Analyze if ingredients contain hidden sugars (maltodextrin, dextrose, syrups).
        Output short findings.
        """
        try:
            res = requests.post(
                self.ollama_url,
                json={
                    "model": self.local_model,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=15,
            )
            if res.status_code == 200:
                return {"local_analysis": res.json().get("response", "")}
        except Exception as e:
            print(f"Local Ollama audit skipped: {e}", flush=True)
        return {}

    def process_image(self, image_path: str, force_cloud: bool = False) -> tuple[dict, str]:
        print("☁️ [Multimodal Vision] Extracting label metadata via API...", flush=True)
        extracted_model = self._call_gemini_multimodal(image_path)
        extracted_data = extracted_model.model_dump()

        # Run local agent processing on your RTX 3050
        print("⚡ [Local GPU Engine] Verifying formulation locally on RTX 3050 (gemma2:2b)...", flush=True)
        local_feedback = self._call_local_gemma_audit(extracted_data)
        extracted_data.update(local_feedback)

        execution_source = "Hybrid (Gemma Multimodal Vision + Local RTX 3050 gemma2:2b)"
        return extracted_data, execution_source
