import os
import json
import base64
import requests
from typing import Dict, Any, Optional
from PIL import Image
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables (GEMINI_API_KEY)
load_dotenv()


class ExtractedNutrition(BaseModel):
    """Pydantic schema for structured Gemma 4 vision outputs."""
    brand_or_product: Optional[str] = Field(default="Unknown Product", description="Name or brand of the product")
    serving_size: Optional[str] = Field(default="1 serving", description="Declared serving size")
    calories: float = Field(default=0.0, description="Declared total calories per serving")
    protein_g: float = Field(default=0.0, description="Total protein in grams")
    carbs_g: float = Field(default=0.0, description="Total carbohydrates in grams")
    fiber_g: float = Field(default=0.0, description="Dietary fiber in grams")
    fat_g: float = Field(default=0.0, description="Total fat in grams")
    marketing_claims: list[str] = Field(default_factory=list, description="Packaging claims like Zero Sugar, High Protein")
    ingredients: list[str] = Field(default_factory=list, description="Full list of declared ingredients")


class InferenceEngine:
    """
    Dual-Inference Harness for NutriLens.
    - Local Mode: Edge inference on RTX 3050 (6GB VRAM) via Ollama.
    - Cloud Mode: Deep multimodal reasoning using Gemma 4 on the Gemini API.
    """

    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.gemini_client = genai.Client(api_key=self.api_key) if self.api_key else None

    def extract_local_ollama(self, image_path: str) -> Dict[str, Any]:
        """
        Executes local multimodal inference using quantized Gemma weights on Ollama.
        Optimized for 6GB VRAM consumer GPUs (RTX 3050).
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at path: {image_path}")

        # Base64 encode image for Ollama HTTP API
        with open(image_path, "rb") as f:
            b64_image = base64.b64encode(f.read()).decode("utf-8")

        prompt = (
            "You are a precision nutrition parser. Analyze this food label and extract the nutrition "
            "facts table and ingredient list. Format strictly as JSON matching these keys:\n"
            "{\n"
            '  "brand_or_product": "string",\n'
            '  "serving_size": "string",\n'
            '  "calories": float,\n'
            '  "protein_g": float,\n'
            '  "carbs_g": float,\n'
            '  "fiber_g": float,\n'
            '  "fat_g": float,\n'
            '  "marketing_claims": ["string"],\n'
            '  "ingredients": ["string"]\n'
            "}"
        )

        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": "gemma",  # Local Ollama model name
                "prompt": prompt,
                "images": [b64_image],
                "format": "json",
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_ctx": 2048
                }
            },
            timeout=30
        )

        if response.status_code != 200:
            raise RuntimeError(f"Ollama API returned status code {response.status_code}: {response.text}")

        res_json = response.json()
        raw_text = res_json.get("response", "")
        
        return json.loads(raw_text)

    def extract_cloud_gemini(self, image_path: str) -> Dict[str, Any]:
        """
        Executes cloud multimodal extraction using Gemma 4 via the Google GenAI SDK.
        Uses structured schema enforcement to guarantee type-safe JSON output.
        """
        if not self.gemini_client:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at path: {image_path}")

        image = Image.open(image_path)
        prompt = (
            "You are an expert food safety auditor. Parse this image to extract all nutritional "
            "metrics, packaging marketing statements, and the complete ingredient list."
        )

        response = self.gemini_client.models.generate_content(
            model="gemma-4-31b-it",
            contents=[prompt, image],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ExtractedNutrition,
                temperature=0.1
            )
        )

        return json.loads(response.text)

    def process(self, image_path: str, force_cloud: bool = False) -> Dict[str, Any]:
        """
        Main execution routing:
        1. If force_cloud is False, attempts Edge execution on RTX 3050 first.
        2. Automatically fails over to Cloud Gemini API if local execution fails.
        3. If force_cloud is True, routes directly to Gemini API.
        """
        if not force_cloud:
            try:
                print("⚡ [Edge Harness] Running local Ollama inference on RTX 3050...")
                return self.extract_local_ollama(image_path)
            except Exception as e:
                print(f"⚠️ [Edge Fallback] Local Ollama execution failed: {e}")
                print("🔄 [Failover] Escalating request to Cloud Gemma 4 via Gemini API...")

        print("☁️ [Cloud Harness] Executing via Gemini API (Gemma 4)...")
        return self.extract_cloud_gemini(image_path)


# Quick CLI testing entrypoint
if __name__ == "__main__":
    import sys
    engine = InferenceEngine()
    test_img = sys.argv[1] if len(sys.argv) > 1 else "fixtures/clean_protein_bar.jpg"
    
    if os.path.exists(test_img):
        print(f"Testing harness with image: {test_img}")
        result = engine.process(test_img, force_cloud=True)
        print(json.dumps(result, indent=2))
    else:
        print(f"Provide a valid test image path or place a test image at '{test_img}'.")