import os
import shutil
import tempfile
import subprocess
import json
from fastapi import FastAPI, UploadFile, File, Form
from harness.engine import HybridInferenceHarness

app = FastAPI(title="NutriLens Core API")
engine = HybridInferenceHarness()


@app.get("/")
def read_root():
    return {"status": "ok", "service": "NutriLens API"}


@app.post("/api/audit")
async def audit_endpoint(file: UploadFile = File(...), force_cloud: bool = Form(False)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # Step 1: Hybrid multimodal extraction
        extracted, execution_source = engine.process_image(tmp_path, force_cloud=force_cloud)

        # Step 2: Deterministic verification via Agent Skill script
        script_path = os.path.join("skills", "nutrilens-auditor", "scripts", "macro_verifier.py")
        proc = subprocess.Popen(
            ["python", script_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, _ = proc.communicate(input=json.dumps(extracted))
        audit_res = json.loads(stdout) if stdout.strip() else {}

        return {
            "status": "success",
            "execution_source": execution_source,
            "extracted": extracted,
            "audit": audit_res
        }
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
