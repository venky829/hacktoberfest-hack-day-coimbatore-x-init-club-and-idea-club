# [Project Name]

> [Nutrilens]

## Team

**Team Name:** [Open Minds]


| Member | Contribution   |
| ------ | -------------- |
| [Naraen AL] | [Frontend developer] |
| [Venkatesh ] | [Backend developer] |
| [Pragnesh] | [setting up harness] |
| [Gowtham] | [skills and tests] |


## Problem Statement

### The Problem

[Consumers face difficulty understanding complex ingredient lists on packaged food labels. Technical chemical names, hidden additives, and ambiguous E-numbers make it hard for everyday shoppers—especially those with dietary restrictions, allergies, or health conditions—to make informed purchasing decisions.]

### Why We Chose This Problem

[We selected this problem to empower consumers with transparency regarding what they consume daily. Simplifying ingredient lists helps people make safer, healthier food choices quickly while shopping..]

## Solution

[The Food Product Ingredient Analyzer allows users to upload an image or input the text of any food label. The system extracts the ingredients, evaluates potential health risks, identifies hidden additives or allergens, and generates an easy-to-understand consumer summary report..]

### Key Features

- [Label OCR & Text Parsing:** Instant extraction of ingredient text from product label photos.]
- [Health Risk & Safety Breakdown:** Categorizes additives, preservatives, and chemicals as safe, moderate, or high concern.]
- [Custom Allergen & Diet Alerts:** Flags ingredients matching user preferences (e.g., vegan, gluten-free, nut allergy]
- [Simplified Consumer Summary:** Translates complex chemical names into plain, readable explanations.]

## Innovation and Differentiation

[Unlike traditional scanner apps that rely solely on static barcode databases, our solution analyzes raw ingredient text directly using OCR and AI. This allows it to evaluate new, unlisted, or local food products in real time without needing a pre-existing product database entry.]

## Technical Implementation

### Architecture

[[User Uploads Label Image / Text] --> B[Frontend Interface]
    B --> C[Backend API Server]
    C --> D[OCR Engine / Text Extractor]
    D --> E[AI Ingredient Analysis Module]
    E --> F[Ingredient & Safety Database]
    E --> G[Health & Risk Report Generation]
    G --> B]

### Technology Stack


| Category        | Technologies                |
| --------------- | --------------------------- |
| Frontend        | [Technologies / N/A]        |
| Backend         | [Technologies / N/A]        |
| Database        | [Technologies / N/A]        |
| AI / ML         | [Models / frameworks / N/A] |
| Infrastructure  | [Technologies / N/A]        |
| APIs / Services | [Services / N/A]            |


If a category or technology is not implemented in the project, specify `N/A` instead of leaving the field blank.

### How It Works

[1. User Interface (ui/app.py): Streamlit dashboard. Handles webcam/file inputs, engine toggles, and displays audit scorecards.

2. Backend Router (backend/main.py): FastAPI gateway. Manages image ingestion, coordinates model extraction, and passes data to verification scripts.

3. Inference Harness (harness/engine.py): Dual-mode VLM parser. Runs Gemma 4 E2B/E4B locally (Ollama on RTX 3050) with failover to Gemma 4 31B (Cloud Gemini API). Enforces JSON output.

4. Agent Skill Engine (skills/): Agent Skill Open Standard container. Houses SKILL.md, deterministic Atwater math scripts (macro_verifier.py), and sweetener lookups (references/).]

### Technical Decisions

1.Hybrid Edge-Cloud Inference (harness/)
Local First: Runs quantized Gemma 4 (E2B/E4B) on local hardware (RTX 3050, 6GB VRAM) via Ollama for zero-latency, private, offline execution.
Automatic Failover: Automatically escalates to Gemma 4 (31B) via the Gemini API if local memory limits or blurry text are encountered, preventing demo crashes.

2.Perception vs. Computation Separation (skills/)
VLM Perception: Gemma 4 acts strictly as an OCR/extractor, outputting raw numerical/ingredient data directly to a structured JSON schema.
Deterministic Math: Offloads mathematical auditing to Python (macro_verifier.py). Uses the 4:4:9 Atwater Factor System to calculate real calories and catch label discrepancies without LLM math hallucinations.

3.Agent Skill Open Standard Compliance
Packaged cleanly in skills/nutrilens-auditor/SKILL.md following standard frontmatter rules.
Uses progressive disclosure—referencing external glycemic index tables (references/) only when needed to keep token costs low.

4.Zero-Conflict 4-Person Modular Workflow
Strictly isolated directory boundaries so team members work in parallel without Git merge conflicts:
ui/ (Streamlit) | backend/ (FastAPI) | harness/ (Gemma Engine) | skills/ & tests/ (Skill & Pytest)

5.Fail-Safe Offline Fixtures (fixtures/)
Includes synthesized mock label images and pre-cached JSON extraction files to ensure 100% demo uptime even if venue Wi-Fi drops.

## Implementation During the Hackathon
During Hack Day, the team built NutriLens—an open-source, edge-first Agent Skill powered by multimodal Gemma 4 that audits packaged food labels for mathematical inaccuracies and deceptive marketing claims.
[1. Dual-Engine Inference Harness (harness/)
Edge Mode: Quantized Gemma 4 (E2B/E4B) running locally via Ollama on an RTX 3050 (6GB VRAM) for offline execution.
Cloud Failover: Dynamic fallback to Gemma 4 (31B) via Gemini API for high-resolution or curved labels.

2. Deterministic Agent Skill Engine (skills/)
Open Standard Compliant: Structured under the agentskills.io specification in SKILL.md.
Atwater Math Verification: Python script (macro_verifier.py) calculating true calories via (P×4)+(Net C×4)+(Fiber×2)+(F×9) to catch label rounding errors without LLM math hallucinations.
Deceptive Filler Detection: Cross-references ingredients against references/glycemic_index.json to flag hidden high-GI sugars (e.g., maltodextrin).

3. FastAPI Backend (backend/)
Created POST /api/audit to bridge image ingestion, Gemma 4 extraction, and skill script execution.

4. Streamlit Dashboard (ui/)
Built an interactive frontend with webcam/file upload, inference toggles, and live Claim Integrity Score (0–100) displays.

5. CI/CD & Demo Fixtures (fixtures/, tests/)
Set up GitHub Actions CI (test.yml), Pytest suites, and offline mock label images for demo reliability.]

### Team Contributions

| [Naraen AL] | [Frontend developer] |
| [Venkatesh ] | [Backend developer] |
| [Pragnesh] | [setting up harness] |
| [Gowtham] | [skills and tests] |

## Working Application

**Live Application:** [Live URL]

[Briefly explain how the deployed application can be accessed and what functionality can be tested.]

The submitted application should be functional and accessible through the provided link where applicable.

## Demo Video

**Demo Video:** [Video URL]

[Provide a short demonstration of the working project, covering the main user flow and important functionality.]

## Open Source and AI Usage

### AI / Models

Gemma 4 (31B-IT via Gemini API) : Primary cloud multimodal model used in harness/engine.py for high-resolution vision parsing, text extraction, and structured JSON generation from food packaging photos.
Gemma 4 (2B / 4B Quantized via Ollama) : Local edge vision model executed directly on consumer GPU hardware (NVIDIA RTX 3050, 6GB VRAM) for offline, zero-latency label extraction.

### Open Source Components

Library / Framework	: google-genai	Official Google SDK used to connect to the Gemini API and invoke Gemma 4 with structured Pydantic schema enforcement.
Library / Framework	fastapi & uvicorn :	Lightweight REST API backend routing requests between the user interface and the inference harness.
Library / Framework	streamlit :	Interactive web dashboard providing image upload, camera capture, and visual metric displays.
Library / Framework	pydantic	Enforces type safety, JSON schema validation, and structured output parsing across the pipeline.
Library / Framework	pillow (PIL)	Handles image preprocessing, formatting, and file stream conversions before model ingestion.
Library / Framework	pytest	Automated testing framework for verifying Atwater factor calculations, deceptive ingredient detection, and allergen scanning.
Specification	Agent Skill Open Standard (agentskills.io)	Open specification governing skills/nutrilens-auditor/SKILL.md structure, parameters, and execution rules.
API / Service	Google Gemini API	Cloud endpoint providing managed access to Gemma 4 multimodal models.
API / Service	Ollama	Local open-source model harness and HTTP inference server for running quantized open weights on edge GPUs.

[Include relevant licenses, attribution, and acknowledgements for external components.]

## Setup and Usage

### Prerequisites

- [Requirement]
- [Requirement]

### Installation

```bash
git clone [repository-url]
cd [project-directory]
[installation-command]
```

### Environment Variables

```env
[VARIABLE_NAME]=[value]
```



### Running the Project

```bash
[run-command]
```

### Usage

[Explain the basic steps required to use the project.]

## Devpost Submission

**Devpost Project:** [Devpost Project URL]

[Add the link to the team's Devpost submission. Ensure the Devpost project page is complete and contains the required project information, links, media, and team details.]

## Credits and License

### Credits

[Credit libraries, frameworks, datasets, models, APIs, contributors, and other external resources used.]

### License

[License name and/or link.]

## Submission Checklist

- [ ] Project title and description added
- [ ] All team members listed
- [ ] Problem clearly explained
- [ ] Reason for choosing the problem explained
- [ ] Solution and key features documented
- [ ] Innovation and differentiation explained
- [ ] Architecture included
- [ ] Technical implementation documented
- [ ] Work completed during the hackathon documented
- [ ] Team contributions documented
- [ ] Working application is functional
- [ ] Live application link added where applicable
- [ ] Demo video added
- [ ] AI and open-source components documented
- [ ] Setup and usage instructions tested
- [ ] Challenges and learnings documented
- [ ] Devpost submission completed
- [ ] Devpost link added
- [ ] Credits added
- [ ] License added
- [ ] Repository is organized and complete
