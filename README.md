# Roger — AI-Powered Voice Enrollment Agent

> **Production AI system** built for a US healthcare technology company, enabling patients to enroll in government assistance programs (SNAP, Medicaid, WIC, etc.) entirely over the phone using natural conversation — no app, no website, no human agent required.

This document describes the engineering architecture, design decisions, and technical depth behind the system. The codebase is proprietary and cannot be shared publicly.

---

## What It Does

A patient dials a Twilio phone number. Roger answers, identifies the caller's organization, presents available assistance programs, walks the caller through eligibility screening and identity verification, collects application answers one question at a time through natural voice conversation, obtains consent, and submits the completed application to the backend — all within a single phone call. If the caller speaks Spanish, Arabic, Haitian Creole, or four other languages, Roger switches automatically.

The system handles multiple organizations (tenants) from the same phone infrastructure, routes each caller to the correct program catalog, and maintains fully isolated session state per concurrent call.

---

## Core Engineering Challenges Solved

| Challenge | Solution |
|---|---|
| Sub-200ms barge-in so the agent stops mid-sentence when a caller interrupts | Parallel Deepgram live WebSocket STT fires `SpeechStarted` → kills FastRTC generator → sends Twilio `clear` flush — total latency ~100–150ms |
| Concurrent multi-tenant sessions with zero state bleed | Fully stateless `MCPClient`; all session tokens passed as explicit kwargs; LangGraph thread-per-call checkpointing |
| 7-language support with mixed STT/TTS provider capabilities | Per-language provider capability matrix; automatic fallback chains (Deepgram → Whisper for Haitian Creole; Deepgram TTS → OpenAI for unsupported languages) |
| Voice input is noisy and ambiguous | LLM structured-output classification at every decision point — never raw string matching |
| Application questions are dynamic and paginated | Stateful LangGraph question-loop with server-side visibility rules evaluated client-side |
| Caller identity verification mid-call | OTP flow wired into enrollment state; token refresh handled transparently on every API call |
| IVR-quality menus without a real IVR | DTMF and voice selection unified through the same LLM classifier; `#` returns to main menu at any stage |

---

## System Architecture

![System Architecture Diagram](images/architecture.png)

---

## Folder Structure

```
roger/
│
├── agents/
│   ├── embedding_service/
│   │   ├── __init__.py
│   │   └── agent.py
│   └── multipurpose_bot/
│       ├── __init__.py
│       ├── agent.py
│       └── subagents/
│           ├── __init__.py
│           ├── program_application_agent.py
│           └── program_intake_agent.py
│
├── config/
│   ├── __init__.py
│   ├── locales/
│   │   ├── __init__.py
│   │   ├── en.py
│   │   ├── es.py
│   │   ├── ht.py
│   │   ├── ar.py
│   │   ├── fr.py
│   │   ├── pt.py
│   │   ├── fa.py
│   │   └── hi.py
│   ├── prompts.py
│   ├── scripts.py
│   ├── settings.py
│   ├── tenant_registry.py
│   └── tenants.yaml
│
├── core/
│   ├── __init__.py
│   ├── stt/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── openai.py
│   │   ├── deepgram.py
│   │   └── utils.py
│   ├── tts/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── openai.py
│   │   ├── deepgram.py
│   │   ├── helpers.py
│   │   └── utils.py
│   ├── thrivelink/
│   │   ├── __init__.py
│   │   ├── client.py
│   │   ├── transforms.py
│   │   └── tools/
│   │       ├── __init__.py
│   │       ├── auth.py
│   │       ├── questions.py
│   │       ├── application.py
│   │       └── menu.py
│   ├── audio_service.py
│   ├── base_agent.py
│   ├── llm_services.py
│   ├── mcp_client.py
│   ├── models.py
│   ├── state.py
│   ├── stream.py
│   └── utils.py
│
├── storage/
│   ├── __init__.py
│   └── vector_store.py
│
├── observability/
│   ├── __init__.py
│   ├── logger.py
│   ├── metrics.py
│   ├── tracing.py
│   └── health.py
│
├── api/
│   ├── __init__.py
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── voice.py
│   │   ├── health.py
│   │   └── admin.py
│   └── middleware.py
│
├── main.py
├── docker-compose.yml
├── Dockerfile
├── langgraph.json
├── .env.example
├── requirements.txt
└── pyproject.toml
```

---

## Component Architecture

**BaseAgent** (ABC)
- MultipurposeBot
  - ProgramApplicationAgent (22 graph nodes: play_intro → … → complete)
  - EmbeddingServiceAgent

**STTModel** (ABC)
- WhisperSTT (OpenAI)
- DeepgramSTT (Deepgram Nova-2)

**TTSModel** (ABC)
- OpenAITTSModel (TTS-1)
- DeepgramTTSModel (Aura)

**AudioService**
- Transcription orchestration with provider fallback
- TTS streaming with provider fallback
- Per-call session isolation

**ThriveLinkAPIClient**
- Async HTTP with token refresh
- Cookie-jar session management
- Per-request auth injection

**VoiceAgentStream**
- Twilio WebSocket frame handling
- DTMF buffering and routing
- Barge-in via speech detection
- Session lifecycle management
└── text_to_speech(text, voice) → bytes (MP3)

---

## Observability

The observability stack provides production-grade monitoring, tracing, and health checks:

**Logging** (`observability/logger.py`)
- Structured JSON logging with request ID correlation
- Configurable log levels per module
- PII masking for phone and identity data

**Metrics** (`observability/metrics.py`)
- Call success/failure rates
- Latency percentiles (p50, p95, p99)
- STT/TTS provider performance
- Agent node execution times

**Tracing** (`observability/tracing.py`)
- OpenTelemetry integration with OTLP export
- Per-call trace context propagation
- Deepgram, OpenAI, and internal service spans

**Health Checks** (`observability/health.py`)
- Liveness probe (basic health)
- Readiness probe (dependencies up)
- Dependency status: LLM, STT, TTS, ThriveLinkAPI

---

## API Routes

The FastAPI application exposes three route groups:

**Voice Routes** (`api/routes/voice.py`)
- `POST /voice/incoming` — Twilio callback, returns TwiML
- `WS /voice/stream/{session_id}` — Bidirectional audio stream

**Health Routes** (`api/routes/health.py`)
- `GET /health/live` — Liveness probe
- `GET /health/ready` — Readiness probe
- `GET /health/metrics` — Prometheus-format metrics

**Admin Routes** (`api/routes/admin.py`)
- `GET /admin/sessions/{session_id}` — Session state dump
- `POST /admin/sessions/{session_id}/reset` — Reset call state
- `GET /admin/config` — Active configuration dump

---

## Deployment

**Docker**
- Multi-stage build: Python 3.11 slim base
- System dependencies: FFmpeg, libsndfile, opus-tools
- Runtime: gunicorn + uvicorn worker pool

**Docker Compose** (`docker-compose.yml`)
- Roger API service
- Chroma vector store
- Environment configuration per tenant

**LangGraph Cloud** (`langgraph.json`)
- Deployment manifest for LangGraph graph execution
- Per-tenant isolation and scaling

**Environment**
- `.env.example` template with all required variables
- Runtime validation via Pydantic settings
- Secrets injected at deployment time

---

## Key Features

- **Multi-tenant support** — Multiple organizations on a single phone number infrastructure
- **7-language support** — Auto-switching between English, Spanish, Arabic, Haitian Creole, French, Portuguese, Farsi, and Hindi
- **Sub-200ms barge-in** — Caller can interrupt mid-sentence with dual STT detection (Deepgram live + FastRTC VAD)
- **Provider fallback** — Automatic fallback between STT/TTS providers if one is unavailable
- **Session isolation** — Fully isolated per-call state with no cross-contamination in concurrent calls
- **Structured outputs** — LLM responses validated against Pydantic schemas, never free-form parsing
- **Observability** — Structured logging, metrics, and tracing for production monitoring

---

## Technology Stack

- **Agent Framework:** LangGraph with state persistence
- **LLM:** OpenAI GPT-4.1-mini and GPT-4o-mini
- **Telephony:** Twilio Media Streams
- **Real-time Audio:** FastRTC (VAD and streaming)
- **STT:** OpenAI Whisper-1, Deepgram Nova-2 (batch and live WebSocket)
- **TTS:** OpenAI TTS-1, Deepgram Aura
- **Vector DB:** ChromaDB for RAG embeddings
- **Web Framework:** FastAPI + Uvicorn
- **Data Validation:** Pydantic v2
- **HTTP Client:** httpx (async)
- **Deployment:** Docker Compose, Nginx, Let's Encrypt

---

## Development Setup

```bash
# Clone and install dependencies
git clone <repo>
cd roger
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Run locally with Docker Compose
docker-compose up -d

# Run tests
pytest tests/ -v

# Start development server
python main.py
```

Visit `http://localhost:8001` for the Twilio webhook setup or `http://localhost:7860` for the Gradio demo.

---

## Contributing

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Commit changes: `git commit -am 'Add feature'`
3. Push to branch: `git push origin feature/your-feature`
4. Submit a pull request

---

## License

Proprietary — All rights reserved. See [LICENSE](LICENSE) for details.
