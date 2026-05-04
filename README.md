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

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TWILIO PSTN                                     │
│   Incoming call → TwiML → Media Stream (WebSocket, µ-law 8kHz, bidirectional)│
└──────────────────────────┬──────────────────────────────────────────────────┘
                           │  WS frames
                           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         STREAM LAYER  (core/stream.py)                        │
│                                                                               │
│  VoiceAgentStream                                                             │
│  ├── _TwilioStartCapture  — intercepts start/DTMF events from WS proxy        │
│  ├── InterruptibleReplyOnPause  — VAD + barge-in via FastRTC ReplyOnPause     │
│  │     └── determine_pause() — fires on_started_talking → kills generator    │
│  ├── _send_greeting_after_connect()  — stream_id resolution + greeting loop  │
│  ├── _handle_dtmf_digit()  — digit buffering, # → main-menu shortcut         │
│  └── _process_return_to_menu()  — state reset + re-invocation                │
└──────────────────────────┬───────────────────────────────────────────────────┘
                           │  PCM audio frames + transcripts
                           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    VOICE AGENT  (scripts/run_gradio_application.py)           │
│                                                                               │
│  LangGraphVoiceAgent                                                          │
│  ├── _get_session() / _cleanup_session()  — per-call session dict            │
│  ├── _process_audio()  — STT → agent invocation → TTS streaming              │
│  ├── _on_stream_connected()  — starts Deepgram live WS per call              │
│  ├── _on_speech()  — barge-in callback: cancel timer + flush + interrupt     │
│  ├── _reset_to_main_menu()  — LangGraph state patch + stage reset            │
│  └── DeepgramStreamingSTT  — live WS connection feeding per-stream           │
└──────────────────────────┬───────────────────────────────────────────────────┘
                           │  LangGraph ainvoke / update_state
                           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                   LANGGRAPH AGENT  (agents/.../program_application_agent.py)  │
│                                                                               │
│  ProgramApplicationAgent  (BaseAgent → LangGraph StateGraph)                 │
│                                                                               │
│  Graph nodes (each is an async function, routes via conditional edges):       │
│  play_intro → process_language_selection → present_categories                │
│  → process_category_selection → present_programs → process_program_selection │
│  → route_to_program → fetch_eligibility_questions → ask_eligibility_question │
│  → collect_user_info → send_otp_with_eligibility → receive_and_verify_otp   │
│  → fetch_questions → ask_question → validate_and_submit_answer               │
│  → confirm_answer → process_confirmation → request_consent                   │
│  → process_consent → finalize_application → initiate_transfer → complete     │
└──────────────────────────┬───────────────────────────────────────────────────┘
                           │  async HTTP (httpx)
                           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│              THRIVELINK TOOL LAYER  (core/thrivelink/)                        │
│                                                                               │
│  MCPClient  (stateless, concurrency-safe)                                     │
│  ├── ThriveLinkAPIClient  — shared httpx.AsyncClient, cookie-jar token store │
│  ├── tools/auth.py    — request_otp, verify_otp, register_patient, refresh   │
│  ├── tools/questions.py — public eligibility questions, application questions │
│  ├── tools/application.py — finalize_application, submit_consent             │
│  └── tools/menu.py    — get_dialer_config (org menu, programs, scripts)      │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Folder Structure

```
Thrivelink-Roger-POC/
│
├── agents/
│   ├── embedding_service/
│   │   └── agent.py                   # EmbeddingServiceAgent — RAG ingestion pipeline
│   └── multipurpose_bot/
│       ├── agent.py                   # MultipurposeBot — top-level router agent
│       └── subagents/
│           ├── program_application_agent.py   # Main enrollment LangGraph (2400+ lines)
│           └── program_intake_agent.py        # Legacy intake agent (pre-MCP)
│
├── config/
│   ├── locales/
│   │   ├── __init__.py               # Provider capability matrix, get_stt_code(), get_tts_voice()
│   │   ├── en.py                     # Full 61-key English script constants
│   │   ├── es.py                     # Full Spanish translations
│   │   ├── ht.py                     # Full Haitian Creole translations
│   │   ├── ar.py  fr.py  pt.py  fa.py  hi.py  # Full locale translations
│   ├── prompts.py                    # AgentPrompts, NodePrompts, language_instruction()
│   ├── scripts.py                    # SNAPScripts — voice script constants with __getattr__ fallback
│   ├── settings.py                   # Pydantic BaseSettings — all env vars with validation
│   ├── tenant_registry.py            # TenantConfig dataclass + YAML-backed registry
│   └── tenants.yaml                  # Per-org configuration (Twilio numbers, program IDs, voices)
│
├── core/
│   ├── stt/
│   │   ├── base.py                   # STTModel ABC
│   │   ├── openai/                   # WhisperSTT implementation
│   │   ├── deepgram/                 # DeepgramSTT (batch) + DeepgramStreamingSTT (live WS)
│   │   └── utils.py                  # get_stt_model() factory
│   ├── tts/
│   │   ├── base.py                   # TTSModel ABC
│   │   ├── openai/                   # OpenAITTSModel — PCM streaming via TTS-1
│   │   ├── deepgram/                 # DeepgramTTSModel — PCM streaming via Aura
│   │   ├── helpers.py                # clean_text_for_tts(), make_silence_pad()
│   │   └── utils.py                  # get_tts_model() factory
│   ├── thrivelink/
│   │   ├── client.py                 # ThriveLinkAPIClient — async httpx + token refresh
│   │   ├── transforms.py             # transform_dialer_config() — API → agent state shape
│   │   └── tools/
│   │       ├── auth.py               # authenticate_request_otp, authenticate_verify_otp, refresh_session
│   │       ├── questions.py          # get_public_eligibility_questions, get_application_questions, submit_answers
│   │       ├── application.py        # finalize_application, submit_consent
│   │       └── menu.py               # get_dialer_config
│   ├── audio_service.py              # AudioService — STT/TTS orchestrator with dual-provider fallback
│   ├── base_agent.py                 # BaseAgent ABC — LangGraph compilation, memory, LLM setup
│   ├── llm_services.py               # Stateless LLM calls: classify_menu_selection, extract_phone,
│   │                                 #   validate_answer, classify_confirmation, classify_consent
│   ├── mcp_client.py                 # MCPClient — high-level stateless wrapper over thrivelink tools
│   ├── mcp_tools.py                  # Legacy MCP JSON-RPC tool definitions (migration reference)
│   ├── models.py                     # Pydantic structured-output schemas (MenuSelection, ExtractionResult, etc.)
│   ├── state.py                      # All TypedDict state schemas (ProgramApplicationState, etc.)
│   ├── stream.py                     # VoiceAgentStream, InterruptibleReplyOnPause, DTMF handling
│   └── utils.py                      # Shared utilities (phone normalisation, retry decorators, etc.)
│
├── storage/
│   └── vector_store.py               # VectorStoreManager — ChromaDB client with collection management
│
├── data/
│   └── vector_store/                 # Persisted ChromaDB embeddings
│
├── scripts/
│   └── run_gradio_application.py     # LangGraphVoiceAgent — Twilio session host + audio loop
│
├── docs/                             # Technical design docs (latency analysis, MCP migration, etc.)
├── nginx/conf.d/                     # Nginx reverse-proxy configuration
├── docker-compose.yml                # Multi-service container orchestration
├── dockerfile                        # Python 3.11 slim + system audio deps
├── langgraph.json                    # LangGraph Cloud deployment manifest
└── main.py                           # FastAPI application factory + all routers
```

---

## Class Hierarchy

```
BaseAgent (ABC)  [core/base_agent.py]
├── build_graph() → StateGraph          # abstract — each subclass defines its own graph
├── compile(checkpointer)               # compiles StateGraph with InMemorySaver
├── _get_default_model()                # GPT-4.1-mini via LangChain ChatOpenAI
├── _get_light_model()                  # GPT-4o-mini for fast classification nodes
└── ainvoke() / invoke()                # delegates to compiled_graph
    │
    ├── MultipurposeBot  [agents/multipurpose_bot/agent.py]
    │   ├── route_to_snap_application() # top-level router node
    │   └── prepare_response()
    │       └── ProgramApplicationAgent  [subagents/program_application_agent.py]
    │           ├── 22 graph nodes (play_intro → … → complete)
    │           ├── _auth_kwargs(state)  # builds {session_id, refresh_token, refresh_birth}
    │           ├── _classify_caller_phone_intent()  # rule-based phone-confirmation classifier
    │           ├── _extract_phone_from_text()       # US phone normalisation
    │           └── fetch_eligibility_questions()    # paginated eligibility loop with retry
    │
    └── EmbeddingServiceAgent  [agents/embedding_service/agent.py]
        ├── analyze_content()
        ├── process_content()     # PDF / URL / text loaders
        ├── chunk_content()       # configurable chunk size + overlap
        └── store_embeddings()    # ChromaDB upsert via VectorStoreManager


STTModel (ABC)  [core/stt/base.py]
├── transcribe(audio_bytes, filename, language_code) → str   # abstract
├── WhisperSTT   [core/stt/openai/whisper.py]                # OpenAI Whisper-1 REST
└── DeepgramSTT  [core/stt/deepgram/nova.py]                 # Deepgram Nova-2 REST


TTSModel (ABC)  [core/tts/base.py]
├── stream_tts(text, voice) → AsyncIterator[(sample_rate, np.ndarray)]  # abstract
├── OpenAITTSModel   [core/tts/openai/model.py]   # PCM streaming via TTS-1
└── DeepgramTTSModel [core/tts/deepgram/model.py] # PCM streaming via Aura


AudioService  [core/audio_service.py]
├── transcribe_audio(bytes, filename, language)          # auto-fallback Deepgram → Whisper
├── transcribe_audio_with_provider(bytes, provider, code)  # locked-provider fast path
├── synthesize_speech_streaming(text, voice)             # auto-fallback OpenAI → Deepgram
├── warm_up_tts(voice)                                   # async TTS warm-up on connect
└── text_to_speech(text, voice) → bytes                  # non-streaming MP3 (REST endpoint)


MCPClient  [core/mcp_client.py]  — stateless, concurrency-safe
├── extract_refreshed_tokens(result) → dict   # @staticmethod — token refresh without mutation
├── request_login_code / verify_otp / register_patient
├── get_questions / submit_answers / send_link_sms
├── get_public_eligibility_questions / request_otp_with_eligibility
├── update_status / update_consent
└── get_organization_menu


ReplyOnPause  [fastrtc — external]
└── InterruptibleReplyOnPause  [core/stream.py]
    ├── determine_pause()    # overridden — fires on_started_talking on VAD edge
    ├── receive()            # overridden — fires on_interrupt when generator dies
    ├── _on_interrupt        # callback → Twilio clear flush
    └── _on_started_talking  # callback → immediate generator kill + queue clear


VoiceAgentStream  [core/stream.py]
├── handle_incoming_call()          # FastAPI route — returns TwiML <Connect><Stream>
├── telephone_handler()             # FastAPI WS route — Twilio media stream handler
├── _send_greeting_after_connect()  # async task — org menu fetch + greeting TTS
├── _handle_dtmf_digit()            # buffers digits, fires # shortcut
├── _process_dtmf_input()           # invokes agent with buffered digits
├── _process_return_to_menu()       # state reset + re-invocation
├── _handle_barge_in()              # Twilio clear + cancel_event set
├── _stream_agent_response()        # sentence-level TTS streaming with cancel checks
└── _send_twilio_clear()            # flushes Twilio audio buffer mid-stream
```

---

## LangGraph Enrollment State Machine

The enrollment flow is a **directed acyclic graph with conditional routing**. Every node is a pure async function that receives `ProgramApplicationState` and returns a partial state update. Routing functions inspect `current_stage` to determine the next node.

```
START
  │
  ▼
play_intro ──── (fetch org menu via get_dialer_config) ────────────────────────┐
  │                                                                             │
  ▼                                                                             │
process_language_selection ── (LLM classify language pad number) ──────────────┤
  │                                                                             │
  ▼                                                                             │
present_categories ── (read menu_categories from state, TTS numbered list) ────┤
  │                                                                             │
  ▼                                                                             │
process_category_selection ── (LLM classify_menu_selection) ───────────────────┤
  │                                                                             │
  ▼                                                                             │
present_programs ── (filter menu_programs by category)                          │
  │                                                                             │
  ▼                                                                             │
process_program_selection ── (LLM classify_menu_selection)                      │
  │                                                                             │
  ▼                                                                             │
route_to_program ── if authenticated → fetch_questions                          │
  │                 else →                                                      │
  ▼                                                                             │
fetch_eligibility_questions ── (paginated MCP calls, retry w/ tenacity)         │
  │                                                                             │
  ▼                                                                             │
ask_eligibility_question ◄──────────────────────────────┐                      │
  │                                                      │ next question        │
  ▼                                                      │                      │
validate_and_submit_answer ── (LLM validate_answer) ────┘                      │
  │  all eligibility done                                                       │
  ▼                                                                             │
collect_user_info ── (first_name → last_name → phone confirm/enter)             │
  │                                                                             │
  ▼                                                                             │
send_otp_with_eligibility ── (MCP request_otp_with_eligibility)                 │
  │                                                                             │
  ▼                                                                             │
receive_and_verify_otp ◄── retry loop (max 3 attempts)                         │
  │                                                                             │
  ▼                                                                             │
fetch_questions ── (MCP get_application_questions, visibility rules applied)    │
  │                                                                             │
  ▼                                                                             │
ask_question ◄─────────────────────────────────────────────────┐               │
  │                                                             │ next question  │
  ▼                                                             │               │
validate_and_submit_answer ── (LLM validate + confirm_answer) ─┘               │
  │  all questions done                                                         │
  ▼                                                                             │
request_consent ── (read consent text from program config)                      │
  │                                                                             │
  ▼                                                                             │
process_consent ── (LLM ConsentClassification)                                  │
  │                                                                             │
  ▼                                                                             │
finalize_application ── (MCP update_consent → MCP update_status "Submitted")   │
  │                                                                             │
  ▼                                                                             │
initiate_transfer ── (if has_transfer_option → Twilio call transfer)            │
  │                                                                             │
  ▼                                                                             │
complete ◄──────────────────────────────────────────────────────────────────────┘

   At ANY stage:  # key / "main menu" → _reset_to_main_menu() → present_categories
   At ANY stage:  max retries exceeded → initiate_transfer (care team handoff)
```

**State type:** `ProgramApplicationState` (TypedDict, ~50 fields) covers authentication tokens, menu navigation, question pagination, answer accumulation, eligibility answers, visibility-rule context, OTP UUID, language selection, STT provider lock, and call transfer metadata. LangGraph's checkpointer snapshots every state transition for debuggability.

---

## Voice Pipeline — Per-Turn Flow

```
Twilio audio frame (µ-law 8kHz)
       │
       ▼
FastRTC PCM decode (16-bit 24kHz)
       │
       ├──► DeepgramStreamingSTT (live WebSocket, parallel)
       │         SpeechStarted event ──► cancel silence timer
       │                              ──► Twilio media clear
       │                              ──► request_interrupt() ──► FastRTC generator kill
       │
       ▼ (on pause detected by FastRTC VAD)
AudioService.transcribe_audio_with_provider()
  └── locked provider: "deepgram" (Nova-2) or "whisper" (Whisper-1)
  └── language code from session state (resolved once after language selection)
       │
       ▼
LangGraphVoiceAgent._process_audio()
  └── is_return_to_menu check → _reset_to_main_menu() if triggered
  └── compiled_graph.ainvoke(state, config={thread_id})
       │
       ▼
ProgramApplicationAgent node (current_stage determines which)
  └── LLM call (structured output via llm_services)
  └── MCP API call (if data fetch / submission needed)
  └── Returns: {response: str, current_stage: str, ...state updates}
       │
       ▼
_stream_agent_response()
  └── split response into sentences
  └── for each sentence:
        AudioService.synthesize_speech_streaming(sentence, voice)
          └── TTSModel.stream_tts() → AsyncIterator[PCM chunks]
        encode PCM → µ-law → base64 → Twilio media JSON
        check cancel_event (barge-in) → break if set
```

**End-to-end latency target:** STT ~200ms + LLM first token ~400ms + TTS first chunk ~150ms = **~750ms first word heard after caller finishes speaking.**

---

## Multi-Tenant Architecture

Each Twilio phone number is mapped to an organization via `tenants.yaml` → `TenantConfig`. On every incoming call:

1. `handle_incoming_call()` extracts `To` from the Twilio POST body and embeds it as a `<Parameter name="to">` in the TwiML `<Stream>`.
2. `_TwilioStartCapture` intercepts the Twilio `start` WebSocket event and extracts `customParameters.to` and `customParameters.from`.
3. `_send_greeting_after_connect()` resolves `to_number` → calls `MCPClient.get_organization_menu()` → populates `organization_id`, `menu_categories`, `menu_programs`, `supported_languages`, `organization_introduction`, and `api_action_scripts` into the initial LangGraph state.
4. All subsequent MCP calls use the `organization_id` from state — no global tenant context.

Concurrent calls from different organizations run on separate LangGraph threads (different `thread_id`). The `MCPClient` holds **zero per-session state** — every method receives auth tokens as explicit keyword arguments. This eliminates the class of bug where a token refresh in one session overwrites the token for another.

---

## RAG & Knowledge Pipeline

The `EmbeddingServiceAgent` provides a content ingestion pipeline for organizational knowledge bases:

- **Loaders:** `TextLoader`, `WebBaseLoader` (URL scraping), `PyPDFLoader` — resolved by content type
- **Chunking:** Configurable `CHUNK_SIZE` (default 500 tokens) with `CHUNK_OVERLAP` (50 tokens) via LangChain text splitters
- **Storage:** `VectorStoreManager` wraps ChromaDB with collection-scoped upserts; supports both local persistence and remote ChromaDB HTTP client (with optional API key and SSL)
- **Retrieval:** `similarity_search` with configurable `k` (default 5); used by downstream knowledge-answer nodes to augment LLM prompts with retrieved context
- **Embedding model:** OpenAI `text-embedding-3-small` (1536-dim)

The RAG pipeline is decoupled from the voice enrollment flow — it runs as a separate agent and populates collections that any agent can query by collection name.

---

## Multilingual Architecture

Language is determined at the start of the call from the dialer config and locked for the entire session.

```
API dialer config
  └── supported_languages: [{language_code, dialer_pad_number, script_in_language}]
       │
       ▼
process_language_selection node
  └── LLM classifies caller's pad press or spoken language name
  └── Sets state.language = "es" | "ht" | "ar" | "fa" | "pt" | "fr" | "en"
  └── Resolves + locks STT provider: stt_provider_locked, stt_language_code
       │
       ▼
Per-turn transcription uses locked provider (no per-turn provider support check)

Per-turn TTS uses get_tts_voice(language, TTS_PROVIDER)
  └── If Deepgram has no voice for this language → auto-route to OpenAI TTS-1

Script constants: SNAPScripts.__getattr__ falls back en.py → English
  └── API-provided translations (api_action_scripts) take priority over locale files
  └── LLM prompt injection (language_instruction()) ensures generated text is in target language
```

**Provider capability matrix** (`config/locales/__init__.py`):

| Language | Deepgram STT | Deepgram TTS | Fallback |
|---|---|---|---|
| `en`, `es`, `pt`, `fa`, `ar`, `fr` | ✅ Nova-2 | `en`, `es` only | OpenAI TTS for others |
| `ht` (Haitian Creole) | ❌ | ❌ | Whisper STT + OpenAI TTS |

---

## Barge-In System

Two parallel detection paths give redundancy and minimize latency:

**Path 1 — FastRTC VAD (always active):**
`InterruptibleReplyOnPause.determine_pause()` monitors VAD state transitions. The moment `started_talking` flips from `False` to `True` while `responding=True`, `_close_generator()` is called synchronously, the audio queue is cleared, and `_on_started_talking` fires — all before any pause is detected. This adds ~0ms overhead on top of FastRTC's native VAD.

**Path 2 — Deepgram live WebSocket (when `DEEPGRAM_API_KEY` set):**
A `DeepgramStreamingSTT` instance is created per call on connect. Every PCM frame is fed via `asyncio.run_coroutine_threadsafe` from FastRTC's worker thread. Deepgram's `SpeechStarted` event fires at speech onset (~50–100ms) — before the VAD window closes — cancels the silence detection timer, sends a Twilio `media clear` to flush already-buffered audio, and calls `request_interrupt()` to set the force-interrupt flag. The next `receive()` call sees the flag, kills the generator, and the agent falls silent.

**Timeline:**
```
T+0ms     Caller starts speaking
T+50ms    Deepgram SpeechStarted → Twilio clear sent, interrupt flag set
T+70ms    FastRTC VAD detects talking → generator killed, queue cleared
T+150ms   Agent is fully silent, Twilio buffer flushed
```

---

## Structured LLM Outputs

Every decision that requires language understanding uses `model.with_structured_output(Schema)` rather than prompt engineering for free-form text parsing. This eliminates parsing failures and makes intent classification deterministic in schema terms.

| Function | Schema | Used By |
|---|---|---|
| `classify_menu_selection()` | `MenuSelection` | Category and program selection nodes |
| `extract_phone()` | `ExtractionResult` | OTP request, user info collection |
| `validate_answer()` | `ValidationResult` | Every application question answer |
| `classify_confirmation()` | `ConfirmationClassification` | confirm_answer, process_confirmation |
| `classify_consent()` | `ConsentClassification` | process_consent |
| `classify_caller_phone_intent()` | `CallerPhoneIntentClassification` | Caller number pre-seeding confirm flow |

All functions accept a `language` parameter which injects a directive into the system message, ensuring the LLM interprets input in the correct language regardless of locale.

---

## Authentication & Token Lifecycle

```
collect_user_info (phone, first_name, last_name)
       │
       ▼
send_otp_with_eligibility  → POST /request-otp-with-eligibility
       └── returns uuid (stored in state.uuid)
       │
       ▼
receive_and_verify_otp  → POST /verify-otp
       └── returns {access_token, refresh_token, refresh_birth, token_expires_at}
       └── all stored in ProgramApplicationState
       │
       ▼
All subsequent MCP calls pass:
  session_id=state.access_token
  refresh_token=state.refresh_token
  refresh_birth=state.refresh_birth

On any 401:
  ThriveLinkAPIClient auto-refreshes via /refresh-session
  └── new tokens returned in response._refreshed_tokens
  └── MCPClient.extract_refreshed_tokens() extracts without mutating shared state
  └── Caller merges into LangGraph state → next call uses new tokens
```

The `MCPClient` is **completely stateless** — it holds no token references. Token refresh is a functional transform: `extract_refreshed_tokens(result) → dict` which the calling node merges back into LangGraph state. This means concurrent calls from different sessions never share or overwrite auth state.

---

## Deployment

The system runs as **four Docker containers** behind an Nginx reverse proxy, deployable to any VM or container cluster:

```
┌──────────────────────────────────────────────────────┐
│                    Nginx (80/443)                     │
│         Let's Encrypt SSL + reverse proxy             │
└───┬─────────────────┬──────────────────┬─────────────┘
    │                 │                  │
    ▼                 ▼                  ▼
┌────────┐      ┌──────────┐      ┌──────────┐
│  API   │      │   RTC    │      │   Demo   │
│ :8001  │      │  :7861   │      │  :7860   │
│FastAPI │      │ FastRTC  │      │ Gradio   │
│Twilio  │      │ Gradio   │      │ Voice UI │
│ hooks  │      │ stream   │      │          │
└────────┘      └──────────┘      └──────────┘
    │
    ▼
ChromaDB (remote HTTP) or local ./data/vector_store
```

**Scaling approach:**
- Each service is stateless at the process level — LangGraph thread state is held in `InMemorySaver` (per-process) or can be migrated to `PostgresSaver` for cross-replica session persistence
- `MCPClient` holds zero mutable state — safe to scale horizontally without sticky sessions
- Twilio WebSocket connections are long-lived (call duration) — each maps to one asyncio task on one replica; load balancing at the Nginx level with WebSocket upgrade support
- TTS warm-up fires on first call connect per replica, amortizing cold-start latency

**CI/CD:** GitHub Actions workflow triggers on push to `main`; builds Docker image, pushes to registry, SSH-deploys to production VM with `docker compose pull && docker compose up -d`.

---

## Observability

- **Structured logging** via `loguru` — every LangGraph node entry/exit, every MCP call, every STT/TTS operation, every barge-in event logged with `stream_id` / `thread_id` for correlation
- **LangSmith tracing** (optional, via `LANGSMITH_TRACING=true`) — full LangGraph execution trace including node inputs/outputs, LLM calls, token counts, and timing
- **Latency audit log** — per-turn STT provider, language code, transcript length, and preview logged at `[lang-audit]` level for multilingual debugging
- **Barge-in audit log** — `[VAD]` chunk-level VAD decisions, `[speech-detect]` state transitions, `[barge-in]` callback fires — all logged with timing

---

## Technology Stack

| Layer | Technology |
|---|---|
| **LLM** | OpenAI GPT-4.1-mini (primary), GPT-4o-mini (classification) |
| **Agent Framework** | LangGraph 1.0.5 — StateGraph, conditional edges, InMemorySaver |
| **LLM Abstraction** | LangChain 1.2 — ChatOpenAI, structured output, message history |
| **Web Framework** | FastAPI 0.128 + Uvicorn |
| **Telephony** | Twilio Media Streams (WebSocket, µ-law 8kHz) |
| **Real-time Audio** | FastRTC 0.0.34 — VAD, ReplyOnPause, PCM streaming |
| **STT** | OpenAI Whisper-1 · Deepgram Nova-2 (batch + live WebSocket) |
| **TTS** | OpenAI TTS-1 (PCM streaming) · Deepgram Aura (PCM streaming) |
| **Vector DB** | ChromaDB 1.4 — local persistence + remote HTTP client |
| **Embeddings** | OpenAI text-embedding-3-small |
| **HTTP Client** | httpx 0.28 (async) |
| **Data Validation** | Pydantic v2 + pydantic-settings |
| **Retry Logic** | tenacity 9.0 — exponential back-off on MCP calls |
| **Deployment** | Docker Compose + Nginx + Let's Encrypt |
| **Tracing** | LangSmith (optional) · Loguru |
| **AWS** | boto3 — Lambda direct-invoke path (legacy MCP transport) |

---

## Key Design Decisions

**Why LangGraph over a simple loop?**
The enrollment flow has ~22 distinct stages with complex branching (retry loops, conditional eligibility, authenticated vs unauthenticated routing, mid-call state resets). LangGraph's `StateGraph` makes every branch explicit in code, makes the full state inspectable at any point, and provides checkpointing for free. A hand-rolled state machine would require the same complexity with none of the tooling.

**Why stateless MCPClient?**
Initial implementation stored tokens on the client instance. This caused a subtle bug: when two concurrent calls hit the same process and one refreshed its token, it would overwrite the other call's token reference. Moving to explicit per-call kwargs (`session_id`, `refresh_token`, `refresh_birth`) made this impossible at the type level.

**Why dual STT (Deepgram live + batch Whisper)?**
Deepgram's live WebSocket fires `SpeechStarted` ~50ms after speech onset — far earlier than any VAD window can close. This is the only way to achieve sub-200ms barge-in. But Deepgram doesn't support all 7 languages. Whisper supports all languages but is batch-only. The dual-path architecture gets the best of both: fast barge-in detection for supported languages, universal language coverage for transcription.

**Why sentence-level TTS streaming?**
Waiting for the full agent response before starting TTS would add hundreds of milliseconds of perceived latency. Splitting the response at sentence boundaries (`.`, `!`, `?`) and streaming each sentence independently cuts time-to-first-audio to ~150ms after the first sentence is generated, while still allowing graceful mid-stream cancellation on barge-in.

**Why ChromaDB for RAG?**
The embedding pipeline is primarily for organizational knowledge bases (program information, eligibility rules, FAQs). ChromaDB's client/server mode allows the same `VectorStoreManager` code to run against local persistence in development and a shared remote instance in production — switching is a config change, not a code change.
