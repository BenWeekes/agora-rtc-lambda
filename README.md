# Agora RTC Lambda Functions

AWS Lambda functions for Agora RTC token generation and ConvoAI agent management to be used in conjunction with the Agora telephony/SIP gateway.

## Table of Contents

- [Overview](#overview)
- [token_gen.py — PSTN CallLookup](#token_genpy--pstn-calllookup)
  - [How It Works](#how-it-works)
  - [Environment Variables (token_gen)](#environment-variables-token_gen)
  - [Request / Response](#request--response)
  - [Lambda Configuration (token_gen)](#lambda-configuration-token_gen)
- [launch_agent.py — ConvoAI Agent Launcher](#launch_agentpy--convoai-agent-launcher)
  - [Features](#features)
  - [Supported Providers](#supported-providers)
  - [Environment Variables (launch_agent)](#environment-variables-launch_agent)
  - [API Usage](#api-usage)
  - [UID Structure](#uid-structure)
  - [Advanced Features](#advanced-features)
  - [Profile-Based Configuration](#profile-based-configuration)
  - [Example Configurations](#example-configurations)
  - [Lambda Configuration (launch_agent)](#lambda-configuration-launch_agent)
- [Token Generation](#token-generation)
- [Troubleshooting](#troubleshooting)

---

## Overview

This repo contains two independent Lambda functions:

| Lambda | File | Purpose |
|--------|------|---------|
| **PSTN CallLookup** | `token_gen.py` | Returns an RTC token + channel for inbound PSTN calls |
| **ConvoAI Agent Launcher** | `launch_agent.py` | Generates tokens, launches/hangs-up Agora ConvoAI agents |

Both share the same v007 token generation code but serve different use cases.

---

## token_gen.py — PSTN CallLookup

Handles the Agora PSTN gateway [CallLookup webhook](https://github.com/AgoraIO-Solutions/pstn-doc#calllookup). When an inbound phone call arrives, the gateway POSTs caller information and this Lambda responds with an RTC token and channel name so the gateway can connect the caller.

### How It Works

1. PSTN gateway sends `POST {did, pin, callerid}`
2. Lambda generates a random 10-character channel name
3. Builds a v007 RTC token (or uses APP_ID if no certificate)
4. Returns the CallLookup response so the gateway joins the caller to the channel

### Environment Variables (token_gen)

#### Required
```bash
APP_ID=your_agora_app_id
```

#### Optional
```bash
APP_CERTIFICATE=your_agora_app_certificate   # omit to use APP_ID as token
USER_UID=101                                  # default: "101"
AUDIO_SCENARIO=0                              # default: "0"
WEBHOOK_URL=https://example.com/webhook       # included in response if set
SDK_OPTIONS={"key":"value"}                   # included in response if set
```

### Request / Response

**Request** (POST from PSTN gateway):
```json
{
  "did": "17177440111",
  "pin": "",
  "callerid": "1765740333"
}
```

**Response:**
```json
{
  "token": "007eJxT...",
  "uid": "101",
  "channel": "A1B2C3D4E5",
  "appid": "your_app_id",
  "audio_scenario": "0"
}
```

Optional fields `webhook_url` and `sdk_options` are included when the corresponding environment variables are set.

### Lambda Configuration (token_gen)

| Setting | Value |
|---------|-------|
| Handler | `token_gen.lambda_handler` |
| Timeout | 10 seconds |
| Memory | 128 MB |

---

## launch_agent.py — ConvoAI Agent Launcher

Launches and manages Agora Conversational AI agents with configurable TTS, STT, and LLM providers.

### Features

- **Multi-vendor TTS support**: Rime, ElevenLabs, OpenAI, Cartesia
- **Multi-vendor STT support**: Ares (Agora built-in), Deepgram
- **Flexible LLM backend**: Any OpenAI-compatible API
- **Profile-based configuration**: Support multiple agent configurations via profiles
- **Token-only mode**: Generate tokens without starting an agent
- **Agent lifecycle management**: Join and hangup capabilities
- **RTM support**: Real-time messaging integration

### Supported Providers

#### Text-to-Speech (TTS)

**1. Rime**
```
TTS_VENDOR=rime
RIME_API_KEY=your_api_key
RIME_SPEAKER=astra (default)
RIME_MODEL_ID=mistv2 (default)
RIME_LANG=eng (default)
RIME_SAMPLING_RATE=16000 (default)
RIME_SPEED_ALPHA=1.0 (default)
```

**2. ElevenLabs**
```
TTS_VENDOR=elevenlabs
TTS_KEY=your_api_key
TTS_VOICE_ID=your_voice_id
TTS_VOICE_STABILITY=1 (default: 0-1)
TTS_VOICE_SAMPLE_RATE=24000 (default)
```

**3. OpenAI**
```
TTS_VENDOR=openai
TTS_KEY=your_api_key
TTS_VOICE_ID=alloy|echo|fable|onyx|nova|shimmer
TTS_VOICE_SPEED=1.0 (default: 0.25-4.0)
```

**4. Cartesia**
```
TTS_VENDOR=cartesia
CARTESIA_API_KEY=your_api_key
CARTESIA_MODEL=sonic-3 (default)
CARTESIA_VOICE_ID=your_voice_id
CARTESIA_SAMPLE_RATE=24000 (default)
```

#### Speech-to-Text (STT/ASR)

**Ares (default)** — Agora's built-in ASR, no API key required:
```
ASR_VENDOR=ares (default)
ASR_LANGUAGE=en-US (default)
```

**Deepgram**
```
ASR_VENDOR=deepgram
DEEPGRAM_KEY=your_api_key
DEEPGRAM_MODEL=nova-3 (default)
DEEPGRAM_LANGUAGE=en (default)
```

#### Large Language Model (LLM)

Any OpenAI-compatible API:
```
LLM_URL=https://api.openai.com/v1/chat/completions
LLM_API_KEY=your_api_key
LLM_MODEL=gpt-4o-mini
```

### Environment Variables (launch_agent)

#### Required
```bash
APP_ID=your_agora_app_id
LLM_URL=your_llm_endpoint
LLM_API_KEY=your_llm_api_key
LLM_MODEL=your_model_name
```

#### Authentication (one of the following)
```bash
# Option 1: APP_CERTIFICATE (recommended)
# Generates v007 tokens for both API auth and channel join.
# API calls use "agora token=<v007_token>" authorization.
APP_CERTIFICATE=your_agora_app_certificate

# Option 2: AGENT_AUTH_HEADER (Basic auth)
# Uses Basic auth for API calls, APP_ID as channel join token.
AGENT_AUTH_HEADER=Basic <base64_key:secret>
```

If both are set, `AGENT_AUTH_HEADER` takes priority for API auth.
If neither is set, API calls will fail (APP_ID alone is not valid for API auth).

#### TTS Configuration
See provider-specific settings above.

#### STT Configuration
```bash
# Default: Ares (no API key needed)
ASR_VENDOR=ares

# Or use Deepgram:
ASR_VENDOR=deepgram
DEEPGRAM_KEY=your_deepgram_key
DEEPGRAM_MODEL=nova-3
DEEPGRAM_LANGUAGE=en
```

#### Optional Settings
```bash
# Agent Behavior
DEFAULT_PROMPT="Your custom system prompt"
DEFAULT_GREETING="hi there"
DEFAULT_FAILURE_MESSAGE="An error occurred, please try again later"
DEFAULT_MAX_HISTORY=32

# Voice Activity Detection
VAD_SILENCE_DURATION_MS=300

# Advanced Features
ENABLE_BHVS=true
ENABLE_RTM=true
ENABLE_AIVAD=true
ENABLE_ERROR_MESSAGE=true

# Agent Settings
IDLE_TIMEOUT=120

# Optional Graph ID
GRAPH_ID=your_graph_id
```

### API Usage

#### Base URL
```
https://your-lambda-url.amazonaws.com/your-stage/
```

#### 1. Launch Agent (Standard)
```bash
GET /?channel=my_channel

# Optional parameters:
# - profile: Configuration profile to use
# - prompt: Override system prompt
# - greeting: Override greeting message
# - tts_vendor: rime|elevenlabs|openai|cartesia
# - voice_id: TTS voice identifier
# - llm_model: Override LLM model
# - debug: Include debug information
```

**Response:**
```json
{
  "audio_scenario": "10",
  "token": "user_rtc_token",
  "uid": "101",
  "channel": "my_channel",
  "appid": "your_app_id",
  "user_token": {
    "token": "user_rtc_token",
    "uid": "101"
  },
  "agent_video_token": {
    "token": "agent_video_rtc_token",
    "uid": "102"
  },
  "agent": {
    "uid": "100"
  },
  "agent_rtm_uid": "100-my_channel",
  "enable_string_uid": false,
  "agent_response": {
    "status_code": 200,
    "response": "{...}",
    "success": true
  }
}
```

#### 2. Token-Only Mode (No Agent Launch)
```bash
GET /?connect=false

# Optional:
# - channel: Specify channel (auto-generated if omitted)
# - profile: Configuration profile
```

**Response:**
```json
{
  "audio_scenario": "10",
  "token": "user_rtc_token",
  "uid": "101",
  "channel": "AUTOGEN123",
  "appid": "your_app_id",
  "user_token": {
    "token": "user_rtc_token",
    "uid": "101"
  },
  "agent_video_token": {
    "token": "agent_video_rtc_token",
    "uid": "102"
  },
  "agent": {
    "uid": "100"
  },
  "agent_rtm_uid": "100-AUTOGEN123",
  "enable_string_uid": false,
  "token_generation_method": "RTC tokens with privileges",
  "agent_response": {
    "status_code": 200,
    "response": "{\"message\":\"Token-only mode...\"}",
    "success": true
  }
}
```

#### 3. Hangup Agent
```bash
GET /?hangup=true&agent_id=your_agent_id

# Required:
# - agent_id: ID of the agent to disconnect
```

**Response:**
```json
{
  "agent_response": {
    "status_code": 200,
    "response": "{...}",
    "success": true
  }
}
```

#### 4. Debug Mode
```bash
GET /?debug=true&channel=my_channel
GET /?debug=true&env_debug=true  # Show environment variables
```

### UID Structure

- **User UID**: `"101"` — For end-user RTC connection
- **Agent UID**: `"100"` — For AI agent audio
- **Agent Video UID**: `"102"` — For agent video stream (if applicable)
- **String UIDs**: Disabled by default (`enable_string_uid: false`)

### Advanced Features

#### RTM (Real-Time Messaging)
Enable text chat alongside voice:
```bash
ENABLE_RTM=true
```

#### AI VAD (Voice Activity Detection)
AI-powered voice activity detection:
```bash
ENABLE_AIVAD=true
```

#### Behaviors (BHVS)
Enable agent behavior extensions:
```bash
ENABLE_BHVS=true
```

#### Error Messages
Return error messages to users:
```bash
ENABLE_ERROR_MESSAGE=true
```

### Profile-Based Configuration

Use profile suffix to override defaults for specific use cases:

```bash
# Default configuration
LLM_MODEL=gpt-4o-mini
DEFAULT_GREETING="Hi there"

# Profile-specific (accessed via ?profile=premium)
LLM_MODEL_premium=gpt-4o
DEFAULT_GREETING_premium="Welcome, premium user"
```

### Example Configurations

#### ElevenLabs + Deepgram + OpenAI
```bash
TTS_VENDOR=elevenlabs
TTS_KEY=sk_...
TTS_VOICE_ID=cgSgspJ2msm6clMCkdW9

ASR_VENDOR=deepgram
DEEPGRAM_KEY=...
DEEPGRAM_MODEL=nova-3

LLM_URL=https://api.openai.com/v1/chat/completions
LLM_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
```

#### Rime + Deepgram + Custom LLM
```bash
TTS_VENDOR=rime
RIME_API_KEY=...
RIME_SPEAKER=astra
RIME_MODEL_ID=mistv2

ASR_VENDOR=deepgram
DEEPGRAM_KEY=...

LLM_URL=https://your-llm-endpoint.com/v1/chat/completions
LLM_API_KEY=...
LLM_MODEL=your-custom-model
```

#### Cartesia + Deepgram + OpenAI
```bash
TTS_VENDOR=cartesia
CARTESIA_API_KEY=...
CARTESIA_MODEL=sonic-3
CARTESIA_VOICE_ID=...
CARTESIA_SAMPLE_RATE=24000

ASR_VENDOR=deepgram
DEEPGRAM_KEY=...
DEEPGRAM_MODEL=nova-3

LLM_URL=https://api.openai.com/v1/chat/completions
LLM_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
```

### Lambda Configuration (launch_agent)

| Setting | Value |
|---------|-------|
| Handler | `launch_agent.lambda_handler` |
| Timeout | 30 seconds |
| Memory | 256 MB |
| CORS | Enable for browser clients |

---

## Token Generation

Both Lambdas use v007 service-based tokens.

### With APP_CERTIFICATE

Generates v007 tokens with RTC privileges:
- **RTC Service**: JOIN_CHANNEL, PUBLISH_AUDIO/VIDEO/DATA_STREAM privileges
- **RTM Service** (launch_agent only): LOGIN privilege with separate RTM UID (`{agent_uid}-{channel}`)

`token_gen.py` generates RTC-only tokens (PSTN callers don't use RTM).
`launch_agent.py` generates tokens with both RTC and RTM services.

Token expires in 24 hours.

### Without APP_CERTIFICATE

Returns `APP_ID` as token for channel join (testing mode).
For `launch_agent.py`, this requires `AGENT_AUTH_HEADER` for API authentication.

---

## Troubleshooting

### Agent doesn't join channel (launch_agent)
- Verify either `APP_CERTIFICATE` or `AGENT_AUTH_HEADER` is set
- Check `APP_ID` matches your Agora project
- Ensure Lambda has internet access (VPC configuration)

### No audio from agent (launch_agent)
- Verify TTS provider credentials
- Check TTS_VENDOR matches your configuration
- Review CloudWatch logs for TTS errors

### Speech recognition not working (launch_agent)
- Verify Deepgram API key
- Check microphone permissions on client side
- Ensure audio is being sent to channel

### PSTN caller not connecting (token_gen)
- Verify the PSTN gateway is configured to POST to your Lambda URL
- Check `APP_ID` is correct
- Review CloudWatch logs for the CallLookup request

### Token errors
- Verify `APP_CERTIFICATE` is correct (must be 32-character hex)
- Check token hasn't expired (24h default)
- Ensure UID matches between client and token

## License

See repository license.
