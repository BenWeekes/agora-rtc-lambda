# Launch Agent - Agora ConvoAI Lambda Function

AWS Lambda function for launching Agora ConvoAI agents with configurable TTS, STT, and LLM providers.

## Features

- **Multi-vendor TTS support**: Rime, ElevenLabs, OpenAI, Cartesia
- **Multi-vendor STT support**: Ares (Agora built-in), Deepgram
- **Flexible LLM backend**: Any OpenAI-compatible API
- **Profile-based configuration**: Support multiple agent configurations via profiles
- **Token-only mode**: Generate tokens without starting an agent
- **Agent lifecycle management**: Join and hangup capabilities
- **RTM support**: Real-time messaging integration

## Supported Providers

### Text-to-Speech (TTS)

#### 1. Rime
```
TTS_VENDOR=rime
RIME_API_KEY=your_api_key
RIME_SPEAKER=astra (default)
RIME_MODEL_ID=mistv2 (default)
RIME_LANG=eng (default)
RIME_SAMPLING_RATE=16000 (default)
RIME_SPEED_ALPHA=1.0 (default)
```

#### 2. ElevenLabs
```
TTS_VENDOR=elevenlabs
TTS_KEY=your_api_key
TTS_VOICE_ID=your_voice_id
TTS_VOICE_STABILITY=1 (default: 0-1)
TTS_VOICE_SAMPLE_RATE=24000 (default)
```

#### 3. OpenAI
```
TTS_VENDOR=openai
TTS_KEY=your_api_key
TTS_VOICE_ID=alloy|echo|fable|onyx|nova|shimmer
TTS_VOICE_SPEED=1.0 (default: 0.25-4.0)
```

#### 4. Cartesia
```
TTS_VENDOR=cartesia
CARTESIA_API_KEY=your_api_key
CARTESIA_MODEL=sonic-3 (default)
CARTESIA_VOICE_ID=your_voice_id
CARTESIA_SAMPLE_RATE=24000 (default)
```

### Speech-to-Text (STT/ASR)

#### Ares (default)
Agora's built-in ASR - no API key required:
```
ASR_VENDOR=ares (default)
ASR_LANGUAGE=en-US (default)
```

#### Deepgram
```
ASR_VENDOR=deepgram
DEEPGRAM_KEY=your_api_key
DEEPGRAM_MODEL=nova-3 (default)
DEEPGRAM_LANGUAGE=en (default)
```

### Large Language Model (LLM)

Any OpenAI-compatible API:
```
LLM_URL=https://api.openai.com/v1/chat/completions
LLM_API_KEY=your_api_key
LLM_MODEL=gpt-4o-mini
```

## Environment Variables

### Required
```bash
APP_ID=your_agora_app_id
APP_CERTIFICATE=your_agora_app_certificate
AGENT_AUTH_HEADER=your_agent_auth_header
LLM_URL=your_llm_endpoint
LLM_API_KEY=your_llm_api_key
LLM_MODEL=your_model_name
```

### TTS Configuration (choose one vendor)
See provider-specific settings above.

### STT Configuration
```bash
# Default: Ares (no API key needed)
ASR_VENDOR=ares

# Or use Deepgram:
ASR_VENDOR=deepgram
DEEPGRAM_KEY=your_deepgram_key
DEEPGRAM_MODEL=nova-3
DEEPGRAM_LANGUAGE=en
```

### Optional Settings
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

## API Usage

### Base URL
```
https://your-lambda-url.amazonaws.com/your-stage/
```

### Endpoints

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

## UID Structure

- **User UID**: `"101"` - For end-user RTC connection
- **Agent UID**: `"100"` - For AI agent audio
- **Agent Video UID**: `"102"` - For agent video stream (if applicable)
- **String UIDs**: Disabled by default (`enable_string_uid: false`)

## Advanced Features

### 1. RTM (Real-Time Messaging)
Enable text chat alongside voice:
```bash
ENABLE_RTM=true
```

### 2. AI VAD (Voice Activity Detection)
AI-powered voice activity detection:
```bash
ENABLE_AIVAD=true
```

### 3. Behaviors (BHVS)
Enable agent behavior extensions:
```bash
ENABLE_BHVS=true
```

### 4. Error Messages
Return error messages to users:
```bash
ENABLE_ERROR_MESSAGE=true
```

## Token Generation

### With APP_CERTIFICATE
Generates full RTC tokens with privileges:
- JOIN_CHANNEL (privilege 1)
- PUBLISH_AUDIO_STREAM (privilege 2)
- PUBLISH_VIDEO_STREAM (privilege 3)
- PUBLISH_DATA_STREAM (privilege 4)
- RTM_LOGIN (privilege 1000)

Token expires in 24 hours.

### Without APP_CERTIFICATE
Returns `APP_ID` as token (testing mode only).

## Lambda Configuration

### Handler
```
launch_agent.lambda_handler
```

### Recommended Settings
- **Timeout**: 30 seconds
- **Memory**: 256 MB
- **Enable CORS**: Yes (for browser clients)

### Environment Variables
Set all required environment variables in Lambda configuration.

## Example Configurations

### ElevenLabs + Deepgram + OpenAI
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

### Rime + Deepgram + Custom LLM
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

### Cartesia + Deepgram + OpenAI
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

## Troubleshooting

### Agent doesn't join channel
- Verify `AGENT_AUTH_HEADER` is correct
- Check `APP_ID` and `APP_CERTIFICATE` match your Agora project
- Ensure Lambda has internet access (VPC configuration)

### No audio from agent
- Verify TTS provider credentials
- Check TTS_VENDOR matches your configuration
- Review CloudWatch logs for TTS errors

### Speech recognition not working
- Verify Deepgram API key
- Check microphone permissions on client side
- Ensure audio is being sent to channel

### Token errors
- Verify `APP_CERTIFICATE` is correct
- Check token hasn't expired (24h default)
- Ensure UID matches between client and token

## License

See repository license.
