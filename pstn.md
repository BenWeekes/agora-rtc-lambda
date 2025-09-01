# PSTN to Agora ConvoAI

When an inbound phone number is dialled, the Agora PSTN gateway will prompt the user for a variable length PIN followed by the # key.  
The PSTN gateway can then call a URL to find out what Agora RTC channel the phone should be connected to and to trigger sending a convoAI agent into the same channel.     

The python code in [lambda_function_convoAI_pstn.py](./lambda_function_convoAI_pstn.py) can be copied to a lambda function and configured in your AWS account as shown in this demo video: 
📹 [Watch Demo Video](https://drive.google.com/file/d/13mw4jCw62K0YsgffvkCO1KKrme1GP7XB/view?usp=sharing)    

The environment variables below can be set in your lambda function to configure the agent behavior per pin.     
The lambda function URL can then be provided to Agora PSTN admin to be assigned to your inbound phone number.

## Setup Instructions

### AWS Lambda Configuration

1. **Create Lambda Function**
   - Runtime: Python 3.13
   - Architecture: x86_64 or arm64
   - Timeout: 10 seconds (minimum)
   - Memory: 512 MB (recommended)

2. **Enable Function URL**
   - Go to Configuration → Function URL
   - Click "Create function URL"
   - Auth type: **NONE** (public access)
   - CORS: **Enabled**
   - Save the function URL for later use

## SIP Call Manager Behavior

The SIP call manager will always pass the following parameters:
- **`did`**: The dialed number (e.g., `441867737373`)
- **`pin`**: The PIN entered by caller (e.g., `3344`)
- **`callerid`**: The caller's phone number (e.g., `447777786300`)

**Important**: The `pin` parameter is **mandatory** and will return a 400 error if missing. For DIDs that are not configured to ask the caller to enter a PIN, the system defaults `pin` to `0000`.

## PIN Behavior

- **PIN "0000"**: Uses default environment variables (`DEFAULT_PROMPT`, `DEFAULT_GREETING`, etc.)
- **Other PINs**: Must have both `DEFAULT_PROMPT_{PIN}` and `DEFAULT_GREETING_{PIN}` environment variables defined, otherwise returns 404 error
- **Validation**: Non-"0000" PINs are validated against existing PIN-specific environment variables

## Environment Variables

The function uses PIN-based environment variable lookup with the following logic:
- For PIN "0000": Uses default variables directly (`VARIABLE_NAME`)
- For other PINs: Requires PIN-specific variables (`VARIABLE_NAME_{PIN}`)
- Other variables fall back from `VARIABLE_NAME_{PIN}` to `VARIABLE_NAME`

| Variable | Example Value | Required | Description |
|----------|--------------|----------|-------------|
| **Agora Configuration** |
| `APP_ID` | `****ff****80cf****b0537` | ✅ | Agora App ID |
| `APP_CERTIFICATE` | *(optional)* | | Agora App Certificate (empty string if security disabled) |
| `AGENT_AUTH_HEADER` | `Basic NzViOGUy****ODMwN****MzYwMGR****QwMjk=` | ✅ | Authorization header for Agora API |
| **PIN-Specific Variables** |
| `DEFAULT_PROMPT_{PIN}` | `You are a helpful AI assistant...` | ✅ (for non-0000 PINs) | System prompt for the AI agent |
| `DEFAULT_GREETING_{PIN}` | `Hello! How can I help you today?` | ✅ (for non-0000 PINs) | Greeting message |
| **Fallback Variables (used for PIN 0000)** |
| `DEFAULT_PROMPT` | `You are a virtual companion...` | ✅ (for PIN 0000) | Fallback system prompt |
| `DEFAULT_GREETING` | `hi there` | ✅ (for PIN 0000) | Fallback greeting message |
| **LLM Configuration** |
| `LLM_URL` | `https://api.groq.com/openai/v1/chat/completions` | ✅ | LLM API endpoint |
| `LLM_API_KEY` | `gsk_your_groq_api_key` | ✅ | LLM API key |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | ✅ | LLM model name |
| `LLM_PARAMS` | `{"baseURL":"https://api.groq.com/openai/v1","apiKey":"gsk_****","model":"llama-3.3-70b-versatile","stream":true}` | | Additional LLM parameters (JSON) |
| **TTS Configuration** |
| `TTS_VENDOR` | `elevenlabs` | ✅ | Text-to-speech vendor |
| `TTS_KEY` | `your_elevenlabs_api_key` | ✅ | TTS API key |
| `TTS_MODEL` | `eleven_flash_v2_5` | ✅ | TTS model name |
| `TTS_VOICE_ID` | `21m00Tcm4TlvDq8ikWAM` | ✅ | TTS voice ID |
| `TTS_VOICE_STABILITY` | `1` | | Voice stability (0-1, default: "1") |
| `TTS_VOICE_SPEED` | `1.0` | | Voice speed multiplier (default: "1.0") |
| `TTS_VOICE_SAMPLE_RATE` | `24000` | | Sample rate in Hz (default: "24000") |
| **ASR Configuration** |
| `ASR_LANGUAGE` | `en-US` | | ASR language (default: "en-US") |
| `ASR_VENDOR` | `deepgram` | | ASR vendor (default: "deepgram") |
| **Optional** |
| `GRAPH_ID` | | | Agora graph ID |

### How to Set Environment Variables in AWS Lambda

1. Navigate to your Lambda function in AWS Console
2. Go to **Configuration** tab
3. Select **Environment variables** from the left menu
4. Click **Edit**
5. Click **Add environment variable** for each variable
6. Enter the Key and Value for each variable from the table above
7. Click **Save**

## Example Environment Variables

```bash
# Agora Configuration
APP_ID=your_agora_app_id
APP_CERTIFICATE=your_agora_app_certificate_or_empty_string
AGENT_AUTH_HEADER=Bearer your_agora_agent_api_token

# LLM Configuration (Groq example)
LLM_URL=https://api.groq.com/openai/v1/chat/completions
LLM_API_KEY=gsk_your_groq_api_key
LLM_MODEL=llama-3.3-70b-versatile

# TTS Configuration (ElevenLabs example)
TTS_VENDOR=elevenlabs
TTS_KEY=your_elevenlabs_api_key
TTS_MODEL=eleven_flash_v2_5
TTS_VOICE_ID=21m00Tcm4TlvDq8ikWAM

# Default configuration (used for PIN 0000)
DEFAULT_PROMPT=You are a virtual companion. The user can both talk and type to you and you will be sent text. Say you can hear them if asked. They can also see you as a digital human. Keep responses to around 10 to 20 words or shorter. Be upbeat and try and keep conversation going by learning more about the user.
DEFAULT_GREETING=hi there

# PIN-specific prompts and greetings (example for PIN 1234)
DEFAULT_PROMPT_1234=You are a helpful AI assistant. Keep responses short and conversational.
DEFAULT_GREETING_1234=Hello! How can I help you today?
```

## API Usage

### Basic Request (from SIP call manager)
```
POST https://your-function-url.lambda-url.region.on.aws/
Content-Type: application/json

{
  "did": "441867737373",
  "pin": "1234",
  "callerid": "447777786300"
}
```

### Token-only Mode (no agent)
```
POST https://your-function-url.lambda-url.region.on.aws/
Content-Type: application/json

{
  "did": "441867737373", 
  "pin": "1234",
  "callerid": "447777786300",
  "connect": "false"
}
```

### Hangup Agent
```
POST https://your-function-url.lambda-url.region.on.aws/
Content-Type: application/json

{
  "hangup": "true",
  "agent_id": "AGENT_ID",
  "pin": "1234"
}
```

### Debug Mode
```
POST https://your-function-url.lambda-url.region.on.aws/
Content-Type: application/json

{
  "did": "441867737373",
  "pin": "1234", 
  "callerid": "447777786300",
  "debug": "true"
}
```

## Response Format

### Success Response
```json
{
  "token": "007...",
  "uid": "user",
  "channel": "ABCD123456",
  "agent_id": "agent_12345"
}
```

### Token-only Response
```json
{
  "token": "007...",
  "uid": "user", 
  "channel": "ABCD123456",
  "agent_id": null,
  "agent_response": {
    "status_code": 200,
    "response": "{\"message\":\"Token-only mode: user token generated successfully\"}",
    "success": true
  }
}
```

### Error Response
```json
{
  "error": "Missing pin parameter"
}
```

## Agora PSTN .pools configuration example entry
```bash
pinlookup_4412345678=https://zeebonggpkkllkks.lambda-url.us-east-1.on.aws
```

## Notes

- Uses Token Version 007 with service-based architecture (ServiceRtc, ServiceRtm)
- The 10-second timeout may be tight for some operations; consider increasing if you encounter timeouts
- The function automatically generates tokens for both RTC and RTM services
- User UID is "user" and Agent UID is "agent" by default
- All API keys should be kept secure and never exposed in client-side code
- Channels are randomly generated 10-character strings (uppercase letters and numbers)

📚 [Agora PSTN API Documentation](https://github.com/AgoraIO-Solutions/pstn-doc)
