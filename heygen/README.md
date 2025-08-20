# HeyGen Lambda Function README

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

### Environment Variables

Set the following environment variables in your Lambda function:

| Variable | Example Value | Description |
|----------|--------------|-------------|
| `AGENT_AUTH_HEADER` | `Basic NzViOGUy****ODMwNGE2Y2E2MzYwMGRhYTkyMWQwMjk=` | Agora agent authorization header |
| `APP_ID` | `20b7c51ff4c644ab80cf5a4e646b0537` | Agora App ID |
| `APP_CERTIFICATE` | *(optional)* | Agora App Certificate |
| `DEFAULT_GREETING` | `Welcome to My Shop Live` | Initial greeting message |
| `DEFAULT_PROMPT` | `You are a live shopping assistant on myshop.live...` | System prompt for the AI |
| `HEYGEN_API_KEY` | `NGNkNmQ5****MTczMzg2OTY0MQ==` | HeyGen API key |
| `HEYGEN_AVATAR_ID` | `Wayne_20240711` | HeyGen avatar ID (optional) |
| `HEYGEN_QUALITY` | `high` | HeyGen quality setting |
| `LLM_API_KEY` | `my_secret_****` | LLM API key |
| `LLM_URL` | `https://sa-astra.agora.io:444/v1/chat/dripshop` | LLM endpoint URL |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | LLM model name |
| `LLM_PARAMS` | `{"baseURL":"https://api.groq.com/openai/v1","apiKey":"gsk_****","model":"llama-3.3-70b-versatile","stream":true}` | Additional LLM parameters (JSON) |
| `TTS_KEY` | `sk_27e975ee****8b4e569` | Text-to-speech API key |
| `TTS_VENDOR` | `elevenlabs` | TTS vendor (elevenlabs or cartesia) |
| `TTS_MODEL` | `eleven_flash_v2_5` | TTS model ID |
| `TTS_VOICE_ID` | `TX3LPaxmHKxFdv7VOQHJ` | TTS voice ID |
| `TTS_VOICE_STABILITY` | `1` | Voice stability (0-1) |
| `TTS_VOICE_SPEED` | `0.9` | Voice speed multiplier |
| `TTS_VOICE_SAMPLE_RATE` | `24000` | Sample rate in Hz |
| `ASR_VENDOR` | `deepgram` | ASR vendor |
| `ASR_LANGUAGE` | `en-US` | ASR language |
| `GRAPH_ID` | *(optional)* | Agora graph ID |

### How to Set Environment Variables in AWS Lambda

1. Navigate to your Lambda function in AWS Console
2. Go to **Configuration** tab
3. Select **Environment variables** from the left menu
4. Click **Edit**
5. Click **Add environment variable** for each variable
6. Enter the Key and Value for each variable from the table above
7. Click **Save**

### Profile Support

The function supports multiple configurations using profiles. To use profiles:
- Add environment variables with suffix: `VARIABLE_NAME_PROFILE`
- Example: `LLM_API_KEY_PROD`, `LLM_URL_STAGING`
- Call the function with `?profile=PROD` or `?profile=STAGING`

## API Usage

### Basic Request
```
GET https://your-function-url.lambda-url.region.on.aws/?channel=my-channel
```

### With Profile
```
GET https://your-function-url.lambda-url.region.on.aws/?channel=my-channel&profile=PROD
```

### Token-only Mode (no agent)
```
GET https://your-function-url.lambda-url.region.on.aws/?channel=my-channel&connect=false
```

### Custom Parameters
```
GET https://your-function-url.lambda-url.region.on.aws/?channel=my-channel&prompt=Custom%20prompt&greeting=Hello&voice_id=VOICE_ID
```

### Hangup Agent
```
GET https://your-function-url.lambda-url.region.on.aws/?hangup=true&agent_id=AGENT_ID
```

### Debug Mode
```
GET https://your-function-url.lambda-url.region.on.aws/?channel=my-channel&debug=true
```

## Response Format

### Success Response
```json
{
  "user_token": {
    "token": "007...",
    "uid": "0"
  },
  "agent_rtm_uid": "1-my-channel",
  "agent_response": {
    "status_code": 200,
    "response": "{\"agent_id\":\"...\"}",
    "success": true
  }
}
```

### Error Response
```json
{
  "error": "Missing channel parameter"
}
```

## Query Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `channel` | string | *required* | Channel name to join |
| `profile` | string | - | Configuration profile to use |
| `connect` | boolean | `true` | Whether to connect agent |
| `prompt` | string | DEFAULT_PROMPT | Custom system prompt |
| `greeting` | string | DEFAULT_GREETING | Custom greeting message |
| `voice_id` | string | TTS_VOICE_ID | TTS voice ID |
| `voice_stability` | float | `1.0` | Voice stability (0-1) |
| `voice_speed` | float | `0.9` | Voice speed multiplier |
| `voice_sample_rate` | int | `24000` | Sample rate |
| `heygen_avatar_id` | string | HEYGEN_AVATAR_ID | HeyGen avatar ID |
| `heygen_quality` | string | `high` | HeyGen quality |
| `heygen_enable` | boolean | `true` | Enable HeyGen avatar |
| `graph_id` | string | GRAPH_ID | Override graph ID |
| `hangup` | boolean | `false` | Hangup agent |
| `agent_id` | string | - | Agent ID (for hangup) |
| `debug` | boolean | `false` | Enable debug mode |

## Notes

- The 10-second timeout may be tight for some operations; consider increasing if you encounter timeouts
- The function automatically generates tokens for both RTC and RTM services
- HeyGen avatar integration creates a video stream on UID "3"
- User UID is "0" and Agent UID is "1" by default
- All API keys should be kept secure and never exposed in client-side code
