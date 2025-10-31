import json
import hmac
from hashlib import sha256
import base64
import struct
from zlib import crc32
import secrets
import time
import random
import string
from collections import OrderedDict
import urllib.parse
import http.client
import os


# Helper function to get environment variables with profile support
def get_env_var(var_name, profile=None, default_value=None):
    """
    Gets an environment variable with profile support.
    If profile is provided, it first checks for VAR_NAME_PROFILE.
    If that doesn't exist, it falls back to VAR_NAME.
    If that doesn't exist, it returns the default_value.
    
    Args:
        var_name: The environment variable name
        profile: Optional profile suffix
        default_value: Default value if neither variable exists
        
    Returns:
        The value of the environment variable or default_value
    """
    if profile:
        # Use exact case as provided
        profiled_var_name = f"{var_name}_{profile}"
        profiled_value = os.environ.get(profiled_var_name)
        if profiled_value is not None:
            return profiled_value
    
    # Fall back to standard variable
    value = os.environ.get(var_name)
    if value is not None:
        return value
    
    # Fall back to default value
    return default_value


# Constants for Agora
def initialize_constants(profile=None):
    """
    Initialize all constants with profile support
    
    Args:
        profile: Optional profile suffix for environment variables
        
    Returns:
        Dictionary of constants
    """
    constants = {
        "APP_ID": get_env_var('APP_ID', profile),
        "APP_CERTIFICATE": get_env_var('APP_CERTIFICATE', profile, ''),
        "AGENT_AUTH_HEADER": get_env_var('AGENT_AUTH_HEADER', profile),
        "AGENT_API_BASE_URL": "https://api.agora.io/api/conversational-ai-agent/v2/projects",
        
        # Optional graph ID - omitted if not present
        "GRAPH_ID": get_env_var('GRAPH_ID', profile),

        # Enable string UID setting
        "ENABLE_STRING_UID": False,

        # Fixed UIDs as strings
        "AGENT_UID": "100",
        "USER_UID": "101",
        "AGENT_VIDEO_UID": "102",
        
        # Constants for token generation
        "VERSION_LENGTH": 3,
        "APP_ID_LENGTH": 32,
        
        # Token expiration (in seconds)
        "TOKEN_EXPIRE": 24 * 3600,  # 24 hours
        "PRIVILEGE_EXPIRE": 24 * 3600,  # 24 hours
        
        # Define LLM settings
        "LLM_URL": get_env_var('LLM_URL', profile),
        "LLM_API_KEY": get_env_var('LLM_API_KEY', profile),
        "LLM_MODEL": get_env_var('LLM_MODEL', profile),
        "LLM_PARAMS": get_env_var('LLM_PARAMS', profile),
        
        # Define TTS settings
        "TTS_VENDOR": get_env_var('TTS_VENDOR', profile),
        "TTS_KEY": get_env_var('TTS_KEY', profile),
        "TTS_MODEL": get_env_var('TTS_MODEL', profile),
        "TTS_VOICE_ID": get_env_var('TTS_VOICE_ID', profile),
        "TTS_VOICE_STABILITY": get_env_var('TTS_VOICE_STABILITY', profile, "1"),
        "TTS_VOICE_SPEED": get_env_var('TTS_VOICE_SPEED', profile, "0.9"),
        "TTS_VOICE_SAMPLE_RATE": get_env_var('TTS_VOICE_SAMPLE_RATE', profile, "24000"),
        "TTS_VOICE_INSTRUCTIONS": get_env_var('TTS_VOICE_INSTRUCTIONS', profile, 
            "Please use standard American English, natural tone, moderate pace, and steady intonation"),
        
        # Rime TTS specific settings
        "RIME_API_KEY": get_env_var('RIME_API_KEY', profile),
        "RIME_SPEAKER": get_env_var('RIME_SPEAKER', profile, "astra"),
        "RIME_MODEL_ID": get_env_var('RIME_MODEL_ID', profile, "mistv2"),
        "RIME_LANG": get_env_var('RIME_LANG', profile, "eng"),
        "RIME_SAMPLING_RATE": get_env_var('RIME_SAMPLING_RATE', profile, "16000"),
        "RIME_SPEED_ALPHA": get_env_var('RIME_SPEED_ALPHA', profile, "1.0"),
        
        # Define ASR settings
        "ASR_LANGUAGE": get_env_var('ASR_LANGUAGE', profile, "en-US"),
        "ASR_VENDOR": get_env_var('ASR_VENDOR', profile, "deepgram"),
        "DEEPGRAM_URL": get_env_var('DEEPGRAM_URL', profile, "wss://api.deepgram.com/v1/listen"),
        "DEEPGRAM_KEY": get_env_var('DEEPGRAM_KEY', profile),
        "DEEPGRAM_MODEL": get_env_var('DEEPGRAM_MODEL', profile, "nova-3"),
        "DEEPGRAM_LANGUAGE": get_env_var('DEEPGRAM_LANGUAGE', profile, "en"),
        
        # VAD settings
        "VAD_SILENCE_DURATION_MS": get_env_var('VAD_SILENCE_DURATION_MS', profile, "300"),
        
        # Advanced features
        "ENABLE_BHVS": get_env_var('ENABLE_BHVS', profile, "true"),
        "ENABLE_RTM": get_env_var('ENABLE_RTM', profile, "false"),
        "ENABLE_AIVAD": get_env_var('ENABLE_AIVAD', profile, "true"),
        
        # Agent settings
        "IDLE_TIMEOUT": get_env_var('IDLE_TIMEOUT', profile, "15"),
        "ENABLE_ERROR_MESSAGE": get_env_var('ENABLE_ERROR_MESSAGE', profile, "true"),
        
        # Default values for prompt and greeting
        "DEFAULT_PROMPT": get_env_var('DEFAULT_PROMPT', profile, 
            "You are a virtual companion. The user can both talk and type to you and you will be sent text. "
            "Say you can hear them if asked. They can also see you as a digital human. "
            "Keep responses to around 10 to 20 words or shorter. Be upbeat and try and keep conversation "
            "going by learning more about the user. "),
        "DEFAULT_GREETING": get_env_var('DEFAULT_GREETING', profile, "hi there"),
        "DEFAULT_FAILURE_MESSAGE": get_env_var('DEFAULT_FAILURE_MESSAGE', profile, "An error occurred, please try again later"),
        "DEFAULT_MAX_HISTORY": get_env_var('DEFAULT_MAX_HISTORY', profile, "32"),
        
        # Controller endpoint
        "CONTROLLER_ENDPOINT": get_env_var('CONTROLLER_ENDPOINT', profile, "wss:wvc-ln-01.trulience.com")
    }
    
    return constants


def generate_random_channel(length=10):
    """
    Generates a random channel name with uppercase letters and numbers

    Args:
        length: Length of the channel name (default: 10)

    Returns:
        Random channel name string
    """
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))


def lambda_handler(event, context):
    """
    Lambda handler function that processes the incoming request, generates tokens,
    and sends an agent to the channel or handles hangup requests.
    """
    # Check if queryStringParameters exists, use empty dict if not
    query_params = event.get('queryStringParameters') or {}
    
    # Get the optional profile parameter
    profile = query_params.get('profile')
    
    # Initialize constants with profile
    constants = initialize_constants(profile)
    
    # Add environment variable debugging if in debug mode
    if 'debug' in query_params:
        # Get all environment variables for debugging
        env_vars = {}
        for key, value in os.environ.items():
            # Mask sensitive values like API keys, only show first/last few characters
            if 'key' in key.lower() or 'secret' in key.lower() or 'token' in key.lower() or 'certificate' in key.lower():
                if value and len(value) > 10:
                    # Show first 4 and last 4 characters with *** in between
                    masked_value = value[:4] + "****" + value[-4:]
                    env_vars[key] = masked_value
                else:
                    env_vars[key] = "[MASKED]"
            else:
                env_vars[key] = value
                
        # Return debug info about environment variables and profile
        if 'env_debug' in query_params:
            return json_response(200, {
                "profile_requested": profile,
                "llm_params_expected_name": f"LLM_PARAMS_{profile}" if profile else "LLM_PARAMS",
                "llm_params_value": constants["LLM_PARAMS"],
                "environment_variables": env_vars,
                "all_constants": constants
            })
    
    # Check for hangup request
    if 'hangup' in query_params and query_params['hangup'].lower() == 'true':
        # Hangup flow
        if 'agent_id' not in query_params:
            return json_response(400, {"error": "Missing agent_id parameter for hangup"})
        
        agent_id = query_params['agent_id']
        hangup_response = hangup_agent(agent_id, constants)
        
        return json_response(200, {
            "agent_response": hangup_response,
            "controller_endpoint": constants["CONTROLLER_ENDPOINT"]
        })
    
    # Normal join flow or token-only flow
    # Get channel from query parameters, or generate random one if not provided
    if 'channel' in query_params and query_params['channel']:
        channel = query_params['channel']
    else:
        channel = generate_random_channel(10)
    
    # Check if this is a connect=false request (token-only mode)
    connect_param = query_params.get('connect', 'true').lower()
    token_only_mode = connect_param == 'false'
    
    # Check if APP_CERTIFICATE exists - only generate RTC tokens if it does
    has_certificate = constants["APP_CERTIFICATE"] and constants["APP_CERTIFICATE"].strip() != ''
    
    if has_certificate:
        # Get token for user with RTC and RTM capabilities
        user_token_data = build_token_with_rtm(channel, constants["USER_UID"], constants)
        
        # Get token for agent video with RTC and RTM capabilities
        agent_video_token_data = build_token_with_rtm(channel, constants["AGENT_VIDEO_UID"], constants)
    else:
        # No certificate - use APP_ID as token (testing mode)
        user_token_data = {
            "token": constants["APP_ID"],
            "uid": constants["USER_UID"]
        }
        agent_video_token_data = {
            "token": constants["APP_ID"],
            "uid": constants["AGENT_VIDEO_UID"]
        }
    
    # If connect=false, return only the user token and agent video token without starting the agent
    if token_only_mode:
        return json_response(200, {
            "audio_scenario": "10",
            "token": user_token_data["token"],
            "uid": user_token_data["uid"],
            "channel": channel,
            "appid": constants["APP_ID"],
            "user_token": user_token_data,
            "agent_video_token": agent_video_token_data,
            "agent": {
                "uid": constants["AGENT_UID"]
            },
            "enable_string_uid": constants["ENABLE_STRING_UID"],
            "controller_endpoint": constants["CONTROLLER_ENDPOINT"],
            "token_generation_method": "RTC tokens with privileges" if has_certificate else "APP_ID only (no APP_CERTIFICATE)",
            "agent_response": {
                "status_code": 200,
                "response": json.dumps({
                    "message": "Token-only mode: user token and agent video token generated successfully",
                    "mode": "token_only",
                    "connect": False
                }),
                "success": True
            }
        })
    
    # Normal connect=true flow: create and send agent
    # Get optional prompt, greeting, voice_id, tts_vendor, and debug parameters
    prompt = query_params.get('prompt', constants["DEFAULT_PROMPT"])
    greeting = query_params.get('greeting', constants["DEFAULT_GREETING"])
    failure_message = query_params.get('failure_message', constants["DEFAULT_FAILURE_MESSAGE"])
    max_history = query_params.get('max_history', constants["DEFAULT_MAX_HISTORY"])
    
    # Get TTS vendor from query params or use default
    tts_vendor = query_params.get('tts_vendor', constants["TTS_VENDOR"])
    
    # Get voice parameters based on TTS vendor
    if tts_vendor == "rime":
        # Rime TTS parameters
        rime_api_key = query_params.get('rime_api_key', constants["RIME_API_KEY"])
        rime_speaker = query_params.get('rime_speaker', constants["RIME_SPEAKER"])
        rime_model_id = query_params.get('rime_model_id', constants["RIME_MODEL_ID"])
        rime_lang = query_params.get('rime_lang', constants["RIME_LANG"])
        rime_sampling_rate = query_params.get('rime_sampling_rate', constants["RIME_SAMPLING_RATE"])
        rime_speed_alpha = query_params.get('rime_speed_alpha', constants["RIME_SPEED_ALPHA"])
    else:
        # Other TTS parameters (existing support)
        voice_id = query_params.get('voice_id', constants["TTS_VOICE_ID"])
        voice_stability = query_params.get('voice_stability', constants["TTS_VOICE_STABILITY"])
        voice_speed = query_params.get('voice_speed', constants["TTS_VOICE_SPEED"])
        voice_sample_rate = query_params.get('voice_sample_rate', constants["TTS_VOICE_SAMPLE_RATE"])
        voice_instructions = query_params.get('voice_instructions', constants["TTS_VOICE_INSTRUCTIONS"])
    
    # Get LLM parameters
    llm_url = query_params.get('llm_url', constants["LLM_URL"])
    llm_api_key = query_params.get('llm_api_key', constants["LLM_API_KEY"])
    llm_model = query_params.get('llm_model', constants["LLM_MODEL"])
    
    # Get ASR parameters
    asr_vendor = query_params.get('asr_vendor', constants["ASR_VENDOR"])
    deepgram_url = query_params.get('deepgram_url', constants["DEEPGRAM_URL"])
    deepgram_key = query_params.get('deepgram_key', constants["DEEPGRAM_KEY"])
    deepgram_model = query_params.get('deepgram_model', constants["DEEPGRAM_MODEL"])
    deepgram_language = query_params.get('deepgram_language', constants["DEEPGRAM_LANGUAGE"])
    
    # Get VAD parameters
    vad_silence_duration = query_params.get('vad_silence_duration_ms', constants["VAD_SILENCE_DURATION_MS"])
    
    # Get advanced features
    enable_bhvs = query_params.get('enable_bhvs', constants["ENABLE_BHVS"]).lower() == "true"
    enable_rtm = query_params.get('enable_rtm', constants["ENABLE_RTM"]).lower() == "true"
    enable_aivad = query_params.get('enable_aivad', constants["ENABLE_AIVAD"]).lower() == "true"
    
    # Get agent settings
    idle_timeout = query_params.get('idle_timeout', constants["IDLE_TIMEOUT"])
    enable_error_message = query_params.get('enable_error_message', constants["ENABLE_ERROR_MESSAGE"]).lower() == "true"
    
    # FIXED: For ConvoAI agent, use APP_ID as the token (not a generated RTC token)
    agent_token = constants["APP_ID"]
    
    # Create the agent payload
    agent_payload = create_agent_payload(
        channel=channel,
        agent_token=agent_token,  # Using APP_ID as token
        prompt=prompt,
        greeting=greeting,
        failure_message=failure_message,
        max_history=max_history,
        tts_vendor=tts_vendor,
        llm_url=llm_url,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
        asr_vendor=asr_vendor,
        deepgram_url=deepgram_url,
        deepgram_key=deepgram_key,
        deepgram_model=deepgram_model,
        deepgram_language=deepgram_language,
        vad_silence_duration=vad_silence_duration,
        enable_bhvs=enable_bhvs,
        enable_rtm=enable_rtm,
        enable_aivad=enable_aivad,
        idle_timeout=idle_timeout,
        enable_error_message=enable_error_message,
        constants=constants,
        # Rime TTS parameters
        rime_api_key=rime_api_key if tts_vendor == "rime" else None,
        rime_speaker=rime_speaker if tts_vendor == "rime" else None,
        rime_model_id=rime_model_id if tts_vendor == "rime" else None,
        rime_lang=rime_lang if tts_vendor == "rime" else None,
        rime_sampling_rate=rime_sampling_rate if tts_vendor == "rime" else None,
        rime_speed_alpha=rime_speed_alpha if tts_vendor == "rime" else None,
        # Other TTS parameters
        voice_id=voice_id if tts_vendor != "rime" else None,
        voice_stability=voice_stability if tts_vendor != "rime" else None,
        voice_speed=voice_speed if tts_vendor != "rime" else None,
        voice_sample_rate=voice_sample_rate if tts_vendor != "rime" else None,
        voice_instructions=voice_instructions if tts_vendor != "rime" else None
    )
    
    # Send the agent to the channel
    agent_response = send_agent_to_channel(channel, agent_payload, constants)
    
    # Include debug info if requested
    debug_info = None
    if 'debug' in query_params:
        debug_info = {
            "agent_payload": agent_payload,
            "channel": channel,
            "token_used": "APP_ID (not RTC token)",
            "api_url": f"{constants['AGENT_API_BASE_URL']}/{constants['APP_ID']}/join",
            "user_tokens_method": "RTC tokens with privileges" if has_certificate else "APP_ID only (no certificate)",
            "has_app_certificate": has_certificate
        }
    
    # Return response with all tokens and agent status
    response_data = {
        "audio_scenario": "10",
        "token": user_token_data["token"],
        "uid": user_token_data["uid"],
        "channel": channel,
        "appid": constants["APP_ID"],
        "user_token": user_token_data,
        "agent_video_token": agent_video_token_data,
        "agent": {
            "uid": constants["AGENT_UID"]
        },
        "enable_string_uid": constants["ENABLE_STRING_UID"],
        "controller_endpoint": constants["CONTROLLER_ENDPOINT"],
        "agent_response": agent_response
    }

    if debug_info:
        response_data["debug"] = debug_info

    return json_response(200, response_data)


def json_response(status_code, body):
    """Helper function to create a JSON response"""
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json"
        },
        "body": json.dumps(body)
    }


def hangup_agent(agent_id, constants):
    """
    Sends a hangup request to the Agora API to disconnect the agent.
    
    Args:
        agent_id: The unique identifier for the agent to hang up
        constants: Dictionary of constants
    
    Returns:
        Dictionary with the status code, response body, and success flag
    """
    # Construct the hangup API URL
    hangup_api_url = f"{constants['AGENT_API_BASE_URL']}/{constants['APP_ID']}/stop/{agent_id}"
    
    # Parse the URL to get host and path
    url_parts = urllib.parse.urlparse(hangup_api_url)
    host = url_parts.netloc
    path = url_parts.path
    
    # Use http.client directly with timeout
    conn = http.client.HTTPSConnection(host, timeout=30)  # 30 second timeout
    
    headers = {
        "Authorization": constants["AGENT_AUTH_HEADER"]
    }
    
    conn.request("DELETE", path, headers=headers)
    
    # Get the response
    response = conn.getresponse()
    status_code = response.status
    response_text = response.read().decode('utf-8')
    
    conn.close()
    
    # Return a dictionary with the response details
    return {
        "status_code": status_code,
        "response": response_text,
        "success": status_code == 200
    }


def build_token_with_rtm(channel, uid, constants):
    """
    Builds a token with both RTC and RTM capabilities
    
    Args:
        channel: The channel name
        uid: The user ID (string)
        constants: Dictionary of constants
    
    Returns:
        Dictionary with token and uid
    """
    # Create token with RTC capabilities
    token = AccessToken(constants["APP_ID"], constants["APP_CERTIFICATE"], channel, uid)
    
    # Add RTC privileges
    token.addPrivilege(1, int(time.time()) + constants["PRIVILEGE_EXPIRE"])  # JOIN_CHANNEL
    token.addPrivilege(2, int(time.time()) + constants["PRIVILEGE_EXPIRE"])  # PUBLISH_AUDIO_STREAM
    token.addPrivilege(3, int(time.time()) + constants["PRIVILEGE_EXPIRE"])  # PUBLISH_VIDEO_STREAM
    token.addPrivilege(4, int(time.time()) + constants["PRIVILEGE_EXPIRE"])  # PUBLISH_DATA_STREAM
    
    # Add RTM privileges
    token.addPrivilege(1000, int(time.time()) + constants["PRIVILEGE_EXPIRE"])  # RTM_LOGIN
    
    built_token = token.build()
    
    return {
        "token": built_token,
        "uid": uid
    }


def create_agent_payload(channel, agent_token, prompt, greeting, failure_message, max_history,
                        tts_vendor, llm_url, llm_api_key, llm_model,
                        asr_vendor, deepgram_url, deepgram_key, deepgram_model, deepgram_language,
                        vad_silence_duration, enable_bhvs, enable_rtm, enable_aivad,
                        idle_timeout, enable_error_message, constants,
                        rime_api_key=None, rime_speaker=None, rime_model_id=None, 
                        rime_lang=None, rime_sampling_rate=None, rime_speed_alpha=None,
                        voice_id=None, voice_stability=None, voice_speed=None,
                        voice_sample_rate=None, voice_instructions=None):
    """
    Creates the complete agent payload in Agora convoAI format
    
    Args:
        channel: The channel name
        agent_token: The agent's token (should be APP_ID for ConvoAI)
        prompt: The system prompt for the LLM
        greeting: The greeting message
        failure_message: The failure message
        max_history: Maximum conversation history
        tts_vendor: TTS vendor (rime, elevenlabs, openai, etc)
        llm_url: LLM API URL
        llm_api_key: LLM API key
        llm_model: LLM model name
        asr_vendor: ASR vendor (deepgram, etc)
        deepgram_url: Deepgram WebSocket URL
        deepgram_key: Deepgram API key
        deepgram_model: Deepgram model
        deepgram_language: Deepgram language
        vad_silence_duration: VAD silence duration in ms
        enable_bhvs: Enable behaviors
        enable_rtm: Enable RTM
        enable_aivad: Enable AI VAD
        idle_timeout: Idle timeout in seconds
        enable_error_message: Enable error messages
        constants: Dictionary of constants
        rime_api_key: Rime API key (if using Rime TTS)
        rime_speaker: Rime speaker (if using Rime TTS)
        rime_model_id: Rime model ID (if using Rime TTS)
        rime_lang: Rime language (if using Rime TTS)
        rime_sampling_rate: Rime sampling rate (if using Rime TTS)
        rime_speed_alpha: Rime speed alpha (if using Rime TTS)
        voice_id: Voice ID for other TTS vendors
        voice_stability: Voice stability for other TTS vendors
        voice_speed: Voice speed for other TTS vendors
        voice_sample_rate: Voice sample rate for other TTS vendors
        voice_instructions: Voice instructions for other TTS vendors
    
    Returns:
        OrderedDict containing the complete agent payload
    """
    
    # Build TTS configuration based on vendor
    tts_config = {
        "vendor": tts_vendor
    }
    
    if tts_vendor == "rime":
        tts_config["skip_patterns"] = [5]
        tts_config["params"] = {
            "api_key": rime_api_key,
            "speaker": rime_speaker,
            "modelId": rime_model_id,
            "lang": rime_lang,
            "samplingRate": int(rime_sampling_rate),
            "speedAlpha": float(rime_speed_alpha)
        }
    elif tts_vendor == "elevenlabs":
        tts_config["params"] = {
            "voice_id": voice_id,
            "model_id": "eleven_turbo_v2_5",
            "optimize_streaming_latency": 3,
            "stability": float(voice_stability),
            "output_format": f"pcm_{voice_sample_rate}"
        }
    elif tts_vendor == "openai":
        tts_config["params"] = {
            "model": "tts-1",
            "voice": voice_id,
            "response_format": "pcm",
            "speed": float(voice_speed)
        }
    
    # Build ASR configuration
    asr_config = {
        "vendor": asr_vendor
    }
    
    if asr_vendor == "deepgram":
        asr_config["params"] = {
            "url": deepgram_url,
            "key": deepgram_key,
            "model": deepgram_model,
            "language": deepgram_language
        }
    
    # Build LLM configuration
    llm_config = {
        "url": llm_url,
        "api_key": llm_api_key,
        "system_messages": [
            {
                "role": "system",
                "content": prompt
            }
        ],
        "greeting_message": greeting,
        "failure_message": failure_message,
        "max_history": int(max_history),
        "params": {
            "model": llm_model
        }
    }
    
    # Build the complete payload in the Agora convoAI format
    payload_items = []
    payload_items.append(("name", channel))
    payload_items.append(("properties", OrderedDict([
        ("parameters", {
            "enable_error_message": enable_error_message
        }),
        ("channel", channel),
        ("token", agent_token),  # This should be APP_ID for ConvoAI
        ("agent_rtc_uid", constants["AGENT_UID"]),
        ("remote_rtc_uids", ["*"]),
        ("advanced_features", {
            "enable_bhvs": enable_bhvs,
            "enable_rtm": enable_rtm,
            "enable_aivad": enable_aivad
        }),
        ("enable_string_uid", False),
        ("idle_timeout", int(idle_timeout)),
        ("llm", llm_config),
        ("vad", {
            "silence_duration_ms": int(vad_silence_duration)
        }),
        ("asr", asr_config),
        ("tts", tts_config)
    ])))
    
    # Convert to OrderedDict to preserve the order
    agent_payload = OrderedDict(payload_items)
    
    return agent_payload


def send_agent_to_channel(channel, agent_payload, constants):
    """
    Sends an agent to the specified Agora RTC channel by calling the REST API
    
    Args:
        channel: The channel name
        agent_payload: The complete agent payload to send
        constants: Dictionary of constants
    
    Returns:
        Dictionary with the status code, response body, and success flag
    """
    # Construct the agent API URL
    agent_api_url = f"{constants['AGENT_API_BASE_URL']}/{constants['APP_ID']}/join"
    
    # Parse the URL to get host and path
    url_parts = urllib.parse.urlparse(agent_api_url)
    host = url_parts.netloc
    path = url_parts.path
    
    # Use http.client directly with timeout
    conn = http.client.HTTPSConnection(host, timeout=30)  # 30 second timeout
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": constants["AGENT_AUTH_HEADER"]
    }
    
    # Convert the payload to JSON
    payload_json = json.dumps(agent_payload, indent=2)
    
    # Log the request for debugging
    print(f"Sending agent to Agora ConvoAI:")
    print(f"URL: {agent_api_url}")
    print(f"Payload: {payload_json}")
    
    conn.request("POST", path, payload_json, headers)
    
    # Get the response
    response = conn.getresponse()
    status_code = response.status
    response_text = response.read().decode('utf-8')
    
    print(f"Response status: {status_code}")
    print(f"Response body: {response_text}")
    
    conn.close()
    
    # Return a dictionary with the response details
    return {
        "status_code": status_code,
        "response": response_text,
        "success": status_code == 200
    }


def getVersion():
    """Returns the version string for the token"""
    return '006'


def packUint16(x):
    """Packs an unsigned 16-bit integer"""
    return struct.pack('<H', int(x))


def packUint32(x):
    """Packs an unsigned 32-bit integer"""
    return struct.pack('<I', int(x))


def packInt32(x):
    """Packs a signed 32-bit integer"""
    return struct.pack('<i', int(x))


def packString(string):
    """Packs a string with its length prefix"""
    return packUint16(len(string)) + string


def packMap(m):
    """Packs a map of key-value pairs where values are strings"""
    ret = packUint16(len(list(m.items())))
    for k, v in list(m.items()):
        ret += packUint16(k) + packString(v)
    return ret


def packMapUint32(m):
    """Packs a map of key-value pairs where values are uint32"""
    ret = packUint16(len(list(m.items())))
    for k, v in list(m.items()):
        ret += packUint16(k) + packUint32(v)
    return ret


class ReadByteBuffer:
    """Helper class for unpacking binary data"""
    def __init__(self, bytes):
        self.buffer = bytes
        self.position = 0

    def unPackUint16(self):
        len = struct.calcsize('H')
        buff = self.buffer[self.position: self.position + len]
        ret = struct.unpack('<H', buff)[0]
        self.position += len
        return ret

    def unPackUint32(self):
        len = struct.calcsize('I')
        buff = self.buffer[self.position: self.position + len]
        ret = struct.unpack('<I', buff)[0]
        self.position += len
        return ret

    def unPackString(self):
        strlen = self.unPackUint16()
        buff = self.buffer[self.position: self.position + strlen]
        ret = struct.unpack('<' + str(strlen) + 's', buff)[0]
        self.position += strlen
        return ret

    def unPackMapUint32(self):
        messages = {}
        maplen = self.unPackUint16()

        for index in range(maplen):
            key = self.unPackUint16()
            value = self.unPackUint32()
            messages[key] = value
        return messages


def unPackContent(buff):
    """Unpacks the content portion of a token"""
    readbuf = ReadByteBuffer(buff)
    signature = readbuf.unPackString()
    crc_channel_name = readbuf.unPackUint32()
    crc_uid = readbuf.unPackUint32()
    m = readbuf.unPackString()
    return signature, crc_channel_name, crc_uid, m


def unPackMessages(buff):
    """Unpacks the messages portion of a token"""
    readbuf = ReadByteBuffer(buff)
    salt = readbuf.unPackUint32()
    ts = readbuf.unPackUint32()
    messages = readbuf.unPackMapUint32()
    return salt, ts, messages


class AccessToken:
    """
    Class for building and parsing Agora access tokens
    """
    def __init__(self, appID='', appCertificate='', channelName='', uid=''):
        self.appID = appID
        self.appCertificate = appCertificate
        self.channelName = channelName
        self.ts = int(time.time()) + 24 * 3600
        self.salt = secrets.SystemRandom().randint(1, 99999999)
        self.messages = {}
        if uid == 0 or uid == "":
            self.uidStr = ""
        else:
            self.uidStr = str(uid)

    def addPrivilege(self, privilege, expireTimestamp):
        """Adds a privilege to the token"""
        self.messages[privilege] = expireTimestamp

    def fromString(self, originToken):
        """Parses a token from a string"""
        try:
            dk6version = getVersion()
            originVersion = originToken[:VERSION_LENGTH]
            if (originVersion != dk6version):
                return False

            originAppID = originToken[VERSION_LENGTH:(VERSION_LENGTH + APP_ID_LENGTH)]
            originContent = originToken[(VERSION_LENGTH + APP_ID_LENGTH):]
            originContentDecoded = base64.b64decode(originContent)
            signature, crc_channel_name, crc_uid, m = unPackContent(originContentDecoded)
            self.salt, self.ts, self.messages = unPackMessages(m)

        except Exception as e:
            print("error:", str(e))
            return False

        return True

    def build(self):
        """Builds a token string"""
        self.messages = OrderedDict(sorted(iter(self.messages.items()), key=lambda x: int(x[0])))
        m = packUint32(self.salt) + packUint32(self.ts) \
            + packMapUint32(self.messages)

        val = self.appID.encode('utf-8') + self.channelName.encode('utf-8') + self.uidStr.encode('utf-8') + m
        signature = hmac.new(self.appCertificate.encode('utf-8'), val, sha256).digest()
        crc_channel_name = crc32(self.channelName.encode('utf-8')) & 0xffffffff
        crc_uid = crc32(self.uidStr.encode('utf-8')) & 0xffffffff
        content = packString(signature) \
                  + packUint32(crc_channel_name) \
                  + packUint32(crc_uid) \
                  + packString(m)

        version = getVersion()
        ret = version + self.appID + base64.b64encode(content).decode('utf-8')
        return ret

