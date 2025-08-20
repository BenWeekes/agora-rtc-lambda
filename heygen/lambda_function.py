import json
import hmac
from hashlib import sha256
import base64
import struct
from zlib import crc32
import zlib
import secrets
import time
import random
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
        
        # Fixed UIDs as strings
        "AGENT_UID": "1",
        "USER_UID": "0", # won't subscribe audio as 0 replaced with random uid
        "AGENT_VIDEO_UID": "3",  # Updated for HeyGen
        
        # Constants for token generation
        "VERSION_LENGTH": 3,
        "APP_ID_LENGTH": 32,
        
        # Token expiration (in seconds)
        "TOKEN_EXPIRE": 24 * 3600,  # 24 hours
        "PRIVILEGE_EXPIRE": 24 * 3600,  # 24 hours
        
        # Define LLM settings
        "LLM_URL": get_env_var('LLM_URL', profile, "https://api.groq.com/openai/v1/chat/completions"),
        "LLM_API_KEY": get_env_var('LLM_API_KEY', profile),
        "LLM_MODEL": get_env_var('LLM_MODEL', profile, "llama-3.3-70b-versatile"),
        "LLM_PARAMS": get_env_var('LLM_PARAMS', profile),
        
        # Define TTS settings
        "TTS_VENDOR": get_env_var('TTS_VENDOR', profile, "elevenlabs"),
        "TTS_KEY": get_env_var('TTS_KEY', profile),
        "TTS_MODEL": get_env_var('TTS_MODEL', profile, "eleven_flash_v2_5"),
        "TTS_VOICE_ID": get_env_var('TTS_VOICE_ID', profile, "TX3LPaxmHKxFdv7VOQHJ"),
        "TTS_VOICE_STABILITY": get_env_var('TTS_VOICE_STABILITY', profile, "1"),
        "TTS_VOICE_SPEED": get_env_var('TTS_VOICE_SPEED', profile, "0.9"),
        "TTS_VOICE_SAMPLE_RATE": get_env_var('TTS_VOICE_SAMPLE_RATE', profile, "24000"),
        
        # Define ASR settings
        "ASR_LANGUAGE": get_env_var('ASR_LANGUAGE', profile, "en-US"),
        "ASR_VENDOR": get_env_var('ASR_VENDOR', profile, "deepgram"),
        
        # Define HeyGen settings
        "HEYGEN_API_KEY": get_env_var('HEYGEN_API_KEY', profile),
        "HEYGEN_AVATAR_ID": get_env_var('HEYGEN_AVATAR_ID', profile, "Wayne_20240711"),
        "HEYGEN_QUALITY": get_env_var('HEYGEN_QUALITY', profile, "high"),
        "HEYGEN_ACTIVITY_IDLE_TIMEOUT": get_env_var('HEYGEN_ACTIVITY_IDLE_TIMEOUT', profile, "35"),
        
        # Default values for prompt and greeting
        "DEFAULT_PROMPT": get_env_var('DEFAULT_PROMPT', profile, 
            "You are a virtual companion. The user can both talk and type to you and you will be sent text. "
            "Say you can hear them if asked. They can also see you as a digital human. "
            "Keep responses to around 20 to 40 words. Be upbeat and try and keep conversation "
            "going by learning more about the user. "),
        "DEFAULT_GREETING": get_env_var('DEFAULT_GREETING', profile, "hi there")
    }
    
    return constants


def lambda_handler(event, context):
    """
    Lambda handler function that processes the incoming request, generates tokens,
    and sends an agent to the channel or handles hangup requests.
    """
    # Check if queryStringParameters exists
    if 'queryStringParameters' not in event or event['queryStringParameters'] is None:
        return json_response(400, {"error": "Missing query parameters"})
    
    query_params = event['queryStringParameters']
    
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
            "agent_response": hangup_response
        })
    
    # Normal join flow or token-only flow
    # Get channel from query parameters
    if 'channel' not in query_params:
        return json_response(400, {"error": "Missing channel parameter"})
    
    channel = query_params['channel']
    
    # Generate agent_rtm_uid for this channel
    agent_rtm_uid = f"{constants['AGENT_UID']}-{channel}"
    
    # Check if this is a connect=false request (token-only mode)
    connect_param = query_params.get('connect', 'true').lower()
    token_only_mode = connect_param == 'false'
    
    # Get token for user with RTC and RTM capabilities
    user_token_data = build_token_with_rtm(channel, constants["USER_UID"], constants)
    
    # If connect=false, return only the user token without starting the agent
    if token_only_mode:
        return json_response(200, {
            "user_token": user_token_data,
            "agent_rtm_uid": agent_rtm_uid,
            "agent_response": {
                "status_code": 200,
                "response": json.dumps({
                    "message": "Token-only mode: user token generated successfully",
                    "mode": "token_only",
                    "connect": False
                }),
                "success": True
            }
        })
    
    # Normal connect=true flow: create and send agent
    # Get optional prompt, greeting, voice_id, and debug parameters
    prompt = query_params.get('prompt', constants["DEFAULT_PROMPT"])
    greeting = query_params.get('greeting', constants["DEFAULT_GREETING"])
    
    # Get voice parameters or use defaults
    voice_id = query_params.get('voice_id', constants["TTS_VOICE_ID"])
    voice_stability = query_params.get('voice_stability', constants["TTS_VOICE_STABILITY"])
    voice_speed = query_params.get('voice_speed', constants["TTS_VOICE_SPEED"])
    voice_sample_rate = query_params.get('voice_sample_rate', constants["TTS_VOICE_SAMPLE_RATE"])
    
    # Get HeyGen parameters or use defaults
    heygen_avatar_id = query_params.get('heygen_avatar_id', constants["HEYGEN_AVATAR_ID"])
    heygen_quality = query_params.get('heygen_quality', constants["HEYGEN_QUALITY"])
    heygen_enable = query_params.get('heygen_enable', 'true').lower() == 'true'
    
    # Also allow graph_id to be overridden via URL parameter
    graph_id = query_params.get('graph_id', constants["GRAPH_ID"])
    
    debug_mode = 'debug' in query_params
    
    # Create agent payload for sending to the channel
    agent_payload = create_agent_payload(
        channel, 
        constants, 
        prompt, 
        greeting, 
        voice_id, 
        voice_stability,
        voice_speed,
        voice_sample_rate,
        graph_id,
        heygen_avatar_id,
        heygen_quality,
        heygen_enable,
        debug_mode  # Pass the debug_mode flag to the create_agent_payload function
    )
    
    # In debug mode, return the clean agent payload
    if debug_mode:
        # Convert OrderedDict to regular dict for clean JSON serialization
        clean_payload = convert_to_clean_dict(agent_payload)
        
        # Return just the clean payload body formatted nicely
        return {
            "isBase64Encoded": False,
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(clean_payload, indent=2)
        }
    
    # Not in debug mode, send the agent to the channel
    agent_response = send_agent_to_channel(channel, agent_payload, constants)
    
    # If the agent response indicates failure, return the error with a proper status code
    if not agent_response.get("success"):
        error_status = agent_response.get("status_code", 500)
        return json_response(error_status, {
            "user_token": user_token_data,
            "agent_rtm_uid": agent_rtm_uid,
            "agent_response": agent_response,
            "error": f"Failed to send agent to channel. API returned status {error_status}."
        })
    
    return json_response(200, {
        "user_token": user_token_data,
        "agent_rtm_uid": agent_rtm_uid,
        "agent_response": agent_response
    })


def convert_to_clean_dict(obj):
    """
    Recursively convert OrderedDict and other objects to regular dicts for clean JSON output
    """
    if isinstance(obj, OrderedDict):
        return {k: convert_to_clean_dict(v) for k, v in obj.items()}
    elif isinstance(obj, dict):
        return {k: convert_to_clean_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_clean_dict(item) for item in obj]
    else:
        return obj


def json_response(status_code, body):
    """
    Creates a properly formatted JSON response for API Gateway
    
    Args:
        status_code: HTTP status code
        body: Dictionary to be serialized to JSON
        
    Returns:
        Dictionary formatted for API Gateway response
    """
    return {
        "isBase64Encoded": False,
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body)
    }


def hangup_agent(agent_id, constants):
    """
    Disconnects an agent from the channel by calling the leave API
    
    Args:
        agent_id: The ID of the agent to disconnect
        constants: Dictionary of constants
        
    Returns:
        Dictionary with the status code, response body, and success flag
    """
    # Construct the agent leave API URL
    agent_leave_url = f"{constants['AGENT_API_BASE_URL']}/{constants['APP_ID']}/agents/{agent_id}/leave"
    
    # Parse the URL to get host and path
    url_parts = urllib.parse.urlparse(agent_leave_url)
    host = url_parts.netloc
    path = url_parts.path
    
    # Use http.client directly with timeout
    conn = http.client.HTTPSConnection(host, timeout=30)
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": constants["AGENT_AUTH_HEADER"]
    }
    
    conn.request("POST", path, "", headers)
    
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


def build_token_with_rtm(channel_name, account, constants):
    """
    Builds a token with both RTC and RTM privileges using v007 token system.
    
    Args:
        channel_name: The channel name
        account: The user's account/UID
        constants: Dictionary of constants
    
    Returns:
        Dictionary containing token and uid
    """
    # Return APP_ID as token if APP_CERTIFICATE is empty
    if not constants["APP_CERTIFICATE"]:
        return {"token": constants["APP_ID"], "uid": account}
    
    # Use v007 token with service-based architecture
    token = AccessToken(constants["APP_ID"], constants["APP_CERTIFICATE"])
    
    # RTC Service
    rtc_service = ServiceRtc(channel_name, account)
    rtc_service.add_privilege(ServiceRtc.kPrivilegeJoinChannel, constants["PRIVILEGE_EXPIRE"])
    rtc_service.add_privilege(ServiceRtc.kPrivilegePublishAudioStream, constants["PRIVILEGE_EXPIRE"])
    rtc_service.add_privilege(ServiceRtc.kPrivilegePublishVideoStream, constants["PRIVILEGE_EXPIRE"])
    rtc_service.add_privilege(ServiceRtc.kPrivilegePublishDataStream, constants["PRIVILEGE_EXPIRE"])
    token.add_service(rtc_service)
    
    # RTM Service
    rtm_service = ServiceRtm(account)
    rtm_service.add_privilege(ServiceRtm.kPrivilegeLogin, constants["TOKEN_EXPIRE"])
    token.add_service(rtm_service)

    return {"token": token.build(), "uid": account}


def create_agent_payload(
    channel, 
    constants, 
    prompt=None, 
    greeting=None, 
    voice_id=None,
    voice_stability=None,
    voice_speed=None,
    voice_sample_rate=None,
    graph_id=None,
    heygen_avatar_id=None,
    heygen_quality=None,
    heygen_enable=True,
    debug_mode=False
):
    """
    Creates the payload for the agent to be sent to the Agora RTC channel
    
    Args:
        channel: The channel name
        constants: Dictionary of constants
        prompt: System prompt for the LLM (defaults to constants["DEFAULT_PROMPT"])
        greeting: Greeting message (defaults to constants["DEFAULT_GREETING"])
        voice_id: Voice ID for TTS (defaults to constants["TTS_VOICE_ID"])
        voice_stability: Voice stability for TTS (defaults to constants["TTS_VOICE_STABILITY"])
        voice_speed: Voice speed for TTS (defaults to constants["TTS_VOICE_SPEED"])
        voice_sample_rate: Voice sample rate for TTS (defaults to constants["TTS_VOICE_SAMPLE_RATE"])
        graph_id: Graph ID (defaults to constants["GRAPH_ID"])
        heygen_avatar_id: HeyGen avatar ID (defaults to constants["HEYGEN_AVATAR_ID"])
        heygen_quality: HeyGen quality setting (defaults to constants["HEYGEN_QUALITY"])
        heygen_enable: Whether to enable HeyGen avatar (defaults to True)
        debug_mode: Whether to include debug info in the payload
    
    Returns:
        Dictionary containing the agent payload
    """
    # Use provided values or defaults from constants
    if prompt is None:
        prompt = constants["DEFAULT_PROMPT"]
    if greeting is None:
        greeting = constants["DEFAULT_GREETING"]
    if voice_id is None:
        voice_id = constants["TTS_VOICE_ID"]
    if voice_stability is None:
        voice_stability = constants["TTS_VOICE_STABILITY"]
    if voice_speed is None:
        voice_speed = constants["TTS_VOICE_SPEED"]
    if voice_sample_rate is None:
        voice_sample_rate = constants["TTS_VOICE_SAMPLE_RATE"]
    if heygen_avatar_id is None:
        heygen_avatar_id = constants["HEYGEN_AVATAR_ID"]
    if heygen_quality is None:
        heygen_quality = constants["HEYGEN_QUALITY"]
    
    # Convert voice parameters to appropriate types
    try:
        voice_stability = float(voice_stability)
        voice_speed = float(voice_speed)
        voice_sample_rate = int(voice_sample_rate)
    except (ValueError, TypeError):
        # If conversion fails, use default values
        voice_stability = 1.0
        voice_speed = 1.0
        voice_sample_rate = 24000
    
    # Convert HeyGen activity idle timeout to int
    try:
        heygen_activity_idle_timeout = int(constants["HEYGEN_ACTIVITY_IDLE_TIMEOUT"])
    except (ValueError, TypeError):
        heygen_activity_idle_timeout = 35
    
    # Get token for agent with RTM privileges
    agent_token = build_token_with_rtm(channel, constants["AGENT_UID"], constants)["token"]
    
    # Get token for HeyGen agent video
    agent_video_token = build_token_with_rtm(channel, constants["AGENT_VIDEO_UID"], constants)["token"]
    
    # Define advanced features with enable_aivad hardcoded to true
    advanced_features = {
        "enable_bhvs": True,
        "enable_rtm": True,
        "enable_aivad": False
    }
    
    # Prepare the LLM params - either use the LLM_PARAMS from environment or default
    llm_params = {}
    
    if constants["LLM_PARAMS"]:
        try:
            # Fix common JSON formatting issues
            llm_params_str = constants["LLM_PARAMS"]
            
            # Replace typographic quotes with regular quotes
            llm_params_str = llm_params_str.replace('\u201c', '"').replace('\u201d', '"')
            llm_params_str = llm_params_str.replace('\u2018', "'").replace('\u2019', "'")
            
            # Try to parse the fixed JSON string
            llm_params = json.loads(llm_params_str)
            
            # Inject additional required values
            llm_params.update({
                "appId": constants["APP_ID"],
                "channel": channel,
                "userId": constants["USER_UID"] + "-" + channel,
                "enable_rtm": True,
                "agent_rtm_uid": constants["AGENT_UID"] + "2" + "-" + channel,
                "agent_rtm_token": agent_token,
                "agent_rtm_channel": channel
            })
        except Exception as e:
            # If parsing fails, use default params
            llm_params = {
                "model": constants["LLM_MODEL"],
                "stream": True
            }
    else:
        # Use default params
        llm_params = {
            "model": constants["LLM_MODEL"],
            "stream": True
        }
    
    # Build TTS configuration based on vendor
    tts_vendor = constants["TTS_VENDOR"].lower()
    
    if tts_vendor == "cartesia":
        # Cartesia TTS configuration
        tts_config = {
            "vendor": "cartesia",
            "params": {
                "api_key": constants["TTS_KEY"],
                "model_id": constants["TTS_MODEL"],  # Default to "sonic-2" if not specified
                "sample_rate": voice_sample_rate,
                "voice": {
                    "mode": "id",
                    "id": voice_id
                }
            }
        }
    else:
        # Default to ElevenLabs TTS configuration
        tts_config = {
            "vendor": constants["TTS_VENDOR"],
            "params": {
                "key": constants["TTS_KEY"],
                "model_id": constants["TTS_MODEL"],
                "voice_id": voice_id,
                "stability": voice_stability, 
                "speed": voice_speed,
                "sample_rate": voice_sample_rate
            }
        }
    
    # Build the parameters dictionary - this goes INSIDE properties
    parameters = {
        "enable_error_message": True,
        "data_channel": "rtm",
        "silence_config": {
            "timeout_ms": 7000,
            "action": "think",
            "content": "Provide a short random fact"
        }
    }
    
    # Build the properties dictionary with parameters at the end
    properties = {
        "channel": channel,
        "token": agent_token,
        "agent_rtc_uid": constants["AGENT_UID"],
        "agent_rtm_uid": constants["AGENT_UID"] + "-" + channel,
        "remote_rtc_uids": [constants["USER_UID"]],
        "advanced_features": advanced_features,
        "enable_string_uid": False,
        "idle_timeout": 0,
        "llm": {
            "vendor": "custom",
            "url": constants["LLM_URL"],
            "api_key": constants["LLM_API_KEY"],
            "system_messages": [
                {
                    "role": "system",
                    "content": prompt
                }
            ],
            "greeting_message": greeting,
            "failure_message": "Sorry but can't talk just now.",
            "max_history": 3,
            "params": llm_params
        },
        "turn_detection": {
            "silence_duration_ms": 300,
            "interrupt_mode": "append"
        },
        "asr": {
            "language": constants["ASR_LANGUAGE"],
            "vendor": constants["ASR_VENDOR"]
        },
        "tts": tts_config,
        "avatar": {
            "enable": heygen_enable,
            "vendor": "heygen",
            "params": {
                "api_key": constants["HEYGEN_API_KEY"],
                "avatar_id": heygen_avatar_id,
                "quality": heygen_quality,
                "agora_appid": constants["APP_ID"],
                "agora_token": agent_video_token,
                "agora_channel": channel,
                "agora_uid":  constants["AGENT_VIDEO_UID"] ,
                "activity_idle_timeout": heygen_activity_idle_timeout
            }
        },
        "parameters": parameters  # Parameters inside properties at the end
    }
    
    # Build the payload
    if debug_mode:
        # For debug mode, return regular dict
        payload = {}
        if graph_id:
            payload["graph_id"] = graph_id
        elif constants.get("GRAPH_ID"):
            payload["graph_id"] = constants["GRAPH_ID"]
        payload["name"] = channel
        payload["properties"] = properties
        return payload
    else:
        # For non-debug mode, use OrderedDict to maintain order
        payload_items = []
        if graph_id:
            payload_items.append(("graph_id", graph_id))
        elif constants.get("GRAPH_ID"):
            payload_items.append(("graph_id", constants["GRAPH_ID"]))
        payload_items.extend([
            ("name", channel),
            ("properties", properties)
        ])
        return OrderedDict(payload_items)


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
    
    # Create a custom JSONEncoder to ensure ordered dictionaries maintain their order
    class OrderedDictEncoder(json.JSONEncoder):
        def encode(self, obj):
            if isinstance(obj, OrderedDict):
                return '{' + ', '.join(f'{json.dumps(k)}: {json.dumps(v)}' for k, v in obj.items()) + '}'
            return super().encode(obj)
    
    # Use the custom encoder to ensure ordered dict keys are preserved
    payload_json = json.dumps(agent_payload, cls=OrderedDictEncoder)
    
    conn.request("POST", path, payload_json, headers)
    
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


# Token v007 implementation with service-based architecture

def get_version():
    return '007'

def pack_uint16(x):
    return struct.pack('<H', int(x))

def unpack_uint16(buffer):
    data_length = struct.calcsize('H')
    return struct.unpack('<H', buffer[:data_length])[0], buffer[data_length:]

def pack_uint32(x):
    return struct.pack('<I', int(x))

def unpack_uint32(buffer):
    data_length = struct.calcsize('I')
    return struct.unpack('<I', buffer[:data_length])[0], buffer[data_length:]

def pack_int16(x):
    return struct.pack('<h', int(x))

def unpack_int16(buffer):
    data_length = struct.calcsize('h')
    return struct.unpack('<h', buffer[:data_length])[0], buffer[data_length:]

def pack_string(string):
    if isinstance(string, str):
        string = string.encode('utf-8')
    return pack_uint16(len(string)) + string

def unpack_string(buffer):
    data_length, buffer = unpack_uint16(buffer)
    return struct.unpack('<{}s'.format(data_length), buffer[:data_length])[0], buffer[data_length:]

def pack_map_uint32(m):
    return pack_uint16(len(m)) + b''.join([pack_uint16(k) + pack_uint32(v) for k, v in m.items()])

def unpack_map_uint32(buffer):
    data_length, buffer = unpack_uint16(buffer)
    data = {}
    for i in range(data_length):
        k, buffer = unpack_uint16(buffer)
        v, buffer = unpack_uint32(buffer)
        data[k] = v
    return data, buffer

def pack_map_string(m):
    return pack_uint16(len(m)) + b''.join([pack_uint16(k) + pack_string(v) for k, v in m.items()])

def unpack_map_string(buffer):
    data_length, buffer = unpack_uint16(buffer)
    data = {}
    for i in range(data_length):
        k, buffer = unpack_uint16(buffer)
        v, buffer = unpack_string(buffer)
        data[k] = v
    return data, buffer


class Service:
    def __init__(self, service_type):
        self.__type = service_type
        self.__privileges = {}

    def __pack_type(self):
        return pack_uint16(self.__type)

    def __pack_privileges(self):
        privileges = OrderedDict(
            sorted(iter(self.__privileges.items()), key=lambda x: int(x[0])))
        return pack_map_uint32(privileges)

    def add_privilege(self, privilege, expire):
        self.__privileges[privilege] = expire

    def service_type(self):
        return self.__type

    def pack(self):
        return self.__pack_type() + self.__pack_privileges()

    def unpack(self, buffer):
        self.__privileges, buffer = unpack_map_uint32(buffer)
        return buffer


class ServiceRtc(Service):
    kServiceType = 1

    kPrivilegeJoinChannel = 1
    kPrivilegePublishAudioStream = 2
    kPrivilegePublishVideoStream = 3
    kPrivilegePublishDataStream = 4

    def __init__(self, channel_name='', uid=0):
        super(ServiceRtc, self).__init__(ServiceRtc.kServiceType)
        self.__channel_name = channel_name.encode('utf-8')
        self.__uid = b'' if uid == 0 else str(uid).encode('utf-8')

    def pack(self):
        return super(ServiceRtc, self).pack() + pack_string(self.__channel_name) + pack_string(self.__uid)

    def unpack(self, buffer):
        buffer = super(ServiceRtc, self).unpack(buffer)
        self.__channel_name, buffer = unpack_string(buffer)
        self.__uid, buffer = unpack_string(buffer)
        return buffer


class ServiceRtm(Service):
    kServiceType = 2

    kPrivilegeLogin = 1

    def __init__(self, user_id=''):
        super(ServiceRtm, self).__init__(ServiceRtm.kServiceType)
        self.__user_id = user_id.encode('utf-8')

    def pack(self):
        return super(ServiceRtm, self).pack() + pack_string(self.__user_id)

    def unpack(self, buffer):
        buffer = super(ServiceRtm, self).unpack(buffer)
        self.__user_id, buffer = unpack_string(buffer)
        return buffer


class ServiceFpa(Service):
    kServiceType = 4

    kPrivilegeLogin = 1

    def __init__(self):
        super(ServiceFpa, self).__init__(ServiceFpa.kServiceType)

    def pack(self):
        return super(ServiceFpa, self).pack()

    def unpack(self, buffer):
        buffer = super(ServiceFpa, self).unpack(buffer)
        return buffer


class ServiceChat(Service):
    kServiceType = 5

    kPrivilegeUser = 1
    kPrivilegeApp = 2

    def __init__(self, user_id=''):
        super(ServiceChat, self).__init__(ServiceChat.kServiceType)
        self.__user_id = user_id.encode('utf-8')

    def pack(self):
        return super(ServiceChat, self).pack() + pack_string(self.__user_id)

    def unpack(self, buffer):
        buffer = super(ServiceChat, self).unpack(buffer)
        self.__user_id, buffer = unpack_string(buffer)
        return buffer


class ServiceEducation(Service):
    kServiceType = 7

    kPrivilegeRoomUser = 1
    kPrivilegeUser = 2
    kPrivilegeApp = 3

    def __init__(self, room_uuid='', user_uuid='', role=-1):
        super(ServiceEducation, self).__init__(ServiceEducation.kServiceType)
        self.__room_uuid = room_uuid.encode('utf-8')
        self.__user_uuid = user_uuid.encode('utf-8')
        self.__role = role

    def pack(self):
        return super(ServiceEducation, self).pack() + pack_string(self.__room_uuid) + pack_string(
            self.__user_uuid) + pack_int16(self.__role)

    def unpack(self, buffer):
        buffer = super(ServiceEducation, self).unpack(buffer)
        self.__room_uuid, buffer = unpack_string(buffer)
        self.__user_uuid, buffer = unpack_string(buffer)
        self.__role, buffer = unpack_int16(buffer)
        return buffer


class AccessToken:
    kServices = {
        ServiceRtc.kServiceType: ServiceRtc,
        ServiceRtm.kServiceType: ServiceRtm,
        ServiceFpa.kServiceType: ServiceFpa,
        ServiceChat.kServiceType: ServiceChat,
        ServiceEducation.kServiceType: ServiceEducation,
    }

    def __init__(self, app_id='', app_certificate='', issue_ts=0, expire=900):
        self.__app_id = app_id
        self.__app_cert = app_certificate

        self.__issue_ts = issue_ts if issue_ts != 0 else int(time.time())
        self.__expire = expire
        self.__salt = secrets.SystemRandom().randint(1, 99999999)

        self.__service = {}

    def __signing(self):
        signing = hmac.new(pack_uint32(self.__issue_ts),
                           self.__app_cert, sha256).digest()
        signing = hmac.new(pack_uint32(self.__salt), signing, sha256).digest()
        return signing

    def __build_check(self):
        def is_uuid(data):
            if len(data) != 32:
                return False
            try:
                bytes.fromhex(data)
            except:
                return False
            return True

        if not is_uuid(self.__app_id) or not is_uuid(self.__app_cert):
            return False
        if not self.__service:
            return False
        return True

    def add_service(self, service):
        self.__service[service.service_type()] = service

    def build(self):
        if not self.__build_check():
            return ''

        self.__app_id = self.__app_id.encode('utf-8')
        self.__app_cert = self.__app_cert.encode('utf-8')
        signing = self.__signing()
        signing_info = pack_string(self.__app_id) + pack_uint32(self.__issue_ts) + pack_uint32(self.__expire) + \
                       pack_uint32(self.__salt) + pack_uint16(len(self.__service))

        for _, service in self.__service.items():
            signing_info += service.pack()

        signature = hmac.new(signing, signing_info, sha256).digest()

        return get_version() + base64.b64encode(zlib.compress(pack_string(signature) + signing_info)).decode('utf-8')

    def from_string(self, origin_token):
        try:
            origin_version = origin_token[:3]
            if origin_version != get_version():
                return False

            buffer = zlib.decompress(
                base64.b64decode(origin_token[3:]))
            signature, buffer = unpack_string(buffer)
            self.__app_id, buffer = unpack_string(buffer)
            self.__issue_ts, buffer = unpack_uint32(buffer)
            self.__expire, buffer = unpack_uint32(buffer)
            self.__salt, buffer = unpack_uint32(buffer)
            service_count, buffer = unpack_uint16(buffer)

            for i in range(service_count):
                service_type, buffer = unpack_uint16(buffer)
                service = AccessToken.kServices[service_type]()
                buffer = service.unpack(buffer)
                self.__service[service_type] = service
        except Exception as e:
            print('Error: {}'.format(repr(e)))
            raise ValueError('Error: parse origin token failed')
        return True
