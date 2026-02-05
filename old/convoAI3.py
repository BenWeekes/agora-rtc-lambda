import json
import hmac
from hashlib import sha256
import base64
import struct
from zlib import crc32
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
        "AGENT_UID": "agent",
        "USER_UID": "user",
        "AGENT_VIDEO_UID": "agent_video",
        
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
        
        # Define ASR settings
        "ASR_LANGUAGE": get_env_var('ASR_LANGUAGE', profile, "en-US"),
        "ASR_VENDOR": get_env_var('ASR_VENDOR', profile, "deepgram"),
        
        # Default values for prompt and greeting
        "DEFAULT_PROMPT": get_env_var('DEFAULT_PROMPT', profile, 
            "You are a virtual companion. The user can both talk and type to you and you will be sent text. "
            "Say you can hear them if asked. They can also see you as a digital human. "
            "Keep responses to around 10 to 20 words or shorter. Be upbeat and try and keep conversation "
            "going by learning more about the user. "),
        "DEFAULT_GREETING": get_env_var('DEFAULT_GREETING', profile, "hi there"),
        
        # Controller endpoint
        "CONTROLLER_ENDPOINT": get_env_var('CONTROLLER_ENDPOINT', profile, "wss:wvc-ln-01.trulience.com")
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
            "agent_response": hangup_response,
            "controller_endpoint": constants["CONTROLLER_ENDPOINT"]
        })
    
    # Normal join flow or token-only flow
    # Get channel from query parameters
    if 'channel' not in query_params:
        return json_response(400, {"error": "Missing channel parameter"})
    
    channel = query_params['channel']
    
    # Check if this is a connect=false request (token-only mode)
    connect_param = query_params.get('connect', 'true').lower()
    token_only_mode = connect_param == 'false'
    
    # Get token for user with RTC and RTM capabilities
    user_token_data = build_token_with_rtm(channel, constants["USER_UID"], constants)
    
    # Get token for agent video with RTC and RTM capabilities
    agent_video_token_data = build_token_with_rtm(channel, constants["AGENT_VIDEO_UID"], constants)
    
    # If connect=false, return only the user token and agent video token without starting the agent
    if token_only_mode:
        return json_response(200, {
            "user_token": user_token_data,
            "agent_video_token": agent_video_token_data,
            "controller_endpoint": constants["CONTROLLER_ENDPOINT"],
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
    # Get optional prompt, greeting, voice_id, and debug parameters
    prompt = query_params.get('prompt', constants["DEFAULT_PROMPT"])
    greeting = query_params.get('greeting', constants["DEFAULT_GREETING"])
    
    # Get voice parameters or use defaults
    voice_id = query_params.get('voice_id', constants["TTS_VOICE_ID"])
    voice_stability = query_params.get('voice_stability', constants["TTS_VOICE_STABILITY"])
    voice_speed = query_params.get('voice_speed', constants["TTS_VOICE_SPEED"])
    voice_sample_rate = query_params.get('voice_sample_rate', constants["TTS_VOICE_SAMPLE_RATE"])
    
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
        debug_mode  # Pass the debug_mode flag to the create_agent_payload function
    )
    
    # In debug mode, return the agent payload instead of sending the agent
    if debug_mode:
        return json_response(200, {
            "user_token": user_token_data,
            "agent_video_token": agent_video_token_data,
            "controller_endpoint": constants["CONTROLLER_ENDPOINT"],
            "agent_payload": agent_payload
        })
    
    # Not in debug mode, send the agent to the channel
    agent_response = send_agent_to_channel(channel, agent_payload, constants)
    
    # If the agent response indicates failure, return the error with a proper status code
    if not agent_response.get("success"):
        error_status = agent_response.get("status_code", 500)
        return json_response(error_status, {
            "user_token": user_token_data,
            "agent_video_token": agent_video_token_data,
            "controller_endpoint": constants["CONTROLLER_ENDPOINT"],
            "agent_response": agent_response,
            "error": f"Failed to send agent to channel. API returned status {error_status}."
        })
    
    return json_response(200, {
        "user_token": user_token_data,
        "agent_video_token": agent_video_token_data,
        "controller_endpoint": constants["CONTROLLER_ENDPOINT"],
        "agent_response": agent_response
    })


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
    Builds a token with both RTC and RTM privileges, similar to the Go example.
    
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
    
    # For this implementation, we'll continue using the AccessToken class from the original code
    # but add both RTC and RTM privileges
    token = AccessToken(constants["APP_ID"], constants["APP_CERTIFICATE"], channel_name, account)
    
    # RTC privileges - for channel
    # 1: Join Channel
    token.addPrivilege(1, constants["PRIVILEGE_EXPIRE"])
    # 2: Publish Audio
    token.addPrivilege(2, constants["PRIVILEGE_EXPIRE"])
    # 3: Publish Video
    token.addPrivilege(3, constants["PRIVILEGE_EXPIRE"])
    # 4: Publish Data Stream
    token.addPrivilege(4, constants["PRIVILEGE_EXPIRE"])
    
    # RTM privilege - for login
    # 1000: RTM Login
    token.addPrivilege(1000, constants["TOKEN_EXPIRE"])
    
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
    
    # Get token for agent with RTM privileges
    agent_token = build_token_with_rtm(channel, constants["AGENT_UID"], constants)["token"]
    
    # Define advanced features with enable_aivad hardcoded to true
    advanced_features = {
        "enable_bhvs": False,
        "enable_rtm": True,
        "enable_aivad": False
    }
    
    # Prepare the LLM params - either use the LLM_PARAMS from environment or default
    llm_params = {}
    
    # Create debug info but don't add it to the payload yet
    debug_info = {
        "llm_params_env": constants["LLM_PARAMS"],
        "llm_model_fallback": constants["LLM_MODEL"]
    }
    
    if constants["LLM_PARAMS"]:
        try:
            # Fix common JSON formatting issues
            llm_params_str = constants["LLM_PARAMS"]
            
            # Replace typographic quotes with regular quotes
            llm_params_str = llm_params_str.replace('\u201c', '"').replace('\u201d', '"')
            llm_params_str = llm_params_str.replace('\u2018', "'").replace('\u2019', "'")
            
            # Try to parse the fixed JSON string
            llm_params = json.loads(llm_params_str)
            debug_info["llm_params_parsed"] = True
            debug_info["llm_params_fixed"] = True
            
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
            # Catch and log all exceptions for better debugging
            debug_info["llm_params_error"] = str(e)
            debug_info["llm_params_parsed"] = False
            
            # If parsing fails, use default params
            llm_params = {
                "model": constants["LLM_MODEL"],
                "stream": True
            }
    else:
        # Use default params
        debug_info["llm_params_parsed"] = False
        debug_info["reason"] = "No LLM_PARAMS found in environment"
        llm_params = {
            "model": constants["LLM_MODEL"],
            "stream": True
        }
    
    # Only add debug info to the llm_params if in debug mode
    if debug_mode:
        llm_params["debug_info"] = debug_info
    
    # We need to ensure the graph_id is first in the serialized JSON
    # To do this, we'll construct our payload as lists of key-value pairs
    # which will maintain order when converted to JSON
    payload_items = []
    
    # Add graph_id first if available
    if graph_id:
        payload_items.append(("graph_id", graph_id))
    elif constants.get("GRAPH_ID"):
        payload_items.append(("graph_id", constants["GRAPH_ID"]))
    
    # Add other required fields in the desired order
    # Note: parameters is now INSIDE properties, and audio_scenario is added
    payload_items.extend([
        ("name", channel),
        ("properties", {
            "parameters": {
                "enable_error_message": True,
                "audio_scenario": "chorus"
            },
            "channel": channel,
            "token": agent_token,
            "agent_rtc_uid": constants["AGENT_UID"],
            "agent_rtm_uid": constants["AGENT_UID"] + "-" + channel,
            "remote_rtc_uids": [constants["USER_UID"]],
            "advanced_features": advanced_features,
            "enable_string_uid": True,
            "idle_timeout": 60,
            "llm": {
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
                "max_history": 32,
                "params": llm_params
            },
            "vad": {
                "silence_duration_ms": 300
            },
            "asr": {
                "language": constants["ASR_LANGUAGE"],
                "vendor": constants["ASR_VENDOR"]
            },
            "tts": {
                "vendor": constants["TTS_VENDOR"],
                "skip_patterns": [5],
                "params": {
                    "key": constants["TTS_KEY"],
                    "model_id": constants["TTS_MODEL"],
                    "voice_id": voice_id,
                    "stability": voice_stability, 
                    "speed": voice_speed,
                    "sample_rate": voice_sample_rate
                }
            }
        })
    ])
    
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