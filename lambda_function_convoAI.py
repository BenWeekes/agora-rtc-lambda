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

# Constants for Agora
APP_ID = "0ee04792a45859f95d7"
APP_CERTIFICATE = "6f3fb0fb1a19f29746" # or empty if security disabled
AGENT_AUTH_HEADER = "Basic NzViOGUyDMwNGE2Y2E2MzYwMGRhYTkyMWQwMjk="
AGENT_API_BASE_URL = "https://api.agora.io/api/conversational-ai-agent/v2/projects"

# Fixed UIDs
AGENT_UID = 110
USER_UID = 111

# Constants for token generation
VERSION_LENGTH = 3
APP_ID_LENGTH = 32

# Define LLM settings
LLM_URL = "https://api.x.ai/v1/chat/completions"
LLM_API_KEY = "xai-aBXOPgZPtaWgCSxbmS4ywat9UvzI2Pmk2sxNlYze1fKhuJFwH5z3As275gp6m1AYLzMMtqxazooj3g4m"
LLM_MODEL = "grok-2-1212"

# Define TTS settings
TTS_VENDOR = "elevenlabs"
TTS_KEY = "e67f1b2746856cb18ecaf04233613ddf"
TTS_MODEL = "eleven_flash_v2_5"
TTS_VOICE_ID = "cgSgspJ2msm6clMCkdW9"

# Define ASR settings
ASR_LANGUAGE = "en-US"
ASR_VENDOR = "deepgram"

# Default values for prompt and greeting
DEFAULT_PROMPT = "You are a helpful chatbot. Keep responses short."
DEFAULT_GREETING = "hi there"

def lambda_handler(event, context):
    """
    Lambda handler function that processes the incoming request, generates tokens,
    and sends an agent to the channel.
    """
    # Get channel from query parameters
    channel = ""
    if ('queryStringParameters' in event and 'channel' in event['queryStringParameters']):
        channel = event['queryStringParameters']['channel'] 
    else:
        return {
            "isBase64Encoded": False,
            "statusCode": 403,
            "headers": {"Content-Type": "application/json"},
            "multiValueHeaders": {},
            "body": "channel not found"
        }
    
    # Get optional prompt, greeting, voice_id, and debug parameters
    prompt = DEFAULT_PROMPT
    greeting = DEFAULT_GREETING
    voice_id = TTS_VOICE_ID
    debug_mode = False
    
    if ('queryStringParameters' in event and 'prompt' in event['queryStringParameters']):
        prompt = event['queryStringParameters']['prompt']
        
    if ('queryStringParameters' in event and 'greeting' in event['queryStringParameters']):
        greeting = event['queryStringParameters']['greeting']
        
    if ('queryStringParameters' in event and 'voice_id' in event['queryStringParameters']):
        voice_id = event['queryStringParameters']['voice_id']
        
    if ('queryStringParameters' in event and 'debug' in event['queryStringParameters']):
        debug_mode = True
    
    # Get token for user
    user_token_data = get_token(channel, USER_UID)
    
    # Create agent payload for sending to the channel
    agent_payload = create_agent_payload(channel, prompt, greeting, voice_id)
    
    # In debug mode, return the agent payload instead of sending the agent
    if debug_mode:
        return {
            'statusCode': 200,
            'body': json.dumps({
                "user_token": user_token_data,
                "agent_payload": agent_payload
            })
        }
    
    # Not in debug mode, send the agent to the channel
    agent_response = send_agent_to_channel(channel, agent_payload)
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            "user_token": user_token_data,
            "agent_response": agent_response
        })
    }

def get_token(channel, uid):
    """
    Generates a token for the given channel and uid.
    If APP_CERTIFICATE is empty, returns APP_ID as the token.
    
    Args:
        channel: The channel name
        uid: The user's UID
    
    Returns:
        Dictionary containing token and uid
    """
    # Return APP_ID as token if APP_CERTIFICATE is empty
    if not APP_CERTIFICATE:
        return {"token": APP_ID, "uid": uid}
    
    token = AccessToken(APP_ID, APP_CERTIFICATE, channel, uid)
    return {"token": token.build(), "uid": uid}

def create_agent_payload(channel, prompt=DEFAULT_PROMPT, greeting=DEFAULT_GREETING, voice_id=TTS_VOICE_ID):
    """
    Creates the payload for the agent to be sent to the Agora RTC channel
    
    Args:
        channel: The channel name
        prompt: System prompt for the LLM (defaults to DEFAULT_PROMPT)
        greeting: Greeting message (defaults to DEFAULT_GREETING)
        voice_id: Voice ID for TTS (defaults to TTS_VOICE_ID)
    
    Returns:
        Dictionary containing the agent payload
    """
    # Get token for agent
    agent_token = get_token(channel, AGENT_UID)["token"]
    
    agent_payload = {
        "name": channel,  # Use channel as the agent name
        "properties": {
            "channel": channel,
            "token": agent_token,
            "agent_rtc_uid": str(AGENT_UID),
            "remote_rtc_uids": ["*"],
            "enable_string_uid": False,
            "idle_timeout": 30,
            "llm": {
                "url": LLM_URL,
                "api_key": LLM_API_KEY,
                "system_messages": [
                    {
                        "role": "system",
                        "content": prompt
                    }
                ],
                "greeting_message": greeting,
                "failure_message": "Sorry, I don't know how to answer this question.",
                "max_history": 3,
                "params": {
                    "model": LLM_MODEL,
                    "stream": True
                }
            },
            "asr": {
                "language": ASR_LANGUAGE,
                "vendor": ASR_VENDOR
            },
            "tts": {
                "vendor": TTS_VENDOR,
                "params": {
                    "key": TTS_KEY,
                    "model_id": TTS_MODEL,
                    "voice_id": voice_id
                }
            }
        }
    }
    
    return agent_payload

def send_agent_to_channel(channel, agent_payload):
    """
    Sends an agent to the specified Agora RTC channel by calling the REST API
    
    Args:
        channel: The channel name
        agent_payload: The complete agent payload to send
    
    Returns:
        Dictionary with the status code, response body, and success flag
    """
    try:
        # Construct the agent API URL
        agent_api_url = f"{AGENT_API_BASE_URL}/{APP_ID}/join"
        
        # Parse the URL to get host and path
        url_parts = urllib.parse.urlparse(agent_api_url)
        host = url_parts.netloc
        path = url_parts.path
        
        # Use http.client directly
        conn = http.client.HTTPSConnection(host)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": AGENT_AUTH_HEADER
        }
        
        payload_json = json.dumps(agent_payload)
        
        conn.request("POST", path, payload_json, headers)
        
        # Get the response
        response = conn.getresponse()
        status_code = response.status
        response_text = response.read().decode('utf-8')
        
        # Log the response for debugging
        print(f"Agent API response: {status_code} - {response_text}")
        
        conn.close()
        
        # Return a dictionary with the response details
        return {
            "status_code": status_code,
            "response": response_text,
            "success": status_code == 200
        }
    except Exception as e:
        error_message = str(e)
        print(f"Error sending agent to channel: {error_message}")
        return {
            "status_code": 500,
            "response": f"Error: {error_message}",
            "success": False
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
        if (uid == 0):
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
