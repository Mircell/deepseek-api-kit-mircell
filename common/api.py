from curl_cffi import requests
from typing import Optional, Dict, Any, Generator, Literal
import json
import os
import sys
import time
import subprocess
from pathlib import Path
from .pow import DeepSeekPOW

ThinkingMode = Literal['detailed', 'simple', 'disabled']
SearchMode = Literal['enabled', 'disabled']

DEBUG_SSE = os.environ.get("DEBUG_SSE", "0") == "1"


def _dbg(*args, **kwargs):
    if DEBUG_SSE:
        print("[SSE-DEBUG]", *args, file=sys.stderr, **kwargs)


class DeepSeekError(Exception):
    """Base exception for all DeepSeek API errors"""
    pass


class AuthenticationError(DeepSeekError):
    """Raised when authentication fails"""
    pass


class RateLimitError(DeepSeekError):
    """Raised when API rate limit is exceeded"""
    pass


class NetworkError(DeepSeekError):
    """Raised when network communication fails"""
    pass


class CloudflareError(DeepSeekError):
    """Raised when Cloudflare blocks the request"""
    pass


class APIError(DeepSeekError):
    """Raised when API returns an error response"""
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class SSEMessageParser:
    """
    پارسر SSE دیپ‌سیک که هر سه فرمت واقعی را پشتیبانی می‌کند:

    فرمت ۱ (توکن خام):
        {"v": " Hello"}

    فرمت ۲ (fragment اولیه):
        {"p": "response/fragments", "o": "APPEND",
         "v": [{"id": 3, "type": "RESPONSE", "content": "Hello", ...}]}

    فرمت ۳ (آپدیت افزایشی):
        {"p": "response/fragments/-1/content", "v": "!"}
        {"p": "response/fragments/-1/thinking_content", "v": "..."}

    فرمت ۴ (وضعیت):
        {"p": "response/status", "o": "SET", "v": "FINISHED"}
        {"p": "response", "o": "BATCH",
         "v": [{"p": "quasi_status", "v": "FINISHED"}]}
    """

    def __init__(self):
        self.thinking_content = ""
        self.content = ""
        self.finished_status = None
        self.current_section = None
        self.response_message_id = None
        self._emitted_finished = False

    # ------------------------------------------------------------------
    def _decode(self, chunk) -> Optional[dict]:
        if isinstance(chunk, bytes):
            chunk = chunk.decode('utf-8', 'ignore')

        json_str = chunk
        if chunk.startswith('data: '):
            json_str = chunk[6:]
        elif chunk.startswith('S.m: '):
            json_str = chunk[5:]

        if not json_str.strip():
            return None

        try:
            obj = json.loads(json_str)
        except json.JSONDecodeError:
            return None

        _dbg("RAW:", json_str[:400])
        return obj

    # ------------------------------------------------------------------
    def _process(self, obj: dict):
        """از یک payload، صفر یا چند partial yield می‌کند."""

        # --- response_message_id در سطح بالا ---
        if 'response_message_id' in obj and len(obj) <= 3:
            self.response_message_id = obj['response_message_id']
            return

        p = obj.get('p')
        o = obj.get('o')
        v = obj.get('v', '')

        # --- BATCH: v یک لیست از sub-ops ---
        if o == 'BATCH' and isinstance(v, list):
            for sub in v:
                if isinstance(sub, dict):
                    yield from self._process(sub)
            return

        # --- v یک dict است ---
        if isinstance(v, dict):
            yield from self._process_nested(v)
            return

        # --- v یک لیست است (response/fragments) ---
        if isinstance(v, list):
            yield from self._process_fragments(v)
            return

        # --- توکن خام: {"v": "..."} بدون p ---
        if not p and isinstance(v, str) and v:
            if self.current_section == 'thinking':
                self.thinking_content += v
                yield {
                    'type': 'thinking',
                    'delta': v,
                    'accumulated': self.thinking_content,
                    'finished': False,
                    'response_message_id': self.response_message_id,
                }
            elif self.current_section == 'content':
                self.content += v
                yield {
                    'type': 'content',
                    'delta': v,
                    'accumulated': self.content,
                    'finished': False,
                    'response_message_id': self.response_message_id,
                }
            return

        # --- p-based (SET / APPEND) ---
        if p:
            # مسیرهای content
            if p.endswith('/content') or p == 'response/content':
                is_thinking = 'thinking' in p
                new_v = str(v) if v is not None else ""
                if o == 'APPEND':
                    if is_thinking:
                        self.thinking_content += new_v
                    else:
                        self.content += new_v
                else:
                    if is_thinking:
                        self.thinking_content = new_v
                    else:
                        self.content = new_v

                if is_thinking:
                    self.current_section = 'thinking'
                    if new_v:
                        yield {
                            'type': 'thinking',
                            'delta': new_v,
                            'accumulated': self.thinking_content,
                            'finished': False,
                            'response_message_id': self.response_message_id,
                        }
                else:
                    self.current_section = 'content'
                    if new_v:
                        yield {
                            'type': 'content',
                            'delta': new_v,
                            'accumulated': self.content,
                            'finished': False,
                            'response_message_id': self.response_message_id,
                        }
                return

            # مسیرهای thinking_content
            if 'thinking_content' in p:
                new_v = str(v) if v is not None else ""
                if o == 'APPEND':
                    self.thinking_content += new_v
                else:
                    self.thinking_content = new_v
                self.current_section = 'thinking'
                if new_v:
                    yield {
                        'type': 'thinking',
                        'delta': new_v,
                        'accumulated': self.thinking_content,
                        'finished': False,
                        'response_message_id': self.response_message_id,
                    }
                return

            # وضعیت
            if p == 'response/status':
                self.finished_status = v
                self.current_section = None
                if not self._emitted_finished:
                    self._emitted_finished = True
                    yield {
                        'type': 'finished',
                        'finished_status': v,
                        'response_message_id': self.response_message_id,
                        'thinking_content': self.thinking_content,
                        'content': self.content,
                    }
                return

            # سایر مسیرها را نادیده بگیر
            return

    # ------------------------------------------------------------------
    def _process_fragments(self, fragments: list):
        """پردازش آرایه response/fragments"""
        for frag in fragments:
            if not isinstance(frag, dict):
                continue
            frag_type = frag.get('type', '')
            frag_content = frag.get('content', '')

            if frag_type == 'THINKING':
                self.thinking_content = frag_content
                self.current_section = 'thinking'
                if frag_content:
                    yield {
                        'type': 'thinking',
                        'delta': frag_content,
                        'accumulated': self.thinking_content,
                        'finished': False,
                        'response_message_id': self.response_message_id,
                    }
            elif frag_type == 'RESPONSE':
                self.content = frag_content
                self.current_section = 'content'
                if frag_content:
                    yield {
                        'type': 'content',
                        'delta': frag_content,
                        'accumulated': self.content,
                        'finished': False,
                        'response_message_id': self.response_message_id,
                    }

    # ------------------------------------------------------------------
    def _process_nested(self, v: dict):
        """پردازش دیکشنری تودرتو"""
        resp = v.get('response') if 'response' in v else v
        if not isinstance(resp, dict):
            return

        tc = resp.get('thinking_content')
        if isinstance(tc, str) and tc:
            if tc.startswith(self.thinking_content):
                delta = tc[len(self.thinking_content):]
            else:
                delta = tc
            self.thinking_content = tc
            if delta:
                yield {
                    'type': 'thinking',
                    'delta': delta,
                    'accumulated': self.thinking_content,
                    'finished': False,
                    'response_message_id': self.response_message_id,
                }

        ct = resp.get('content')
        if isinstance(ct, str) and ct:
            if ct.startswith(self.content):
                delta = ct[len(self.content):]
            else:
                delta = ct
            self.content = ct
            if delta:
                yield {
                    'type': 'content',
                    'delta': delta,
                    'accumulated': self.content,
                    'finished': False,
                    'response_message_id': self.response_message_id,
                }

        st = resp.get('status')
        if st:
            self.finished_status = st
            if st == 'FINISHED' and not self._emitted_finished:
                self._emitted_finished = True
                yield {
                    'type': 'finished',
                    'finished_status': st,
                    'response_message_id': self.response_message_id,
                    'thinking_content': self.thinking_content,
                    'content': self.content,
                }

    # ------------------------------------------------------------------
    # API عمومی
    # ------------------------------------------------------------------
    def parse_sse_streaming(self, chunk):
        obj = self._decode(chunk)
        if obj is None:
            return
        yield from self._process(obj)

    def parse_sse(self, chunk):
        obj = self._decode(chunk)
        if obj is None:
            return None
        result = None
        for partial in self._process(obj):
            if partial.get('type') == 'finished':
                result = {
                    'response_message_id': self.response_message_id,
                    'thinking_content': self.thinking_content,
                    'content': self.content,
                    'finished_status': self.finished_status,
                }
        return result


class DeepSeekAPI:
    BASE_URL = "https://chat.deepseek.com/api/v0"

    def __init__(self, auth_token: str):
        if not auth_token or not isinstance(auth_token, str):
            raise AuthenticationError("Invalid auth token provided")

        try:
            from importlib.metadata import distribution, PackageNotFoundError
            distribution('curl-cffi').version
        except PackageNotFoundError:
            print("\033[93mWarning: curl-cffi not found. Please install: pip install curl-cffi\033[0m", file=sys.stderr)

        self.auth_token = auth_token
        self.pow_solver = DeepSeekPOW()

        cookies_path = Path(__file__).parent / 'cookies.json'
        try:
            with open(cookies_path, 'r') as f:
                cookie_data = json.load(f)
                self.cookies = cookie_data.get('cookies', {})
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"\033[93mWarning: Could not load cookies from {cookies_path}: {e}\033[0m", file=sys.stderr)
            self.cookies = {}

    def _get_headers(self, pow_response: Optional[str] = None) -> Dict[str, str]:
        headers = {
            'accept': '*/*',
            'accept-language': 'en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,am;q=0.4,de;q=0.3',
            'authorization': f'Bearer {self.auth_token}',
            'content-type': 'application/json',
            'origin': 'https://chat.deepseek.com',
            'referer': 'https://chat.deepseek.com/',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36',
            'x-app-version': '20241129.1',
            'x-client-locale': 'en_US',
            'x-client-platform': 'web',
            'x-client-version': '1.0.0-always',
        }
        if pow_response:
            headers['x-ds-pow-response'] = pow_response
        return headers

    def _refresh_cookies(self) -> None:
        try:
            script_path = Path(__file__).parent / 'bypass.py'
            subprocess.run([sys.executable, script_path], check=True)
            time.sleep(2)
            cookies_path = Path(__file__).parent / 'cookies.json'
            with open(cookies_path, 'r') as f:
                self.cookies = json.load(f).get('cookies', {})
        except Exception as e:
            print(f"\033[93mWarning: Failed to refresh cookies: {e}\033[0m", file=sys.stderr)

    def _make_request(self, method: str, endpoint: str, json_data: Dict[str, Any], pow_required: bool = False) -> Any:
        url = f"{self.BASE_URL}{endpoint}"
        retry_count = 0
        max_retries = 2

        while retry_count < max_retries:
            try:
                headers = self._get_headers()
                if pow_required:
                    challenge = self._get_pow_challenge()
                    pow_response = self.pow_solver.solve_challenge(challenge)
                    headers = self._get_headers(pow_response)

                response = requests.request(
                    method=method, url=url, headers=headers, json=json_data,
                    cookies=self.cookies, impersonate='chrome120', timeout=None
                )

                if "<!DOCTYPE html>" in response.text and "Just a moment" in response.text:
                    print("\033[93mWarning: Cloudflare protection detected. Bypassing...\033[0m", file=sys.stderr)
                    if retry_count < max_retries - 1:
                        self._refresh_cookies()
                        retry_count += 1
                        continue

                if response.status_code == 401:
                    raise AuthenticationError("Invalid or expired authentication token")
                elif response.status_code == 429:
                    raise RateLimitError("API rate limit exceeded")
                elif response.status_code >= 500:
                    raise APIError(f"Server error occurred: {response.text}", response.status_code)
                elif response.status_code != 200:
                    raise APIError(f"API request failed: {response.text}", response.status_code)

                return response.json()

            except requests.exceptions.RequestException as e:
                raise NetworkError(f"Network error occurred: {str(e)}")
            except json.JSONDecodeError:
                raise APIError("Invalid JSON response from server")

        raise APIError("Failed to bypass Cloudflare protection after multiple attempts")

    def _get_pow_challenge(self) -> Dict[str, Any]:
        try:
            response = self._make_request(
                'POST', '/chat/create_pow_challenge',
                {'target_path': '/api/v0/chat/completion'}
            )
            return response['data']['biz_data']['challenge']
        except KeyError:
            raise APIError("Invalid challenge response format from server")

    def create_chat_session(self) -> str:
        """Creates a new chat session and returns the session ID"""
        try:
            response = self._make_request('POST', '/chat_session/create', {'character_id': None})
            return response['data']['biz_data']['id']
        except KeyError:
            raise APIError("Invalid session creation response format from server")

    def chat_completion(self, chat_session_id: str, prompt: str,
                        parent_message_id: Optional[str] = None,
                        thinking_enabled: bool = True,
                        search_enabled: bool = True) -> Generator[Dict[str, Any], None, None]:
        """
        Send a message and get streaming response.
        """
        if not prompt or not isinstance(prompt, str):
            raise ValueError("Prompt must be a non-empty string")
        if not chat_session_id or not isinstance(chat_session_id, str):
            raise ValueError("Chat session ID must be a non-empty string")

        json_data = {
            'chat_session_id': chat_session_id,
            'parent_message_id': parent_message_id,
            'prompt': prompt,
            'ref_file_ids': [],
            'thinking_enabled': thinking_enabled,
            'search_enabled': search_enabled,
        }

        try:
            headers = self._get_headers(
                pow_response=self.pow_solver.solve_challenge(self._get_pow_challenge())
            )

            response = requests.post(
                f"{self.BASE_URL}/chat/completion",
                headers=headers, json=json_data, cookies=self.cookies,
                impersonate='chrome120', stream=True, timeout=None
            )

            if response.status_code != 200:
                error_text = next(response.iter_lines(), b'').decode('utf-8', 'ignore')
                if response.status_code == 401:
                    raise AuthenticationError("Invalid or expired authentication token")
                elif response.status_code == 429:
                    raise RateLimitError("API rate limit exceeded")
                else:
                    raise APIError(f"API request failed: {error_text}", response.status_code)

            parser = SSEMessageParser()
            for chunk in response.iter_lines():
                try:
                    for partial in parser.parse_sse_streaming(chunk):
                        yield partial
                        if partial.get('type') == 'finished':
                            return
                except Exception as e:
                    raise APIError(f"Error parsing response chunk: {str(e)}")

        except requests.exceptions.RequestException as e:
            raise NetworkError(f"Network error occurred during streaming: {str(e)}")

    def chat_completion_with_messages(self, chat_session_id: str, messages: list,
                                      thinking_enabled: bool = True,
                                      search_enabled: bool = True) -> Generator[Dict[str, Any], None, None]:
        """
        Send a message and get streaming response (messages format).
        """
        if not messages or not isinstance(messages, list):
            raise ValueError("messages must be a non-empty list")

        json_data = {
            'chat_session_id': chat_session_id,
            'prompt': messages,
            'thinking_enabled': thinking_enabled,
            'search_enabled': search_enabled,
        }

        try:
            headers = self._get_headers(
                pow_response=self.pow_solver.solve_challenge(self._get_pow_challenge())
            )

            response = requests.post(
                f"{self.BASE_URL}/chat/completion",
                headers=headers, json=json_data, cookies=self.cookies,
                impersonate='chrome120', stream=True, timeout=None
            )

            if response.status_code != 200:
                error_text = next(response.iter_lines(), b'').decode('utf-8', 'ignore')
                if response.status_code == 401:
                    raise AuthenticationError("Invalid or expired authentication token")
                elif response.status_code == 429:
                    raise RateLimitError("API rate limit exceeded")
                else:
                    raise APIError(f"API request failed: {error_text}", response.status_code)

            parser = SSEMessageParser()
            for chunk in response.iter_lines():
                try:
                    for partial in parser.parse_sse_streaming(chunk):
                        yield partial
                        if partial.get('type') == 'finished':
                            return
                except Exception as e:
                    raise APIError(f"Error parsing response chunk: {str(e)}")

        except requests.exceptions.RequestException as e:
            raise NetworkError(f"Network error occurred during streaming: {str(e)}")