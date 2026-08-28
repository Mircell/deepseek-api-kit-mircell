import requests
import json
import time
import uuid
from typing import List, Dict, Any, Optional, Generator
from common.api import DeepSeekAPI
from common.config import DEEPSEEK_API_KEY

class DeepSeekAdapter(DeepSeekAPI):
    """
    Adapter for DeepSeek API that supports tool calling.
    Extends the base DeepSeekAPI to send tools and tool_choice in the request.
    """
    
    def chat_completion(
        self,
        chat_id: str,
        messages: List[Dict[str, Any]],
        parent_message_id: Optional[str] = None,
        thinking_enabled: bool = True,
        search_enabled: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Send a chat completion request to DeepSeek API with support for tools.
        
        Args:
            chat_id: Session ID
            messages: List of messages in OpenAI format (role, content)
            parent_message_id: Parent message ID for threading
            thinking_enabled: Enable thinking mode
            search_enabled: Enable search
            tools: List of tools definitions
            tool_choice: Tool choice strategy
            
        Yields:
            Chunks of the response with 'type', 'delta', etc.
        """
        # Build the payload
        payload = {
            "model": "deepseek-chat",
            "messages": messages,
            "stream": True,
            "thinking": {"type": "enabled" if thinking_enabled else "disabled"},
        }
        
        if search_enabled:
            payload["search"] = {"enabled": True}
        
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"
        
        # Headers
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Make the request
        response = requests.post(
            "https://chat.deepseek.com/api/v0",
            json=payload,
            headers=headers,
            stream=True
        )
        
        if response.status_code != 200:
            raise Exception(f"DeepSeek API error: {response.status_code} - {response.text}")
        
        # Process the stream
        buffer = ""
        for line in response.iter_lines():
            if not line:
                continue
            line = line.decode('utf-8')
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    # Extract delta and finish_reason
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    finish_reason = chunk.get("choices", [{}])[0].get("finish_reason")
                    
                    # Check for tool calls in delta
                    if "tool_calls" in delta:
                        # DeepSeek may send tool_calls in the delta
                        tool_calls = delta.get("tool_calls", [])
                        for tool_call in tool_calls:
                            # Yield tool call as a special chunk
                            yield {
                                "type": "tool_call",
                                "tool_call": tool_call
                            }
                    
                    # Check for content
                    if "content" in delta and delta["content"]:
                        # Check if content contains DSML tags
                        content = delta["content"]
                        # Yield as content chunk (will be parsed later by dsml_parser)
                        yield {
                            "type": "content",
                            "delta": content
                        }
                    
                    # Check for reasoning_content (thinking)
                    if "reasoning_content" in delta and delta["reasoning_content"]:
                        yield {
                            "type": "thinking",
                            "delta": delta["reasoning_content"]
                        }
                    
                    # Check for finish reason
                    if finish_reason:
                        # Extract response_message_id if available
                        response_message_id = chunk.get("response_message_id") or chunk.get("id")
                        yield {
                            "type": "finished",
                            "response_message_id": response_message_id
                        }
                        
                except json.JSONDecodeError:
                    continue
        
        # Final yield to signal end
        yield {
            "type": "finished",
            "response_message_id": None
        }