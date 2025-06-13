# -*- coding: utf-8 -*-
"""
Genesis.py - Universal GenAI Interface
Enhanced version with bug fixes and optimizations

@author: Victor Mctrix (Enhanced)
@version: 0.1.1
"""

import json
import base64
import requests
from typing import List, Dict, Optional, Union
import os
from markitdown import MarkItDown

class Genesis:
    """Universal GenAI interface for OpenRouter API with enhanced error handling and optimization"""
    
    def __init__(self, key: str, httpRef: str = "", projTitle: str = ""):
        """
        Initialize Genesis instance
        
        Args:
            key: OpenRouter API key
            httpRef: HTTP referer for rankings (optional)
            projTitle: Project title for rankings (optional)
        """
        self.key = key
        self.httpRef = httpRef
        self.projTitle = projTitle
        self.systemContents = []
        self.userContents = []
        self.last_response = None
        self.last_error = None
        
    # ---Helper Functions---
    def ImgToBase64(self, filename: str) -> Optional[str]:
        """
        ✅ FIXED: Convert image file to base64 string with better error handling
        
        Args:
            filename: Path to image file
            
        Returns:
            Base64 encoded image string or None if error
        """
        try:
            # Check if file exists
            if not os.path.exists(filename):
                print(f"Error: File {filename} not found")
                return None
                
            # Determine file type
            filename_lower = filename.lower()
            if any(ext in filename_lower for ext in ['jpg', 'jpeg']):
                fileType = "jpeg"
            elif 'png' in filename_lower:
                fileType = "png"
            elif 'webp' in filename_lower:
                fileType = "webp"
            elif 'gif' in filename_lower:
                fileType = "gif"
            else:
                print(f"Error: Unsupported file type for {filename}")
                return None
            
            # Read and encode file
            with open(filename, 'rb') as image_file:
                encoded = f"data:image/{fileType};base64,{base64.b64encode(image_file.read()).decode('utf-8')}"
            
            return encoded
            
        except Exception as e:
            print(f"Error converting image {filename}: {e}")
            return None
    
    def FileToMD(self, filename: str) -> Optional[str]:
        """
        ✅ ENHANCED: Convert file to markdown string with error handling
        
        Args:
            filename: Path to file
            
        Returns:
            Markdown content or None if error
        """
        try:
            if not os.path.exists(filename):
                print(f"Error: File {filename} not found")
                return None
                
            md_converter = MarkItDown()
            result = md_converter.convert(filename)
            return str(result.text_content)
            
        except Exception as e:
            print(f"Error converting file {filename} to markdown: {e}")
            return None
    
    def CreateDict(self, dicttype: str, value: str) -> Dict:
        """
        ✅ FIXED: Create content dictionary with proper image handling
        
        Args:
            dicttype: Type of content ('text', 'image_url')
            value: Content value
            
        Returns:
            Formatted content dictionary
        """
        if dicttype == "image_url":
            # Handle both local files and URLs
            if not value.startswith(('http://', 'https://', 'data:image')):
                # Local file - convert to base64
                encoded_value = self.ImgToBase64(value)
                if encoded_value is None:
                    return {"type": "text", "text": f"[Error: Could not load image {value}]"}
                value = encoded_value
            
            return {
                "type": "image_url",
                "image_url": {"url": value}
            }
        else:
            return {
                "type": "text",
                "text": value
            }
    
    def CheckUserContentsExist(self) -> bool:
        """Check if user contents exist"""
        return len(self.userContents) > 0
    
    def CheckSystemContentsExist(self) -> bool:
        """Check if system contents exist"""
        return len(self.systemContents) > 0
    
    # ---Public Functions---
    def PushMsgToSystem(self, value: str) -> None:
        """Add text message to system contents"""
        self.systemContents.append(self.CreateDict("text", value))
        
    def PushFileToSystem(self, filename: str) -> bool:
        """
        Add file content to system contents
        
        Returns:
            True if successful, False otherwise
        """
        content = self.FileToMD(filename)
        if content:
            self.systemContents.append(self.CreateDict("text", content))
            return True
        return False
    
    def DebugCheckSystem(self) -> None:
        """Debug: Print system contents"""
        print("System Contents:")
        for i, content in enumerate(self.systemContents):
            print(f"  {i}: {content}")
        
    def PopMsgOfSystem(self) -> bool:
        """Remove last message from system contents"""
        if self.systemContents:
            self.systemContents.pop()
            return True
        return False
        
    def PushMsgToUser(self, dicttype: str, value: str) -> None:
        """Add message to user contents"""
        self.userContents.append(self.CreateDict(dicttype, value))
    
    def PushImgToUser(self, value: str, fileType: str = None) -> None:
        """
        ✅ FIXED: Add image to user contents with proper handling
        
        Args:
            value: Image path, URL, or base64 string
            fileType: File type (deprecated, auto-detected)
        """
        if value.startswith('data:image'):
            # Already base64 encoded
            self.userContents.append({
                "type": "image_url",
                "image_url": {"url": value}
            })
        else:
            # File path or URL
            self.userContents.append(self.CreateDict("image_url", value))
        
    def PushFileToUser(self, filename: str) -> bool:
        """
        ✅ FIXED: Add file content to user contents (was typo UserContents)
        
        Returns:
            True if successful, False otherwise
        """
        content = self.FileToMD(filename)
        if content:
            self.userContents.append(self.CreateDict("text", content))
            return True
        return False
        
    def DebugCheckUser(self) -> None:
        """Debug: Print user contents"""
        print("User Contents:")
        for i, content in enumerate(self.userContents):
            print(f"  {i}: {content}")
        
    def PopMsgOfUser(self) -> bool:
        """Remove last message from user contents"""
        if self.userContents:
            self.userContents.pop()
            return True
        return False
    
    def ClearAll(self) -> None:
        """✅ NEW: Clear all contents"""
        self.systemContents.clear()
        self.userContents.clear()
        self.last_response = None
        self.last_error = None
    
    def TXRX(self, LLM: str = "openai/gpt-4o-2024-11-20", provider: List[str] = None, 
             max_tokens: int = None, temperature: float = None) -> Optional[str]:
        """
        ✅ ENHANCED: Send message to AI with better error handling and options
        
        Args:
            LLM: Model name
            provider: List of preferred providers
            max_tokens: Maximum tokens in response
            temperature: Response randomness (0.0-1.0)
            
        Returns:
            AI response or None if error
        """
        if not self.CheckSystemContentsExist() or not self.CheckUserContentsExist():
            error_msg = "Error in TXRX(): missing systemContent or userContent."
            print(error_msg)
            self.last_error = error_msg
            return None
        
        if provider is None:
            provider = ["OpenAI"]
        
        # Build request data
        request_data = {
            "model": LLM,
            "messages": [
                {
                    "role": "system",
                    "content": self.systemContents
                },
                {
                    "role": "user", 
                    "content": self.userContents
                }
            ],
            "provider": {
                "order": provider
            }
        }
        
        # Add optional parameters
        if max_tokens:
            request_data["max_tokens"] = max_tokens
        if temperature is not None:
            request_data["temperature"] = temperature
        
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.key}",
                    "HTTP-Referer": f"{self.httpRef}",
                    "X-Title": f"{self.projTitle}",
                    "Content-Type": "application/json"
                },
                data=json.dumps(request_data, ensure_ascii=False).encode("utf-8"),
                timeout=30  # Add timeout
            )
            
            self.last_response = response
            
            if response.status_code != 200:
                error_msg = f"HTTP Error {response.status_code}: {response.text}"
                print(error_msg)
                self.last_error = error_msg
                return None
            
            response_data = response.json()
            
            if "error" in response_data:
                error_msg = f"API Error: {response_data['error']}"
                print(error_msg)
                self.last_error = error_msg
                return None
            
            if "choices" in response_data and len(response_data["choices"]) > 0:
                content = response_data["choices"][0]["message"]["content"]
                self.last_error = None
                return content
            else:
                error_msg = "Error: No response content received"
                print(error_msg)
                self.last_error = error_msg
                return None
                
        except requests.exceptions.Timeout:
            error_msg = "Error: Request timeout"
            print(error_msg)
            self.last_error = error_msg
            return None
        except requests.exceptions.RequestException as e:
            error_msg = f"Error: Request failed - {e}"
            print(error_msg)
            self.last_error = error_msg
            return None
        except json.JSONDecodeError as e:
            error_msg = f"Error: Invalid JSON response - {e}"
            print(error_msg)
            self.last_error = error_msg
            return None
        except Exception as e:
            error_msg = f"Unexpected error: {e}"
            print(error_msg)
            self.last_error = error_msg
            return None
    
    # ---String Representation---
    def __str__(self) -> str:
        """✅ FIXED: String representation of the object"""
        return f"Genesis(project='{self.projTitle}', system_msgs={len(self.systemContents)}, user_msgs={len(self.userContents)})"
    
    def __repr__(self) -> str:
        """Enhanced representation for debugging"""
        name = "Genesis v0.2.0 (Enhanced)\n\n"
        
        if not self.CheckSystemContentsExist():
            return name + "Error: missing element in systemContents."
        elif not self.CheckUserContentsExist():
            return name + "Error: missing element in userContents."
        else:
            system_preview = self.systemContents[0].get("text", "")[:100] + "..." if len(self.systemContents[0].get("text", "")) > 100 else self.systemContents[0].get("text", "")
            user_preview = self.userContents[0].get("text", "")[:100] + "..." if len(self.userContents[0].get("text", "")) > 100 else self.userContents[0].get("text", "")
            
            return (name + 
                   f"System Rule: {system_preview}\n\n" + 
                   f"User Message: {user_preview}\n\n" +
                   f"Last Error: {self.last_error}")
        
    # ---Arithmetic Functions---
    def __add__(self, nextItem):
        """✅ ENHANCED: Combine two Genesis objects"""
        if not isinstance(nextItem, Genesis):
            raise TypeError("Can only add Genesis objects together")
            
        sumObj = Genesis(self.key, self.httpRef, self.projTitle)
        sumObj.systemContents = self.systemContents + nextItem.systemContents
        sumObj.userContents = self.userContents + nextItem.userContents
        return sumObj
            
    def __sub__(self, nextItem):
        """✅ FIXED: Remove duplicate content (more logical than arithmetic subtraction)"""
        if not isinstance(nextItem, Genesis):
            raise TypeError("Can only subtract Genesis objects")
            
        subtractObj = Genesis(self.key, self.httpRef, self.projTitle)
        
        # Remove duplicate system contents
        subtractObj.systemContents = [content for content in self.systemContents 
                                    if content not in nextItem.systemContents]
        
        # Remove duplicate user contents  
        subtractObj.userContents = [content for content in self.userContents
                                  if content not in nextItem.userContents]
        
        return subtractObj
    
    # ---Comparison Functions---
    def __eq__(self, nextItem) -> bool:
        """Check equality"""
        if not isinstance(nextItem, Genesis):
            return False
        return (self.key == nextItem.key and 
                self.httpRef == nextItem.httpRef and 
                self.projTitle == nextItem.projTitle and 
                self.systemContents == nextItem.systemContents and 
                self.userContents == nextItem.userContents)
    
    def __lt__(self, nextItem) -> bool:
        """Compare by total content length"""
        if not isinstance(nextItem, Genesis):
            raise TypeError("Can only compare Genesis objects")
        return (len(self.systemContents) + len(self.userContents) < 
                len(nextItem.systemContents) + len(nextItem.userContents))
    
    def __gt__(self, nextItem) -> bool:
        """Compare by total content length"""
        if not isinstance(nextItem, Genesis):
            raise TypeError("Can only compare Genesis objects")
        return (len(self.systemContents) + len(self.userContents) > 
                len(nextItem.systemContents) + len(nextItem.userContents))

# ✅ ENHANCED: Better example with environment variable for API key
def main():
    """Example usage with enhanced error handling"""
    
    # ✅ SECURITY: Use environment variable for API key
    key = os.getenv('OPENROUTER_API_KEY', 'your-api-key-here')
    if key == 'your-api-key-here':
        print("⚠️  Warning: Using default API key. Set OPENROUTER_API_KEY environment variable.")
    
    httpRef = "https://your-website.com"  # Optional: Your website for rankings
    projectTitle = "Investment Analysis"
    
    try:
        # Create Genesis instance
        AI = Genesis(key, httpRef, projectTitle)
        
        # Set system prompt
        AI.PushMsgToSystem("You are an investment advisor. Provide specific investment advice and stock recommendations based on market analysis.")
        
        # Set user prompt
        AI.PushMsgToUser("text", "As a value investor, what stocks would you recommend for long-term investment in 2024?")
        
        print("🤖 Sending request to AI...")
        
        # Get response with enhanced parameters
        response = AI.TXRX(
            LLM="openai/gpt-4o-2024-11-20", 
            provider=["OpenAI"],
            max_tokens=1000,
            temperature=0.7
        )
        
        if response:
            print(f"\n🎯 AI Response:\n{response}")
        else:
            print(f"❌ Failed to get response. Last error: {AI.last_error}")
            
    except Exception as e:
        print(f"❌ Unexpected error in main: {e}")
    
    finally:
        # Clean up
        if 'AI' in locals():
            del AI

if __name__ == "__main__":
    main()
