import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from utils.config import SYSTEM_PROMPT

load_dotenv(Path(__file__).with_name(".env"))

logger = logging.getLogger(__name__)

MODEL_NAME = "gemini-3.1-flash-lite"
MAX_HISTORY_MESSAGES = 20
EMPTY_RESPONSE = "I'm sorry, I couldn't generate a response just now. Please try again."
API_ERROR_RESPONSE = (
    "I'm sorry, I'm having trouble reaching the coaching service right now. "
    "Check the terminal for the error details, then try again."
)

class ChatAssistant:
    
    def __init__(self, api_key=None):
        resolved_api_key = (
            api_key if api_key is not None else os.getenv("GEMINI_API_KEY")
        )
        if isinstance(resolved_api_key, str):
            resolved_api_key = resolved_api_key.strip()
        if not resolved_api_key:
            raise ValueError(
                "Gemini API key is required. Set GEMINI_API_KEY or pass api_key."
            )
        if resolved_api_key == "your_gemini_api_key_here":
            raise ValueError(
                "Replace the placeholder in .env with a valid Gemini API key."
            )
        self._api_key = resolved_api_key
        self.client = genai.Client(api_key=resolved_api_key)
        self.conversation_history = []
    
    def add_to_history(self, role, content):
        if role not in {"user", "assistant", "system"}:
            raise ValueError("role must be 'user', 'assistant', or 'system'")
        self.conversation_history.append({"role": role, "content": str(content)})
        self.conversation_history = self.conversation_history[-MAX_HISTORY_MESSAGES:]
    
    def get_conversation_context(self):
        messages = [f"System: {SYSTEM_PROMPT}"]
        messages.extend(
            f"{message['role'].capitalize()}: {message['content']}"
            for message in self.conversation_history
        )
        return "\n\n".join(messages)
    
    def chat(self, user_input):
        if not isinstance(user_input, str) or not user_input.strip():
            raise ValueError("user_input must be a non-empty string")

        previous_history = self.conversation_history.copy()
        self.add_to_history("user", user_input.strip())
        fullprompt = self.get_conversation_context()
        try:
            response = self.client.models.generate_content(
                model=MODEL_NAME,
                contents=fullprompt,
            )
        except Exception as exc:
            self.conversation_history = previous_history
            error_detail = str(exc).replace(self._api_key, "[REDACTED]")
            logger.error(
                "Gemini API request failed (%s): %s",
                type(exc).__name__,
                error_detail,
            )
            return API_ERROR_RESPONSE

        response_text = getattr(response, "text", None)
        if not isinstance(response_text, str) or not response_text.strip():
            response_text = EMPTY_RESPONSE
        else:
            response_text = response_text.strip()
        self.add_to_history("assistant", response_text)
        return response_text
    
    def reset_conversation(self):
        self.conversation_history.clear()

def answer_fitness_query(query, assistant=None):
    try:
        if assistant is None:
            assistant = ChatAssistant()
        return assistant.chat(query)
    except Exception as exc:
        logger.error(
            "Unable to answer fitness query (%s): %s",
            type(exc).__name__,
            str(exc),
        )
        return API_ERROR_RESPONSE

def main():
    """Main function to demonstrate the fitness coach assistant."""
    print("Coach Alex - Personal Fitness Coach")
    print("Type 'quit' to exit, 'reset' to start new conversation")
    print("=" * 50)
    try:
        assistant = ChatAssistant()
    except ValueError as exc:
        print(f"Unable to start Coach Alex: {exc}")
        return

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == "quit":
            print("Coach Alex: Keep taking small steps toward your goals. Take care!")
            break
        if user_input.lower() == "reset":
            assistant.reset_conversation()
            print("Coach Alex: Conversation reset. What would you like to work on?")
            continue
        if not user_input:
            continue
        print(f"Coach Alex: {answer_fitness_query(user_input, assistant)}")

if __name__ == "__main__":
    main()
