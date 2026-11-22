import os


def load_system_prompt():
    """Load the system prompt from prompt.txt file."""
    prompt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompt.txt")
    try:
        with open(prompt_path, encoding="utf-8") as prompt_file:
            prompt = prompt_file.read().strip()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"System prompt file not found: {prompt_path}") from exc

    if not prompt:
        raise ValueError(f"System prompt file is empty: {prompt_path}")
    return prompt

def get_system_prompt():
    """Get the system prompt for the fitness coach."""
    return load_system_prompt()

SYSTEM_PROMPT = get_system_prompt()
