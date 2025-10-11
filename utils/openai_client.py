from typing import Optional
from openai import OpenAI

def get_openai_client(api_key: str, org: Optional[str] = None) -> OpenAI:
    # Construct a client without leaking keys in code
    if not api_key:
        raise ValueError("Missing OpenAI API key")
    kwargs = {"api_key": api_key}
    if org:
        kwargs["organization"] = org
    return OpenAI(**kwargs)