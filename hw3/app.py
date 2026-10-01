"""A Gemini-powered LangChain assistant for cybersecurity coursework."""

import hashlib
import ipaddress
import os
import sys
import traceback
from code import InteractiveInterpreter
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.state import CompiledStateGraph


SYSTEM_PROMPT = """You are a cybersecurity teaching assistant for a college course.
Explain your reasoning clearly and use tools when calculations or exact results
are needed. Use sha256_text for text hashes and inspect_ip for IP classification.
IP classification alone does not indicate whether an address is malicious.
Use the Python REPL for educational calculations; print results to see them.
Keep work within the user's authorized coursework and lab environment.
Do not access credentials or modify files unless explicitly requested.
"""


@tool
def sha256_text(text: str) -> str:
    """Return the SHA-256 hex digest of the exact UTF-8 text, including whitespace.

    Useful for integrity demonstrations. This hashes text, not file contents,
    and is not a password-storage scheme. Do not supply secrets or passwords.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@tool
def inspect_ip(address: str) -> dict:
    """Classify a single IPv4 or IPv6 address without making network requests.

    Return normalized notation and addressing properties for log analysis.
    Accept an IP literal, not a hostname, URL, or CIDR range. These properties
    do not establish reputation, reachability, or whether an address is safe.
    """
    try:
        ip = ipaddress.ip_address(address.strip())
    except ValueError:
        return {"error": "Provide a valid IPv4 or IPv6 address literal."}
    return {
        "address": str(ip),
        "version": ip.version,
        "is_private": ip.is_private,
        "is_global": ip.is_global,
        "is_loopback": ip.is_loopback,
        "is_link_local": ip.is_link_local,
        "is_multicast": ip.is_multicast,
        "is_reserved": ip.is_reserved,
        "is_unspecified": ip.is_unspecified,
    }


def make_python_repl():
    """Create a Python tool with a separate persistent namespace for each agent.

    Execution occurs in this process with the user's permissions. This is not
    a sandbox; use only in a trusted local lab. Output and errors are captured.
    """
    interpreter = InteractiveInterpreter()

    @tool
    def python_repl(code: str) -> str:
        """Execute Python code in a persistent local REPL; use print for output.

        Variables persist across calls. Submit complete Python statements,
        without Markdown fences. This tool runs real code with local access.
        """
        output = StringIO()
        with redirect_stdout(output), redirect_stderr(output):
            try:
                incomplete = interpreter.runsource(code, symbol="exec")
            except SystemExit:
                return "Error: SystemExit is not allowed in this REPL."
        if incomplete:
            return "Error: incomplete Python code; submit a complete block."
        return output.getvalue() or "Code executed successfully (no output)."

    return python_repl


def build_agent() -> CompiledStateGraph:
    """Load environment configuration and build the Gemini tool-calling agent.

    GOOGLE_API_KEY is required (GEMINI_API_KEY is also accepted). An optional
    GEMINI_MODEL selects the model. A local .env supplies missing variables
    without overriding variables already set in the shell.
    """
    load_dotenv(Path(__file__).with_name(".env"))
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key or not api_key.strip():
        raise ValueError("Set GOOGLE_API_KEY in your environment or .env file.")

    model = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL") or "gemini-3.5-flash-lite",
        api_key=api_key,
        vertexai=False,
        timeout=60,
        max_retries=2,
    )
    return create_agent(
        model=model,
        tools=[make_python_repl(), sha256_text, inspect_ip],
        system_prompt=SYSTEM_PROMPT,
    )


def print_request_error(exc: Exception) -> None:
    """Print the full traceback and provider details, including wrapped errors.

    SDKs expose different status and payload attributes. Print those present
    on each exception in the cause/context chain to standard error.
    """
    print(f"Request failed ({type(exc).__name__}): {exc}", file=sys.stderr)
    traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
    current = exc
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        print(f"Error details ({type(current).__name__}): {current}", file=sys.stderr)
        for name in ("code", "status", "status_code", "message", "details",
                     "error_details", "response_json", "body"):
            value = getattr(current, name, None)
            if value is not None and not callable(value):
                print(f"  {name}: {value}", file=sys.stderr)
        response = getattr(current, "response", None)
        if response is not None:
            for name in ("status_code", "reason_phrase", "text"):
                value = getattr(response, name, None)
                if value is not None:
                    print(f"  response.{name}: {value}", file=sys.stderr)
        current = (current.__cause__ if current.__cause__ is not None
                   else current.__context__)


def main() -> int:
    """Run an interactive chat, retaining conversation history until exit."""
    try:
        agent = build_agent()
    except ValueError as exc:
        print(f"Configuration error: {exc}")
        return 1

    print("Cybersecurity assistant — type 'exit' or 'quit' to finish.")
    print("The Python REPL executes local code. Use a trusted lab environment.")
    messages = []
    while True:
        try:
            prompt = input("\nYou: ").strip()
            if prompt.lower() in {"exit", "quit"}:
                break
            if not prompt:
                continue
            result = agent.invoke(
                {"messages": [*messages, {"role": "user", "content": prompt}]},
                config={"recursion_limit": 20},
            )
            messages = result["messages"]
            print(f"\nAssistant: {messages[-1].text}")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        except Exception as exc:
            print_request_error(exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
