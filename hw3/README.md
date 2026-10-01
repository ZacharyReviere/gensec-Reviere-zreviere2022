# Cybersecurity coursework agent

An interactive LangChain agent using Google Gemini and three tools:

- `python_repl`: runs Python calculations and retains variables during the chat.
- `sha256_text`: hashes exact UTF-8 text for integrity demonstrations.
- `inspect_ip`: classifies IPv4/IPv6 literals locally without network probes.

## Run

Install the dependencies with `uv sync`. Copy `.env.example` to `.env` and set
`GOOGLE_API_KEY` to your Google AI Studio API key, or export it in your shell.
`GEMINI_API_KEY` is accepted as a fallback. Optionally set `GEMINI_MODEL` to a
Gemini model available to your account; the default is `gemini-3.5-flash-lite`.
If you previously set `GEMINI_MODEL` in your shell or `.env`, update it as well;
that setting overrides the default in `app.py`.

```sh
uv run python app.py
```

Type `exit` or `quit` to stop. Example prompts:

- “Compute the SHA-256 hash of the exact text hello.”
- “Inspect 192.168.1.10 and explain the returned properties.”
- “Use Python to calculate how many addresses are in an IPv4 /24 subnet.”

The REPL executes real Python with your local permissions; it is not sandboxed
and has no execution timeout. Use a disposable lab environment for experiments.
Prompts and tool outputs are sent to Gemini, so use synthetic coursework data
and do not submit credentials. SHA-256 here is not password hashing, and IP
classification does not determine whether an address is malicious.

The implementation follows LangChain's [agent API](https://reference.langchain.com/python/langchain/agents/factory/create_agent)
and [Gemini integration](https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai).
