# Homework 4: cybersecurity MCP agent

FAU ID: zreviere2022

The agent connects through FastMCP to the custom server in `mcp-server.py`
and the published Model Context Protocol Fetch reference server:
https://github.com/modelcontextprotocol/servers/tree/main/src/fetch
Fetch runs as a separate stdio process via `uv tool run`, pinned to 2025.4.7,
with `mcp<2` because that published version uses the MCP 1.x API.
Its dependencies are isolated from the agent's environment. The agent dependencies
are locked in `uv.lock`; the isolated Fetch tool resolves its own transitive dependencies.

## Setup and execution

From the repository root:

```sh
uv --cache-dir hw4/.uv-cache sync --project hw4 --locked
cp hw4/.env.example hw4/.env
# Edit hw4/.env to supply your own API key and desired Gemini model.
uv --cache-dir hw4/.uv-cache run --project hw4 --env-file hw4/.env python hw4/agent.py
```

Alternatively export GOOGLE_API_KEY and optionally GOOGLE_MODEL in your shell,
then omit `--env-file`. Secrets are never embedded in Python code. `.env` is ignored.
The default model is `gemini-flash-latest`; choose a model available to your account.
Model requests may incur API usage charges.

```sh
# Real MCP integration check, no Gemini key needed:
uv --cache-dir hw4/.uv-cache run --project hw4 python hw4/agent.py --smoke-test
# Supply a different assessment task:
uv --cache-dir hw4/.uv-cache run --project hw4 --env-file hw4/.env python hw4/agent.py "Assess public SSH on port 22 without MFA."
# Offline orchestration tests:
uv --cache-dir hw4/.uv-cache run --project hw4 python -m unittest discover -s hw4 -p 'test_*.py'
```

## Workflow

For the default task, Gemini identifies port 3389, explains MFA, retrieves CISA's
ransomware guidance through Fetch, chooses and justifies likelihood/impact ratings,
and asks the custom server to calculate a risk score. It then writes a report with
mitigations and source URLs. Tool names have custom_ and fetch_ prefixes to avoid
collisions. Only the four required tools are exposed to Gemini. Tool activity is
printed to stderr for a screencast; the report goes to stdout.

The agent discovers tool schemas and sends them to Gemini as function declarations.
It executes requested calls, returns structured results, and preserves the full
model response (including any thought signatures) in the conversation history.
The loop stops after 12 model rounds. Missing successful tool calls are explicitly
marked in the final report. Errors and unavailable sources must not be presented
as successful evidence. Fetched content is treated as untrusted reference data.

Port associations do not establish that a service is running. The risk matrix is
an illustrative classroom score, not CVSS. The smoke test uses fixed ratings 4 and 5;
the live agent chooses ratings and must explain them. Network access is required
for dependency downloads, Fetch, and Gemini; a live Gemini run also requires a key.

## Validation

Syntax checks and four offline orchestration tests passed. The real two-server
smoke test successfully identified RDP, explained MFA, fetched the CISA guide,
and calculated 4 x 5 = 20 (critical). Missing credentials produce an actionable
error before servers start. After the function-response fix, a live Gemini run
accepted the tool results and requested another Fetch call without the role error.
The run later stopped with Gemini's 503 UNAVAILABLE (high demand), so a complete
live final report was not verified. Function responses use `role="user"` and retain
the function name, call ID, and structured result. The SDK may log protocol
negotiation warnings while reconnecting to the older Fetch server; the integration
test completed successfully after that negotiation.
