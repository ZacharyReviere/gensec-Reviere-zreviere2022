"""Test orchestration, failure reporting, and loop bounds without API calls."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

import agent
from google.genai import types


def reply(*calls, text=None):
    """Build a realistic SDK response for scripted model turns."""
    parts = [types.Part(function_call=types.FunctionCall(name=n, args=a, id=f"call-{n}"))
             for n, a in calls]
    if text is not None:
        parts.append(types.Part.from_text(text=text))
    return types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(role="model", parts=parts))])


class AgentTests(unittest.IsolatedAsyncioTestCase):
    """Verify tool dispatch and model history with deterministic fake backends."""

    def setUp(self):
        """Create a discovered tool catalog and success-returning MCP client."""
        self.client = SimpleNamespace(
            list_tools=AsyncMock(return_value=[SimpleNamespace(
                name=name, description=name,
                inputSchema={"type": "object", "properties": {}})
                for name in agent.REQUIRED_TOOLS]),
            call_tool=AsyncMock(return_value=SimpleNamespace(is_error=False, content=[])),
        )

    async def test_coordinates_all_tools_and_preserves_history(self):
        """Route both servers' tools and return results to the next model turn."""
        responses = [reply(("custom_check_port", {"port": 3389}),
                           ("custom_explain_security_term", {"term": "MFA"})),
                     reply(("fetch_fetch", {"url": agent.GUIDANCE_URL})),
                     reply(("custom_calculate_risk_score", {"likelihood": 4, "impact": 5})),
                     reply(text="Assessment complete")]
        model = SimpleNamespace(models=SimpleNamespace(
            generate_content=AsyncMock(side_effect=responses)))
        result = await agent.assess(self.client, model, "test-model", "Assess RDP")
        self.assertEqual(result, "Assessment complete")
        self.assertEqual(self.client.call_tool.await_count, 4)
        history = model.models.generate_content.call_args.kwargs['contents']
        self.assertEqual([c.role for c in history],
                         ['user', 'model', 'user', 'model', 'user', 'model', 'user', 'model'])
        for model_turn, result_turn in zip(history[1::2], history[2::2]):
            calls = [part.function_call for part in model_turn.parts]
            results = [part.function_response for part in result_turn.parts]
            self.assertEqual(len(calls), len(results))
            for call, result in zip(calls, results):
                self.assertEqual(result.name, call.name)
                self.assertEqual(result.id, call.id)
                self.assertEqual(result.response, {'is_error': False, 'content': []})

    async def test_premature_report_is_marked_incomplete(self):
        """Do not silently accept a report without successful tool use."""
        model = SimpleNamespace(models=SimpleNamespace(
            generate_content=AsyncMock(return_value=reply(text="Unsupported report"))))
        result = await agent.assess(self.client, model, "test", "task")
        self.assertIn("Workflow incomplete", result)

    async def test_unknown_tool_and_connection_error(self):
        """Reject undiscovered tools and expose connection failures as data."""
        result = await agent.invoke_tool(self.client, 'unknown', {}, agent.REQUIRED_TOOLS)
        self.assertIn('error', result)
        self.client.call_tool.assert_not_awaited()
        self.client.call_tool.side_effect = TimeoutError('unavailable')
        result = await agent.invoke_tool(self.client, 'fetch_fetch', {}, agent.REQUIRED_TOOLS)
        self.assertIn('TimeoutError', result['error'])

    async def test_round_limit_and_missing_tools(self):
        """Stop repeated model calls and reject incomplete server catalogs."""
        model = SimpleNamespace(models=SimpleNamespace(generate_content=AsyncMock(
            return_value=reply(('custom_check_port', {'port': 3389})))))
        with self.assertRaisesRegex(RuntimeError, 'after 2 rounds'):
            await agent.assess(self.client, model, 'test', 'task', max_rounds=2)
        self.client.list_tools.return_value = []
        with self.assertRaisesRegex(RuntimeError, 'Required MCP tools unavailable'):
            await agent.assess(self.client, model, 'test', 'task')


if __name__ == '__main__':
    unittest.main()
