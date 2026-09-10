import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
import httpx
from anthropic import AuthenticationError, RateLimitError, APIConnectionError
from anthropic.types import Message, TextBlock, ThinkingBlock


from graph import run
from claude import generate_answer
from steps import initial_state
from test_workflows import detector_result


def response_text(content):
    return Message(id="test", type="message", role="assistant", model="test-model",
                   content=[TextBlock(type="text", text=content)],
                   stop_reason="end_turn", usage={"input_tokens": 1, "output_tokens": 1})


class ClaudeTests(unittest.TestCase):
    def setUp(self):
        dino = patch('weapons.predict_weapons', return_value=[])
        self.dino = dino.start()
        self.addCleanup(dino.stop)

    @patch('claude.describe_object', return_value='A silver car.')
    @patch('steps.load_yolo')
    @patch('claude.Anthropic')
    def test_summary_receives_only_statistics_and_descriptions_are_displayed(self, model, load, describe):
        model.return_value.__enter__.return_value.messages.create.return_value = response_text(content='Estimated occupancy is 84%.')
        load.return_value.predict.return_value = [detector_result([0] * 37 + [1] * 4 + [2])]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'private-parking.png'
            Image.new('RGB', (10, 10)).save(path)
            result = run(initial_state('How crowded?', path, 50))
            self.assertIn('Estimated occupancy is 84%.', result['answer'])
            self.assertIn('Car 37: A silver car.', result['answer'])
            self.assertEqual(describe.call_count, 37)
        self.assertEqual(model.return_value.__enter__.return_value.messages.create.call_count, 1)
        for call in model.return_value.__enter__.return_value.messages.create.call_args_list:
            messages = call.kwargs['messages']
            payload = json.loads(messages[0]['content'])
            self.assertEqual(payload, {'question': 'How crowded?', 'statistics': {
                'weapon_candidates': [],
                'counts': {'car': 37, 'truck': 4, 'bus': 1, 'motorcycle': 0, 'person': 0},
                'total': 42, 'parking_capacity': 50, 'occupancy': 84.0,
                'descriptions': [{'label': 'car', 'object_id': i, 'description': 'A silver car.'}
                                     for i in range(1, 38)]}})
            self.assertNotIn('private-parking', str(messages))
        request = model.return_value.__enter__.return_value.messages.create.call_args.kwargs
        self.assertEqual(request['temperature'], 0)
        self.assertEqual(request['max_tokens'], 1200)
        self.assertEqual(request['messages'][0]['role'], 'user')
        self.assertIn('parking-lot question', request['system'])
        model.assert_called_once_with(timeout=60, max_retries=2)

    @patch('claude.Anthropic')
    def test_empty_response(self, model):
        model.return_value.__enter__.return_value.messages.create.return_value = response_text(content='')
        with self.assertRaisesRegex(RuntimeError, 'no text'):
            generate_answer('question', {}, 'test-model')

    @patch('claude.Anthropic')
    def test_text_blocks_are_joined_and_other_blocks_ignored(self, model):
        response = response_text('  First ')
        response.content.extend([
            ThinkingBlock(type='thinking', thinking='internal reasoning', signature='test'),
            TextBlock(type='text', text='second.  '),
        ])
        model.return_value.__enter__.return_value.messages.create.return_value = response
        self.assertEqual(generate_answer('question', {}, 'test-model'), 'First second.')
        model.return_value.__exit__.assert_called_once()

    @patch('claude.Anthropic')
    def test_api_errors_are_readable_and_client_is_closed(self, model):
        request = httpx.Request('POST', 'https://api.anthropic.com/v1/messages')
        errors = [
            (AuthenticationError('bad key', response=httpx.Response(401, request=request), body=None), 'rejected the API key'),
            (RateLimitError('rate limit', response=httpx.Response(429, request=request), body=None), 'rate limit reached'),
            (APIConnectionError(request=request), 'API request failed'),
        ]
        for error, message in errors:
            with self.subTest(error=type(error).__name__):
                model.reset_mock()
                model.return_value.__enter__.return_value.messages.create.side_effect = error
                with self.assertRaisesRegex(RuntimeError, message):
                    generate_answer('question', {}, 'test-model')
                model.return_value.__exit__.assert_called_once()


if __name__ == '__main__':
    unittest.main()
