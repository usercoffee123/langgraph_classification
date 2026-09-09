import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from langchain_core.messages import AIMessage

from graph import run
from claude import generate_answer
from steps import initial_state
from test_workflows import detector_result


class ClaudeTests(unittest.TestCase):
    @patch('steps.load_yolo')
    @patch('claude.ChatAnthropic')
    def test_only_question_and_aggregate_statistics_reach_claude(self, model, load):
        model.return_value.invoke.return_value = AIMessage(content='Estimated occupancy is 84%.')
        load.return_value.predict.return_value = [detector_result([0] * 37 + [1] * 4 + [2])]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'private-parking.png'
            Image.new('RGB', (10, 10)).save(path)
            result = run(initial_state('How crowded?', path, 50))
            self.assertEqual(result['answer'], 'Estimated occupancy is 84%.')
        self.assertEqual(model.return_value.invoke.call_count, 1)
        for call in model.return_value.invoke.call_args_list:
            messages = call.args[0]
            payload = json.loads(messages[1][1])
            self.assertEqual(payload, {'question': 'How crowded?', 'statistics': {
                'counts': {'car': 37, 'truck': 4, 'bus': 1, 'motorcycle': 0},
                'total': 42, 'parking_capacity': 50, 'occupancy': 84.0}})
            self.assertNotIn('private-parking', str(messages))
        self.assertEqual(model.call_args.kwargs['temperature'], 0)

    @patch('claude.ChatAnthropic')
    def test_empty_response(self, model):
        model.return_value.invoke.return_value = AIMessage(content='')
        with self.assertRaisesRegex(RuntimeError, 'no text'):
            generate_answer('question', {}, 'test-model')


if __name__ == '__main__':
    unittest.main()
