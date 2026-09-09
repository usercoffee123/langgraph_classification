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
    @patch('claude.describe_object', return_value='A silver car.')
    @patch('steps.load_yolo')
    @patch('claude.ChatAnthropic')
    def test_summary_receives_only_statistics_and_descriptions_are_displayed(self, model, load, describe):
        model.return_value.invoke.return_value = AIMessage(content='Estimated occupancy is 84%.')
        load.return_value.predict.return_value = [detector_result([0] * 37 + [1] * 4 + [2])]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'private-parking.png'
            Image.new('RGB', (10, 10)).save(path)
            result = run(initial_state('How crowded?', path, 50))
            self.assertIn('Estimated occupancy is 84%.', result['answer'])
            self.assertIn('Car 37: A silver car.', result['answer'])
            self.assertEqual(describe.call_count, 37)
        self.assertEqual(model.return_value.invoke.call_count, 1)
        for call in model.return_value.invoke.call_args_list:
            messages = call.args[0]
            payload = json.loads(messages[1][1])
            self.assertEqual(payload, {'question': 'How crowded?', 'statistics': {
                'counts': {'car': 37, 'truck': 4, 'bus': 1, 'motorcycle': 0, 'person': 0},
                'total': 42, 'parking_capacity': 50, 'occupancy': 84.0,
                'descriptions': [{'label': 'car', 'object_id': i, 'description': 'A silver car.'}
                                     for i in range(1, 38)]}})
            self.assertNotIn('private-parking', str(messages))
        self.assertEqual(model.call_args.kwargs['temperature'], 0)

    @patch('claude.ChatAnthropic')
    def test_empty_response(self, model):
        model.return_value.invoke.return_value = AIMessage(content='')
        with self.assertRaisesRegex(RuntimeError, 'no text'):
            generate_answer('question', {}, 'test-model')


if __name__ == '__main__':
    unittest.main()
