"""Exercise the public CLI, configuration loading, and single graph dispatch."""

import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import demo


class CLITests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.image = Path(self.directory.name) / 'parking.png'
        Image.new('RGB', (10, 10)).save(self.image)

    @patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key', 'ANTHROPIC_MODEL': 'test-model'}, clear=True)
    @patch('demo.load_dotenv')
    @patch('builtins.input', return_value='quit')
    @patch('demo.ImageConversation')
    @patch('demo.run')
    def test_default_invokes_graph_once_with_unknown_capacity(self, run, conversation, user_input, load_env):
        run.return_value = {'trace': ['Detection completed'], 'answer': 'Occupancy unknown.'}
        output = io.StringIO()
        with patch('sys.argv', ['demo.py', '--image', str(self.image)]), redirect_stdout(output):
            demo.main()
        run.assert_called_once()
        state = run.call_args.args[0]
        self.assertIsNone(state['parking_capacity'])
        self.assertTrue(state['use_llm'])
        self.assertEqual(state['model'], 'test-model')
        self.assertNotIn('test-key', str(state))
        self.assertIn('Occupancy unknown.', output.getvalue())
        load_env.assert_called_once_with(Path(demo.__file__).parent / '.env', override=False)

    @patch.dict(os.environ, {}, clear=True)
    @patch('demo.load_dotenv')
    @patch('builtins.input', return_value='quit')
    @patch('demo.ImageConversation')
    @patch('demo.run')
    def test_offline_runs_graph_without_api_key(self, run, conversation, user_input, load_env):
        weights = Path(self.directory.name) / 'detector.pt'
        weights.write_bytes(b'test placeholder; graph is mocked')
        run.return_value = {'trace': [], 'answer': 'Local statistics.'}
        args = ['demo.py', '--image', str(self.image), '--offline', '--yolo-model', str(weights)]
        with patch('sys.argv', args), redirect_stdout(io.StringIO()):
            demo.main()
        run.assert_called_once()
        self.assertFalse(run.call_args.args[0]['use_llm'])

    @patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'}, clear=True)
    @patch('demo.load_dotenv')
    @patch('builtins.input', return_value='quit')
    @patch('demo.ImageConversation')
    @patch('demo.run')
    def test_dino_options_reach_graph_state(self, run, conversation, user_input, load_env):
        run.return_value = {'trace': [], 'answer': 'Candidates.'}
        args = ['demo.py', '--image', str(self.image), '--dino-model', 'local-dino',
                '--dino-threshold', '0.5', '--dino-text-threshold', '0.3']
        with patch('sys.argv', args), redirect_stdout(io.StringIO()):
            demo.main()
        state = run.call_args.args[0]
        self.assertEqual(state['dino_model'], 'local-dino')
        self.assertEqual(state['dino_threshold'], 0.5)
        self.assertEqual(state['dino_text_threshold'], 0.3)

    @patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'}, clear=True)
    @patch('demo.load_dotenv')
    @patch('builtins.input', side_effect=['Any bicycles?', 'Any backpacks?', 'quit'])
    @patch('demo.ImageConversation')
    @patch('demo.run')
    def test_interactive_followups_reuse_counting_and_conversation(self, run, conversation, user_input, load_env):
        run.return_value = {'answer': 'car: 2, person: 4'}
        conversation.return_value.ask.return_value = 'One candidate.'
        with patch('sys.argv', ['demo.py', '--image', str(self.image)]), redirect_stdout(io.StringIO()):
            demo.main()
        run.assert_called_once()
        conversation.assert_called_once_with(run.return_value)
        self.assertEqual([call.args[0] for call in conversation.return_value.ask.call_args_list],
                         ['Any bicycles?', 'Any backpacks?'])

    @patch.dict(os.environ, {}, clear=True)
    @patch('demo.load_dotenv')
    @patch('demo.ImageConversation')
    @patch('demo.run')
    def test_missing_key_still_reports_counts(self, run, conversation, load_env):
        run.return_value = {'answer': 'car: 2, person: 4'}
        output = io.StringIO()
        with patch('sys.argv', ['demo.py', '--image', str(self.image)]), redirect_stdout(output):
            demo.main()
        self.assertIn('car: 2, person: 4', output.getvalue())
        conversation.assert_not_called()


if __name__ == '__main__':
    unittest.main()
