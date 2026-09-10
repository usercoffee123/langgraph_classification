import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from anthropic.types import TextBlock, ToolUseBlock
from types import SimpleNamespace
from PIL import Image

from claude import ImageConversation
from steps import initial_state


def response(*blocks):
    return SimpleNamespace(content=list(blocks), stop_reason='tool_use' if any(b.type == 'tool_use' for b in blocks) else 'end_turn')


def text(value):
    return TextBlock(type='text', text=value)


def tool(query, id='call1', name='query_objects'):
    return ToolUseBlock(type='tool_use', id=id, name=name, input={'query': query})


class ConversationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / 'private.png'
        Image.new('RGB', (20, 20)).save(path)
        self.state = initial_state('count', path, device='cpu')
        self.state['counts'] = {'car': 2, 'person': 4}
        self.chat = ImageConversation(self.state)
        p = patch('claude.Anthropic')
        self.create = p.start().return_value.__enter__.return_value.messages.create
        self.addCleanup(p.stop)

    @patch('claude.query_objects')
    def test_counts_need_no_dino_and_followups_preserve_history(self, query):
        self.create.side_effect = [response(text('2 cars and 4 people.')), response(text('4 people.'))]
        self.chat.ask('How many cars and people?')
        self.chat.ask('How many of those are people?')
        query.assert_not_called()
        self.assertEqual(len(self.chat.messages), 4)
        self.assertNotIn('private.png', self.chat.system)
        self.assertIn('"car": 2', self.chat.system)

    @patch('claude.query_objects')
    @patch('steps.load_yolo')
    def test_graph_results_reach_llm_on_each_turn_without_rerunning_yolo(self, yolo, query):
        from graph import run
        from test_workflows import detector_result

        detected = detector_result([0, 4, 24])
        detected.names[24] = 'backpack'
        yolo.return_value.predict.return_value = [detected]
        state = run(self.state)
        chat = ImageConversation(state)
        self.create.side_effect = [response(text('One car.')), response(text('Its box is [1, 2, 3, 4].'))]
        chat.ask('How many cars?')
        chat.ask('Where is it?')
        for call in self.create.call_args_list:
            system = call.kwargs['system']
            evidence = json.loads(system.split('\nSaved YOLO results: ', 1)[1])
            self.assertEqual(evidence['detections'], state['detections'])
            self.assertEqual(evidence['detections'][0]['confidence'], .94)
            self.assertEqual(evidence['image_size'], {'width': 20, 'height': 20})
            self.assertEqual(evidence['counts']['car'], 1)
            self.assertEqual(evidence['counts']['backpack'], 1)
            self.assertEqual(evidence['detections'][-1]['label'], 'backpack')
            self.assertNotIn(state['image_path'], system)
            self.assertNotIn('image_path', evidence)
            self.assertNotIn('model', evidence)
        yolo.return_value.predict.assert_called_once()
        query.assert_not_called()

    @patch('claude.query_objects')
    def test_tool_round_trip_and_multiple_calls(self, query):
        query.return_value = {'count': 1, 'detections': [{'label': 'bicycle', 'confidence': .8, 'xyxy': [1, 2, 3, 4]}]}
        self.create.side_effect = [response(tool('bicycle'), tool('backpack', 'call2')), response(text('One candidate for each.'))]
        self.assertEqual(self.chat.ask('Find bicycles and backpacks.'), 'One candidate for each.')
        self.assertEqual(query.call_args_list[0].args, (self.state, 'bicycle'))
        results = self.chat.messages[2]['content']
        self.assertEqual([r['tool_use_id'] for r in results], ['call1', 'call2'])
        self.assertEqual(json.loads(results[0]['content'])['count'], 1)
        self.assertEqual(self.create.call_args.kwargs['tools'][0]['name'], 'query_objects')
        # Bind against the installed SDK signature, not just an unrestricted mock.
        import inspect
        from anthropic.resources.messages import Messages
        inspect.signature(Messages.create).bind(None, **self.create.call_args.kwargs)

    @patch('claude.query_objects', side_effect=RuntimeError('/private/path/model failed'))
    def test_tool_failure_is_error_not_empty_detections(self, query):
        self.create.side_effect = [response(tool('bicycle')), response(text('The search failed.'))]
        self.chat.ask('Any bicycles?')
        result = self.chat.messages[2]['content'][0]
        self.assertTrue(result['is_error'])
        self.assertNotIn('/private/path', result['content'])

    @patch('claude.query_objects')
    def test_unknown_tool_not_executed(self, query):
        self.create.side_effect = [response(tool('bicycle', name='read_file')), response(text('Cannot use that tool.'))]
        self.chat.ask('Find bicycles.')
        query.assert_not_called()
        self.assertTrue(self.chat.messages[2]['content'][0]['is_error'])

    @patch('claude.query_objects', return_value={'count': 0, 'detections': []})
    def test_repeated_tool_calls_are_bounded_and_failed_turn_not_saved(self, query):
        self.create.return_value = response(tool('bicycle'))
        with self.assertRaisesRegex(RuntimeError, 'round limit'):
            self.chat.ask('Find bicycles.')
        self.assertEqual(query.call_count, 5)
        self.assertEqual(self.chat.messages, [])

    def test_offline_never_calls_claude(self):
        self.state['use_llm'] = False
        with self.assertRaisesRegex(ValueError, 'require Claude'):
            self.chat.ask('Any bicycles?')
        self.create.assert_not_called()
