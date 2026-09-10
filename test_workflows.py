import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

from graph import run
from steps import calculate_occupancy, initial_state


def detector_result(labels):
    return SimpleNamespace(names={0: 'car', 1: 'truck', 2: 'bus', 3: 'motorcycle', 4: 'person'},
                           boxes=SimpleNamespace(cls=Mock(tolist=lambda: labels),
                                                 conf=Mock(tolist=lambda: [0.94] * len(labels)),
                                                 xyxy=Mock(tolist=lambda: [[1, 2, 3, 4]] * len(labels))))


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        dino = patch('dino.predict_objects', return_value=[])
        self.dino = dino.start()
        self.addCleanup(dino.stop)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.image = Path(self.directory.name) / 'parking.png'
        Image.new('RGB', (10, 10)).save(self.image)

    def state(self, **kwargs):
        return initial_state('How crowded is this parking lot?', self.image, 50, use_llm=False, **kwargs)

    @patch('steps.load_yolo')
    def test_graph_counts_vehicles_and_calculates_84_percent(self, load):
        load.return_value.predict.return_value = [detector_result([0] * 37 + [1] * 4 + [2, 4])]
        state = self.state()
        graph = run(state)
        self.assertEqual(graph['counts'], {'car': 37, 'truck': 4, 'bus': 1, 'motorcycle': 0, 'person': 1})
        self.assertEqual(graph['total'], 42)
        self.assertEqual(graph['occupancy'], 84)
        self.assertEqual(len(graph['detections']), 43)
        self.assertEqual(graph['detections'][0], {'label': 'car', 'confidence': 0.94, 'xyxy': [1, 2, 3, 4]})
        self.assertEqual(len(graph['trace']), 3)
        self.dino.assert_not_called()
        self.assertEqual(state['counts'], {})
        self.assertEqual(state['trace'], [])
        self.assertIsInstance(load.return_value.predict.call_args.kwargs['source'], Image.Image)

    @patch('steps.load_yolo')
    @patch('claude.Anthropic')
    def test_zero_detections_offline(self, claude, load):
        load.return_value.predict.return_value = [detector_result([])]
        result = run(self.state())
        self.assertEqual(result['total'], 0)
        self.assertEqual(result['occupancy'], 0)
        claude.assert_not_called()

    @patch('steps.load_yolo')
    def test_confidence_filter_and_motorcycles(self, load):
        result = detector_result([0, 3])
        result.boxes.conf = Mock(tolist=lambda: [0.1, 0.9])
        load.return_value.predict.return_value = [result]
        result = run(self.state(confidence=0.5))
        self.assertEqual(result['counts']['car'], 0)
        self.assertEqual(result['counts']['motorcycle'], 1)
        self.assertEqual(load.return_value.predict.call_args.kwargs['conf'], 0.5)

    @patch('steps.load_yolo')
    @patch('claude.Anthropic')
    def test_people_only_offline_have_zero_vehicle_occupancy(self, claude, load):
        load.return_value.predict.return_value = [detector_result([4, 4])]
        result = run(self.state())
        self.assertEqual(result['counts']['person'], 2)
        self.assertEqual(result['total'], 0)
        self.assertEqual(result['occupancy'], 0)
        self.assertIn('person: 2', result['answer'])
        claude.assert_not_called()

    def test_over_capacity_is_not_clamped(self):
        state = self.state()
        state['counts'] = {'car': 60}
        self.assertEqual(calculate_occupancy(state)['occupancy'], 120)

    @patch('steps.load_yolo')
    def test_unknown_capacity_does_not_produce_occupancy(self, load):
        load.return_value.predict.return_value = [detector_result([0] * 6)]
        state = initial_state('How crowded?', self.image, use_llm=False)
        result = run(state)
        self.assertEqual(result['total'], 6)
        self.assertIsNone(result['occupancy'])
        self.assertIn('Occupancy unknown', result['answer'])
        self.assertNotIn('%', result['answer'])

    def test_bad_inputs(self):
        for capacity in (0, -1, 1.5, True):
            with self.subTest(capacity=capacity), self.assertRaises(ValueError):
                initial_state('count', self.image, capacity)
        for confidence in (0, -0.1, 1.1, float('nan')):
            with self.subTest(confidence=confidence), self.assertRaises(ValueError):
                self.state(confidence=confidence)
        with self.assertRaises(ValueError):
            initial_state(' ', self.image, 50)
        with self.assertRaises(ValueError):
            initial_state('count', self.image.parent / 'missing.png', 50)

    @patch('steps.load_yolo')
    @patch('claude.Anthropic')
    def test_corrupt_image_stops_before_detector_or_claude(self, claude, load):
        self.image.write_bytes(b'not an image')
        state = self.state()
        state['use_llm'] = True
        with self.assertRaisesRegex(RuntimeError, 'Local YOLO detection failed'):
            run(state)
        load.assert_not_called()
        claude.assert_not_called()

    @patch('steps.load_yolo')
    @patch('claude.Anthropic')
    def test_inference_failure_does_not_call_claude(self, claude, load):
        load.return_value.predict.side_effect = OSError('bad weights')
        state = self.state()
        state['use_llm'] = True
        with self.assertRaisesRegex(RuntimeError, 'bad weights'):
            run(state)
        claude.assert_not_called()


if __name__ == '__main__':
    unittest.main()
