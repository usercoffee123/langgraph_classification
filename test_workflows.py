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
        self.assertEqual(graph['image_size'], {'width': 10, 'height': 10})
        self.assertEqual(len(graph['trace']), 3)
        self.dino.assert_not_called()
        self.assertEqual(state['counts'], {})
        self.assertEqual(state['trace'], [])
        self.assertIsInstance(load.return_value.predict.call_args.kwargs['source'], Image.Image)

    @patch('steps.load_yolo')
    def test_state_image_dimensions_match_oriented_detector_input(self, load):
        image = Image.new('RGB', (40, 20))
        exif = Image.Exif()
        exif[274] = 6
        image.save(self.image, exif=exif)
        load.return_value.predict.return_value = [detector_result([])]
        result = run(self.state())
        self.assertEqual(result['image_size'], {'width': 20, 'height': 40})
        self.assertEqual(load.return_value.predict.call_args.kwargs['source'].size, (20, 40))
        self.assertEqual(result['detections'], [])

    @patch('steps.load_yolo')
    def test_all_model_classes_retained_reported_and_available_to_llm(self, load):
        from steps import statistics

        detected = detector_result([2, 24, 24, 39, 67, 99, 1])
        detected.names = {2: 'car', 24: 'backpack', 39: 'bottle', 67: 'cell phone',
                          99: 'custom crate', 1: 'bicycle', 4: 'person'}
        detected.boxes.conf = Mock(tolist=lambda: [.94] * 6 + [.1])
        load.return_value.predict.return_value = [detected]
        result = run(self.state())
        expected = {'car': 1, 'backpack': 2, 'bottle': 1, 'cell phone': 1,
                    'custom crate': 1, 'bicycle': 0, 'person': 0}
        self.assertEqual(result['counts'], expected)
        self.assertEqual(len(result['detections']), 6)
        for name in ('backpack', 'bottle', 'cell phone', 'custom crate'):
            self.assertIn(name, [item['label'] for item in result['detections']])
            self.assertIn(f'{name}: {expected[name]}', result['answer'])
        self.assertIn('6 object(s) across 5 detected class(es)', result['answer'])
        self.assertNotIn('bicycle:', result['answer'])
        self.assertEqual(statistics(result)['counts'], expected)
        self.assertEqual(statistics(result)['detections'], result['detections'])
        self.assertEqual(result['total'], 1)
        self.assertEqual(result['occupancy'], 2)

    @patch('steps.load_yolo')
    @patch('claude.Anthropic')
    def test_zero_detections_offline(self, claude, load):
        load.return_value.predict.return_value = [detector_result([])]
        result = run(self.state())
        self.assertEqual(result['total'], 0)
        self.assertEqual(result['occupancy'], 0)
        self.assertEqual(result['counts'], dict.fromkeys(detector_result([]).names.values(), 0))
        self.assertIn('No objects detected above the configured confidence threshold.', result['answer'])
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
