import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

from graph import graph, run
from steps import initial_state, statistics
from test_workflows import detector_result
from weapons import detect_weapons, predict_weapons, load_grounding_dino, WEAPON_PROMPT


class WeaponTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / 'private.png'
        image = Image.new('RGB', (40, 20), 'red')
        exif = Image.Exif()
        exif[274] = 6
        image.save(self.path, exif=exif)

    def state(self, **kwargs):
        return initial_state('Describe this image.', self.path, use_llm=False, **kwargs)

    @patch('weapons.predict_weapons')
    def test_full_oriented_image_threshold_clipping_and_duplicate_suppression(self, predict):
        predict.return_value = [
            {'label': 'handgun', 'confidence': .9, 'xyxy': [-1, 1, 11, 15]},
            {'label': 'rifle', 'confidence': .8, 'xyxy': [0, 1, 11, 15]},
            {'label': 'knife', 'confidence': .7, 'xyxy': [12, 20, 24, 45]},
            {'label': 'knife', 'confidence': .1, 'xyxy': [12, 1, 18, 5]},
        ]
        state = self.state()
        result = detect_weapons(state)
        self.assertEqual(predict.call_args.args[0].size, (20, 40))
        self.assertTrue(predict.call_args.kwargs['local_files_only'])
        self.assertEqual(result['weapon_detections'], [
            {'label': 'handgun', 'confidence': .9, 'xyxy': [0, 1, 11, 15]},
            {'label': 'knife', 'confidence': .7, 'xyxy': [12, 20, 20, 40]},
        ])
        self.assertEqual(state['weapon_detections'], [])
        self.assertEqual(state['trace'], [])

    @patch('weapons.predict_weapons')
    @patch('steps.load_yolo')
    def test_graph_detects_weapons_without_people_and_keeps_occupancy_separate(self, yolo, predict):
        yolo.return_value.predict.return_value = [detector_result([])]
        predict.return_value = [{'label': 'knife', 'confidence': .8, 'xyxy': [1, 1, 10, 10]}]
        state = run(self.state(parking_capacity=10))
        self.assertEqual(state['total'], 0)
        self.assertEqual(state['occupancy'], 0)
        self.assertIn('Candidate 1: knife', state['answer'])
        evidence = statistics(state)
        self.assertEqual(evidence['weapon_candidates'], [{'label': 'knife', 'confidence': .8}])
        self.assertNotIn('xyxy', str(evidence))
        self.assertNotIn('private', str(evidence))
        edges = {(edge.source, edge.target) for edge in graph.get_graph().edges}
        self.assertIn(('detect', 'detect_weapons'), edges)
        self.assertIn(('detect_weapons', 'describe_detections'), edges)

    @patch('weapons.predict_weapons', side_effect=OSError('missing model'))
    @patch('claude.describe_object')
    @patch('steps.load_yolo')
    def test_dino_failure_stops_graph_before_claude(self, yolo, describe, predict):
        yolo.return_value.predict.return_value = [detector_result([4])]
        state = self.state()
        state['use_llm'] = True
        with self.assertRaisesRegex(RuntimeError, 'Grounding DINO weapon detection failed'):
            run(state)
        describe.assert_not_called()
        self.assertFalse(predict.call_args.kwargs['local_files_only'])

    @patch('weapons.predict_weapons')
    def test_malformed_candidates_fail(self, predict):
        for box, score in [([0, 0, float('nan'), 2], .8), ([2, 2, 1, 1], .8), ([0, 0, 2, 2], float('nan'))]:
            with self.subTest(box=box, score=score):
                predict.return_value = [{'label': 'knife', 'confidence': score, 'xyxy': box}]
                with self.assertRaisesRegex(RuntimeError, 'Invalid weapon detection'):
                    detect_weapons(self.state())

    def test_invalid_thresholds(self):
        for key in ('weapon_threshold', 'weapon_text_threshold'):
            for value in (0, -1, 1.1, float('nan'), True):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.state(**{key: value})

    @patch('weapons.load_grounding_dino')
    def test_transformers_boundary_uses_prompt_thresholds_and_image_size(self, load):
        import torch

        processor, model = MagicMock(), MagicMock()
        load.return_value = processor, model
        model.device = 'cpu'
        inputs = MagicMock()
        inputs.keys.return_value = ['input_ids']
        inputs.__getitem__.return_value = torch.tensor([[1]])
        inputs.input_ids = torch.tensor([[1]])
        processor.return_value.to.return_value = inputs
        processor.post_process_grounded_object_detection.return_value = [{
            'text_labels': ['handgun'], 'scores': torch.tensor([.8]),
            'boxes': torch.tensor([[1., 2., 8., 9.]]),
        }]
        image = Image.new('RGB', (30, 20))
        result = predict_weapons(image, 'local-model', .4, .3, local_files_only=True)
        from devices import resolve_device
        load.assert_called_once_with('local-model', True, resolve_device())
        self.assertEqual(processor.call_args.kwargs['text'], WEAPON_PROMPT)
        kwargs = processor.post_process_grounded_object_detection.call_args.kwargs
        self.assertEqual(kwargs, {'threshold': .4, 'text_threshold': .3, 'target_sizes': [(20, 30)]})
        self.assertEqual(result[0]['label'], 'handgun')
        self.assertEqual(result[0]['xyxy'], [1., 2., 8., 9.])

    @patch('weapons.load_grounding_dino')
    def test_empty_boxes_with_phantom_label_returns_no_candidates(self, load):
        import torch

        processor, model = MagicMock(), MagicMock()
        load.return_value = processor, model
        model.device = 'cpu'
        processor.post_process_grounded_object_detection.return_value = [{
            'text_labels': [''], 'scores': torch.empty(0), 'boxes': torch.empty((0, 4)),
        }]
        self.assertEqual(predict_weapons(Image.new('RGB', (30, 20)), 'local', .35, .25,
                                         local_files_only=True, device='cpu'), [])

    @patch('weapons.load_grounding_dino')
    def test_nonempty_mismatched_results_still_fail(self, load):
        import torch

        processor, model = MagicMock(), MagicMock()
        load.return_value = processor, model
        model.device = 'cpu'
        processor.post_process_grounded_object_detection.return_value = [{
            'text_labels': ['knife', 'rifle'], 'scores': torch.tensor([.8]),
            'boxes': torch.tensor([[1., 2., 8., 9.]]),
        }]
        with self.assertRaises(ValueError):
            predict_weapons(Image.new('RGB', (30, 20)), 'local', .35, .25,
                            local_files_only=True, device='cpu')

    @patch('torch.cuda.is_available', return_value=False)
    @patch('transformers.AutoModelForZeroShotObjectDetection.from_pretrained')
    @patch('transformers.AutoProcessor.from_pretrained')
    def test_offline_loader_forbids_downloads_and_uses_cpu_eval(self, processor, model, cuda):
        load_grounding_dino.cache_clear()
        self.addCleanup(load_grounding_dino.cache_clear)
        loaded_processor, loaded_model = load_grounding_dino('test-local-model', True, 'cpu')
        processor.assert_called_once_with('test-local-model', local_files_only=True)
        model.assert_called_once_with('test-local-model', local_files_only=True)
        model.return_value.to.assert_called_once_with('cpu')
        model.return_value.to.return_value.eval.assert_called_once()
        self.assertIs(loaded_processor, processor.return_value)
        self.assertIs(loaded_model, model.return_value.to.return_value.eval.return_value)
        load_grounding_dino('test-local-model', True, 'cpu')
        self.assertEqual(model.call_count, 1)
        load_grounding_dino('test-local-model', True, 'mps')
        self.assertEqual(model.call_count, 2)
        model.return_value.to.assert_called_with('mps')


if __name__ == '__main__':
    unittest.main()
