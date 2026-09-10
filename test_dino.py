import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

from steps import initial_state
from dino import query_objects, predict_objects, load_grounding_dino


class DinoTests(unittest.TestCase):
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

    @patch('dino.predict_objects')
    def test_full_oriented_image_threshold_clipping_and_duplicate_suppression(self, predict):
        predict.return_value = [
            {'label': 'handgun', 'confidence': .9, 'xyxy': [-1, 1, 11, 15]},
            {'label': 'handgun', 'confidence': .8, 'xyxy': [0, 1, 11, 15]},
            {'label': 'knife', 'confidence': .7, 'xyxy': [12, 20, 24, 45]},
            {'label': 'knife', 'confidence': .1, 'xyxy': [12, 1, 18, 5]},
        ]
        state = self.state()
        result = query_objects(state, 'bicycle')
        self.assertEqual(predict.call_args.args[0].size, (20, 40))
        self.assertTrue(predict.call_args.kwargs['local_files_only'])
        self.assertEqual(predict.call_args.kwargs['query'], 'bicycle.')
        self.assertEqual(result['count'], 2)
        self.assertEqual(result['detections'], [
            {'label': 'handgun', 'confidence': .9, 'xyxy': [0, 1, 11, 15]},
            {'label': 'knife', 'confidence': .7, 'xyxy': [12, 20, 20, 40]},
        ])
        self.assertEqual(state['trace'], [])

    @patch('dino.predict_objects')
    def test_malformed_candidates_fail(self, predict):
        for box, score in [([0, 0, float('nan'), 2], .8), ([2, 2, 1, 1], .8), ([0, 0, 2, 2], float('nan'))]:
            with self.subTest(box=box, score=score):
                predict.return_value = [{'label': 'knife', 'confidence': score, 'xyxy': box}]
                with self.assertRaisesRegex(RuntimeError, 'Invalid object query'):
                    query_objects(self.state(), 'bicycle')

    def test_invalid_thresholds(self):
        for key in ('dino_threshold', 'dino_text_threshold'):
            for value in (0, -1, 1.1, float('nan'), True):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.state(**{key: value})

    @patch('dino.predict_objects')
    def test_invalid_queries_do_not_run_model(self, predict):
        for query in ('', '  ', '...', 'x' * 201, None, 123):
            with self.subTest(query=query), self.assertRaises(ValueError):
                query_objects(self.state(), query)
        predict.assert_not_called()

    @patch('dino.predict_objects')
    def test_overlapping_different_classes_are_kept(self, predict):
        predict.return_value = [
            {'label': 'chair', 'confidence': .9, 'xyxy': [1, 1, 10, 10]},
            {'label': 'backpack', 'confidence': .8, 'xyxy': [1, 1, 10, 10]},
        ]
        self.assertEqual(query_objects(self.state(), 'chair. backpack')['count'], 2)

    @patch('dino.load_grounding_dino')
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
        result = predict_objects(image, 'local-model', .4, .3, local_files_only=True, query='bicycle.')
        from devices import resolve_device
        load.assert_called_once_with('local-model', True, resolve_device())
        self.assertEqual(processor.call_args.kwargs['text'], 'bicycle.')
        kwargs = processor.post_process_grounded_object_detection.call_args.kwargs
        self.assertEqual(kwargs, {'threshold': .4, 'text_threshold': .3, 'target_sizes': [(20, 30)]})
        self.assertEqual(result[0]['label'], 'handgun')
        self.assertEqual(result[0]['xyxy'], [1., 2., 8., 9.])

    @patch('dino.load_grounding_dino')
    def test_empty_boxes_with_phantom_label_returns_no_candidates(self, load):
        import torch

        processor, model = MagicMock(), MagicMock()
        load.return_value = processor, model
        model.device = 'cpu'
        processor.post_process_grounded_object_detection.return_value = [{
            'text_labels': [''], 'scores': torch.empty(0), 'boxes': torch.empty((0, 4)),
        }]
        self.assertEqual(predict_objects(Image.new('RGB', (30, 20)), 'local', .35, .25,
                                         local_files_only=True, device='cpu', query='bicycle.'), [])

    @patch('dino.load_grounding_dino')
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
            predict_objects(Image.new('RGB', (30, 20)), 'local', .35, .25,
                            local_files_only=True, device='cpu', query='bicycle.')

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
