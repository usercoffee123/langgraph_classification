import base64
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from langchain_core.messages import AIMessage

from graph import run
from steps import describe_cars, initial_state
from test_workflows import detector_result


class CarDescriptionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'private-image.png'
        self.image = Image.new('RGB', (40, 20), 'red')
        self.image.paste('blue', (20, 0, 40, 20))
        self.image.save(self.path)

    def state(self, box=None, **kwargs):
        state = initial_state('Describe the cars.', self.path, **kwargs)
        state['detections'] = [{'label': 'car', 'confidence': 0.9, 'xyxy': box or [0, 0, 20, 20]}]
        return state

    @patch('steps.load_yolo')
    @patch('claude.ChatAnthropic')
    def test_graph_sends_separate_clamped_car_crops_and_displays_each_description(self, model, load):
        detection = detector_result([0, 1, 0])
        detection.boxes.xyxy.tolist = lambda: [[-3, 0, 20, 20], [0, 0, 40, 20], [20, 0, 45, 20]]
        load.return_value.predict.return_value = [detection]
        model.return_value.invoke.side_effect = [AIMessage(content='A red car.'),
                                                 AIMessage(content='A blue car.'),
                                                 AIMessage(content='Three vehicles detected.')]
        state = initial_state('Describe the cars.', self.path)
        result = run(state)
        self.assertEqual(model.return_value.invoke.call_count, 3)
        self.assertEqual([car['detection_index'] for car in result['car_descriptions']], [0, 2])
        self.assertEqual([car['xyxy'] for car in result['car_descriptions']],
                         [[0, 0, 20, 20], [20, 0, 40, 20]])
        self.assertIn('Car 1: A red car.', result['answer'])
        self.assertIn('Car 2: A blue car.', result['answer'])
        self.assertEqual(state['car_descriptions'], [])
        for index, call in enumerate(model.return_value.invoke.call_args_list[:2]):
            messages = call.args[0]
            source = messages[1][1][0]['source']
            self.assertEqual(source['media_type'], 'image/jpeg')
            with Image.open(BytesIO(base64.b64decode(source['data']))) as crop:
                self.assertEqual(crop.size, (20, 20))
                pixel = crop.getpixel((10, 10))
                self.assertGreater(pixel[0 if index == 0 else 2], 240)
                self.assertLess(pixel[2 if index == 0 else 0], 15)
            self.assertNotIn('private-image', str(messages))

    @patch('claude.describe_car', return_value='A car.')
    def test_crop_uses_exif_orientation_matching_detector(self, describe):
        exif = Image.Exif()
        exif[274] = 6  # Rotate clockwise: red half is now above blue half.
        self.image.save(self.path, exif=exif)
        describe_cars(self.state([0, 0, 20, 20]))
        with Image.open(BytesIO(describe.call_args.args[0])) as crop:
            self.assertGreater(crop.getpixel((10, 15))[0], 240)
            self.assertFalse(crop.getexif())

    @patch('claude.describe_car', return_value='A car.')
    def test_large_crop_is_bounded(self, describe):
        Image.new('RGB', (2000, 1000), 'red').save(self.path)
        describe_cars(self.state([0, 0, 2000, 1000]))
        with Image.open(BytesIO(describe.call_args.args[0])) as crop:
            self.assertEqual(crop.size, (768, 384))

    @patch('claude.describe_car')
    def test_offline_and_no_car_skip_api(self, describe):
        self.assertEqual(describe_cars(self.state(use_llm=False))['car_descriptions'], [])
        state = self.state()
        state['detections'][0]['label'] = 'truck'
        self.assertEqual(describe_cars(state)['car_descriptions'], [])
        describe.assert_not_called()

    @patch('claude.describe_car')
    def test_invalid_boxes_fail_before_api(self, describe):
        for box in ([5, 5, 1, 1], [0, 0, 0, 4], [50, 50, 60, 60], [0, 0, float('nan'), 4]):
            with self.subTest(box=box), self.assertRaisesRegex(RuntimeError, 'Car description failed'):
                describe_cars(self.state(box))
        describe.assert_not_called()


if __name__ == '__main__':
    unittest.main()
