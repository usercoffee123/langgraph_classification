import base64
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageFilter
from langchain_core.messages import AIMessage

from graph import run
from steps import describe_detections, initial_state
from test_workflows import detector_result


class ObjectDescriptionTests(unittest.TestCase):
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
        self.assertEqual([car['detection_index'] for car in result['descriptions']], [0, 2])
        self.assertEqual([car['xyxy'] for car in result['descriptions']],
                         [[0, 0, 20, 20], [20, 0, 40, 20]])
        self.assertIn('Car 1: A red car.', result['answer'])
        self.assertIn('Car 2: A blue car.', result['answer'])
        self.assertEqual(state['descriptions'], [])
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

    @patch('steps.load_yolo')
    @patch('claude.ChatAnthropic')
    def test_people_get_separate_crops_labels_and_do_not_increase_occupancy(self, model, load):
        detection = detector_result([0, 4, 4])
        detection.boxes.xyxy.tolist = lambda: [[0, 0, 20, 20], [20, 0, 40, 20], [20, 0, 40, 20]]
        load.return_value.predict.return_value = [detection]
        model.return_value.invoke.side_effect = [AIMessage(content='A red car.'),
                                                 AIMessage(content='A person in blue.'),
                                                 AIMessage(content='A standing person.'),
                                                 AIMessage(content='One vehicle and two people.')]
        result = run(initial_state('Describe cars and people.', self.path, 10))
        self.assertEqual(result['counts']['person'], 2)
        self.assertEqual(result['total'], 1)
        self.assertEqual(result['occupancy'], 10)
        self.assertEqual([(item['label'], item['object_id'], item['detection_index'])
                          for item in result['descriptions']],
                         [('car', 1, 0), ('person', 1, 1), ('person', 2, 2)])
        self.assertIn('Person 1: A person in blue.', result['answer'])
        self.assertIn('Person 2: A standing person.', result['answer'])
        for call in model.return_value.invoke.call_args_list[1:3]:
            messages = call.args[0]
            self.assertIn('clothing', messages[0][1])
            self.assertEqual(messages[1][1][1]['text'], 'Describe this person.')
            data = messages[1][1][0]['source']['data']
            with Image.open(BytesIO(base64.b64decode(data))) as crop:
                self.assertGreater(crop.getpixel((10, 10))[2], 240)

    @patch('claude.ChatAnthropic')
    def test_sharpening_sends_original_and_enhanced_crop_without_changing_source(self, model):
        model.return_value.invoke.return_value = AIMessage(content='A person; details unclear.')
        image = Image.new('RGB', (40, 20), (80, 80, 80))
        image.paste((170, 170, 170), (20, 0, 40, 20))
        image.filter(ImageFilter.GaussianBlur(0.7)).save(self.path)
        source_bytes = self.path.read_bytes()
        state = self.state([0, 0, 40, 20])
        state['detections'][0]['label'] = 'person'
        describe_detections(state)
        baseline = model.return_value.invoke.call_args.args[0][1][1][0]['source']['data']
        state['sharpen_crops'] = True
        result = describe_detections(state)
        messages = model.return_value.invoke.call_args.args[0]
        blocks = messages[1][1]
        sources = [block['source'] for block in blocks if block['type'] == 'image']
        self.assertEqual(len(sources), 2)
        self.assertEqual(sources[0]['data'], baseline)
        self.assertNotEqual(sources[1]['data'], baseline)
        for source in sources:
            with Image.open(BytesIO(base64.b64decode(source['data']))) as crop:
                self.assertEqual(crop.size, (40, 20))
        self.assertEqual(self.path.read_bytes(), source_bytes)
        self.assertEqual(result['descriptions'][0]['xyxy'], [0, 0, 40, 20])
        self.assertIn('original crop', blocks[0]['text'])
        self.assertIn('SAME crop', blocks[2]['text'])
        self.assertIn('primary evidence', messages[0][1])

    @patch('claude.describe_object', return_value='A car.')
    def test_crop_uses_exif_orientation_matching_detector(self, describe):
        exif = Image.Exif()
        exif[274] = 6  # Rotate clockwise: red half is now above blue half.
        self.image.save(self.path, exif=exif)
        describe_detections(self.state([0, 0, 20, 20]))
        with Image.open(BytesIO(describe.call_args.args[0])) as crop:
            self.assertGreater(crop.getpixel((10, 15))[0], 240)
            self.assertFalse(crop.getexif())

    @patch('claude.describe_object', return_value='A car.')
    def test_large_crop_is_bounded(self, describe):
        Image.new('RGB', (2000, 1000), 'red').save(self.path)
        describe_detections(self.state([0, 0, 2000, 1000]))
        with Image.open(BytesIO(describe.call_args.args[0])) as crop:
            self.assertEqual(crop.size, (768, 384))

    @patch('claude.describe_object')
    def test_offline_and_no_car_skip_api(self, describe):
        self.assertEqual(describe_detections(self.state(use_llm=False, sharpen_crops=True))['descriptions'], [])
        state = self.state()
        state['detections'][0]['label'] = 'truck'
        self.assertEqual(describe_detections(state)['descriptions'], [])
        describe.assert_not_called()

    @patch('claude.describe_object')
    def test_invalid_boxes_fail_before_api(self, describe):
        for box in ([5, 5, 1, 1], [0, 0, 0, 4], [50, 50, 60, 60], [0, 0, float('nan'), 4]):
            with self.subTest(box=box), self.assertRaisesRegex(RuntimeError, 'Object description failed'):
                describe_detections(self.state(box))
        describe.assert_not_called()


if __name__ == '__main__':
    unittest.main()
