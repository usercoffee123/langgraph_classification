"""Local vehicle detection and deterministic parking occupancy calculations."""

import math
from io import BytesIO
from functools import lru_cache
from pathlib import Path
from typing import TypedDict

VEHICLE_CLASSES = ('car', 'truck', 'bus', 'motorcycle')


class Detection(TypedDict):
    label: str
    confidence: float
    xyxy: list[float]


class CarDescription(TypedDict):
    car_id: int
    detection_index: int
    xyxy: list[int]
    description: str


class State(TypedDict):
    question: str
    image_path: str
    parking_capacity: int | None
    yolo_model: str
    confidence: float
    detections: list[Detection]
    counts: dict[str, int]
    car_descriptions: list[CarDescription]
    total: int
    occupancy: float | None
    answer: str
    trace: list[str]
    use_llm: bool
    model: str


def initial_state(question: str, image_path: str | Path, parking_capacity: int | None = None, *,
                  use_llm: bool = True, model: str = 'claude-sonnet-4-6',
                  yolo_model: str = 'yolo26x.pt', confidence: float = 0.25) -> State:
    if not question.strip():
        raise ValueError('Enter a nonempty question.')
    if parking_capacity is not None and (isinstance(parking_capacity, bool) or not isinstance(parking_capacity, int) or parking_capacity < 1):
        raise ValueError('Parking capacity must be a positive integer.')
    if not 0 < confidence <= 1:
        raise ValueError('Confidence must be greater than 0 and at most 1.')
    path = Path(image_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f'Image file does not exist: {path}')
    if path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}:
        raise ValueError('Provide a local JPG, PNG, BMP, WebP, or TIFF image.')
    return State(question=question.strip(), image_path=str(path), parking_capacity=parking_capacity,
                 yolo_model=yolo_model, confidence=confidence, detections=[], counts={}, car_descriptions=[],
                 total=0, occupancy=None, answer='', trace=[], use_llm=use_llm, model=model)


@lru_cache(maxsize=2)
def load_yolo(weights: str):
    # Import lazily so workflow tests do not require loading PyTorch or weights.
    from ultralytics import YOLO
    return YOLO(weights)


def detect_vehicles(state: State) -> dict:
    try:
        # Decode explicitly: only a local still image is passed to the detector.
        from PIL import Image, ImageOps
        with Image.open(state['image_path']) as source:
            image = ImageOps.exif_transpose(source).convert('RGB')
        yolo = load_yolo(state['yolo_model'])
        results = yolo.predict(source=image, conf=state['confidence'], verbose=False, save=False)
        result = results[0]
        if result.boxes is None:
            raise ValueError('The selected model does not return detection boxes.')
        detections = []
        counts = dict.fromkeys(VEHICLE_CLASSES, 0)
        for cls, confidence, xyxy in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(),
                                         result.boxes.xyxy.tolist(), strict=True):
            label = result.names[int(cls)]
            if label in counts and confidence >= state['confidence']:
                detections.append(Detection(label=label, confidence=float(confidence),
                                            xyxy=[float(value) for value in xyxy]))
                counts[label] += 1
    except Exception as error:
        raise RuntimeError(f'Local YOLO detection failed: {error}') from error
    return {'detections': detections, 'counts': counts,
            'trace': [*state['trace'], f'YOLO: detected {sum(counts.values())} vehicle(s) locally']}


def describe_cars(state: State) -> dict:
    """Crop each YOLO car box locally, then ask Claude to describe that crop."""
    cars = [(index, detection) for index, detection in enumerate(state['detections'])
            if detection['label'] == 'car']
    if not state['use_llm'] or not cars:
        reason = 'offline mode' if not state['use_llm'] else 'no cars detected'
        return {'car_descriptions': [],
                'trace': [*state['trace'], f'Skip car descriptions: {reason}']}

    from PIL import Image, ImageOps
    from claude import describe_car

    descriptions = []
    try:
        # Use the same EXIF orientation as detection so coordinates match.
        with Image.open(state['image_path']) as source:
            image = ImageOps.exif_transpose(source).convert('RGB')
        for car_id, (index, detection) in enumerate(cars, start=1):
            box = detection['xyxy']
            if len(box) != 4 or not all(math.isfinite(value) for value in box):
                raise ValueError(f'Invalid bounding box for car {car_id}.')
            x1, y1, x2, y2 = box
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f'Invalid bounding box for car {car_id}.')
            bounds = [max(0, math.floor(x1)), max(0, math.floor(y1)),
                      min(image.width, math.ceil(x2)), min(image.height, math.ceil(y2))]
            if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
                raise ValueError(f'Bounding box for car {car_id} is outside the image.')
            crop = image.crop(tuple(bounds))
            crop.thumbnail((768, 768), Image.Resampling.LANCZOS)
            buffer = BytesIO()
            crop.save(buffer, format='JPEG', quality=85)
            description = describe_car(buffer.getvalue(), state['model'])
            descriptions.append(CarDescription(car_id=car_id, detection_index=index,
                                               xyxy=bounds, description=description))
    except (OSError, ValueError, RuntimeError) as error:
        raise RuntimeError(f'Car description failed: {error}') from error
    return {'car_descriptions': descriptions,
            'trace': [*state['trace'], f'Claude described {len(descriptions)} car crop(s)']}


def calculate_occupancy(state: State) -> dict:
    capacity = state['parking_capacity']
    total = sum(state['counts'].values())
    if capacity is None:
        return {'total': total, 'occupancy': None,
                'trace': [*state['trace'], 'Occupancy unknown: no parking capacity supplied']}
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
        raise ValueError('Parking capacity must be a positive integer.')
    occupancy = total / capacity * 100
    return {'total': total, 'occupancy': occupancy,
            'trace': [*state['trace'], f'Calculate: {total} / {capacity} = {occupancy:.1f}%']}


def statistics(state: State) -> dict:
    """Summary evidence: statistics and description text, without paths or boxes."""
    evidence = {key: state[key] for key in ('counts', 'total', 'parking_capacity', 'occupancy')}
    evidence['car_descriptions'] = [
        {'car_id': car['car_id'], 'description': car['description']}
        for car in state['car_descriptions']
    ]
    return evidence


def explain(state: State) -> dict:
    if state['use_llm']:
        from claude import generate_answer
        text = generate_answer(state['question'], statistics(state), state['model'])
        event = 'Claude explains the calculated statistics'
    else:
        counts = ', '.join(f'{name}: {count}' for name, count in state['counts'].items())
        text = f"Detected vehicles: {counts}. Total detected: {state['total']}.\n"
        if state['occupancy'] is None:
            text += 'Occupancy unknown: parking capacity was not supplied.\n'
        else:
            text += (f"Detected-vehicle / supplied-capacity ratio: {state['occupancy']:.1f}% "
                     f"({state['total']} / {state['parking_capacity']}).\n")
            if state['occupancy'] > 100:
                text += 'Detections exceed capacity; check the capacity and image coverage.\n'
        text += ('These are model detections, not a verified vehicle count or occupied-space measurement. '
                 'Dense scenes and occluded vehicles can cause severe undercounting. '
                 'These statistics alone cannot establish how crowded the lot is.')
        event = 'Return local statistics without Claude'
    if state['car_descriptions']:
        details = [f"Car {car['car_id']}: {car['description']}" for car in state['car_descriptions']]
        text += '\n\nCar descriptions:\n' + '\n'.join(details)
    return {'answer': text, 'trace': [*state['trace'], event]}
