"""Local vehicle detection and deterministic parking occupancy calculations."""

from functools import lru_cache
from pathlib import Path
from typing import TypedDict

VEHICLE_CLASSES = ('car', 'truck', 'bus', 'motorcycle')


class Detection(TypedDict):
    label: str
    confidence: float
    xyxy: list[float]


class State(TypedDict):
    question: str
    image_path: str
    parking_capacity: int | None
    yolo_model: str
    confidence: float
    detections: list[Detection]
    counts: dict[str, int]
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
                 yolo_model=yolo_model, confidence=confidence, detections=[], counts={},
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
    """Explicit allowlist of evidence sent to Claude; no image, path, or boxes."""
    return {key: state[key] for key in ('counts', 'total', 'parking_capacity', 'occupancy')}


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
    return {'answer': text, 'trace': [*state['trace'], event]}
