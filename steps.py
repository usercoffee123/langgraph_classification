"""Local vehicle and person detection and deterministic parking occupancy calculations."""

from functools import lru_cache
from pathlib import Path
from typing import TypedDict

from dino import DEFAULT_DINO_MODEL
from devices import resolve_device

VEHICLE_CLASSES = ('car', 'truck', 'bus', 'motorcycle')
DETECTION_CLASSES = (*VEHICLE_CLASSES, 'person')


class Detection(TypedDict):
    label: str
    confidence: float
    xyxy: list[float]


class State(TypedDict):
    question: str
    image_path: str
    parking_capacity: int | None
    yolo_model: str
    device: str
    confidence: float
    dino_model: str
    dino_threshold: float
    dino_text_threshold: float
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
                  yolo_model: str = 'yolo26x.pt', confidence: float = 0.25,
                  dino_model: str = DEFAULT_DINO_MODEL,
                  dino_threshold: float = 0.35, dino_text_threshold: float = 0.25,
                  device: str = 'auto') -> State:
    if device not in {'auto', 'mps', 'cuda', 'cpu'}:
        raise ValueError('Device must be auto, mps, cuda, or cpu.')
    if not question.strip():
        raise ValueError('Enter a nonempty question.')
    if parking_capacity is not None and (isinstance(parking_capacity, bool) or not isinstance(parking_capacity, int) or parking_capacity < 1):
        raise ValueError('Parking capacity must be a positive integer.')
    if not 0 < confidence <= 1:
        raise ValueError('Confidence must be greater than 0 and at most 1.')
    for value in (dino_threshold, dino_text_threshold):
        if isinstance(value, bool) or not 0 < value <= 1:
            raise ValueError('DINO thresholds must be greater than 0 and at most 1.')
    if not dino_model.strip():
        raise ValueError('Provide a Grounding DINO model ID or local directory.')
    path = Path(image_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f'Image file does not exist: {path}')
    if path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}:
        raise ValueError('Provide a local JPG, PNG, BMP, WebP, or TIFF image.')
    return State(question=question.strip(), image_path=str(path), parking_capacity=parking_capacity,
                 yolo_model=yolo_model, device=device, confidence=confidence, detections=[], counts={},
                 total=0, occupancy=None, answer='', trace=[], use_llm=use_llm, model=model,
                 dino_model=dino_model, dino_threshold=dino_threshold,
                 dino_text_threshold=dino_text_threshold)


@lru_cache(maxsize=2)
def load_yolo(weights: str):
    # Import lazily so workflow tests do not require loading PyTorch or weights.
    from ultralytics import YOLO
    return YOLO(weights)


def detect_objects(state: State) -> dict:
    try:
        # Decode explicitly: only a local still image is passed to the detector.
        from PIL import Image, ImageOps
        with Image.open(state['image_path']) as source:
            image = ImageOps.exif_transpose(source).convert('RGB')
        yolo = load_yolo(state['yolo_model'])
        device = resolve_device(state.get('device', 'auto'))
        results = yolo.predict(source=image, conf=state['confidence'], verbose=False, save=False, device=device)
        result = results[0]
        if result.boxes is None:
            raise ValueError('The selected model does not return detection boxes.')
        detections = []
        counts = dict.fromkeys(DETECTION_CLASSES, 0)
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
            'trace': [*state['trace'], f'YOLO ({device}): detected {sum(counts.values())} vehicle/person detection(s) locally']}


def calculate_occupancy(state: State) -> dict:
    capacity = state['parking_capacity']
    total = sum(state['counts'].get(label, 0) for label in VEHICLE_CLASSES)
    if capacity is None:
        return {'total': total, 'occupancy': None,
                'trace': [*state['trace'], 'Occupancy unknown: no parking capacity supplied']}
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
        raise ValueError('Parking capacity must be a positive integer.')
    occupancy = total / capacity * 100
    return {'total': total, 'occupancy': occupancy,
            'trace': [*state['trace'], f'Calculate: {total} / {capacity} = {occupancy:.1f}%']}


def statistics(state: State) -> dict:
    return {key: state[key] for key in ('counts', 'total', 'parking_capacity', 'occupancy')}


def explain(state: State) -> dict:
    counts = ', '.join(f'{name}: {count}' for name, count in state['counts'].items())
    text = f"YOLO detections: {counts}. Total vehicles: {state['total']}."
    if state['occupancy'] is None:
        text += '\nOccupancy unknown: parking capacity was not supplied.'
    else:
        text += f"\nDetected-vehicle / supplied-capacity ratio: {state['occupancy']:.1f}%."
        if state['occupancy'] > 100:
            text += ' Detections exceed capacity; check image coverage and capacity.'
    text += '\nCounts are model detections and may miss or misclassify objects.'
    return {'answer': text, 'trace': [*state['trace'], 'Report YOLO counts locally']}
