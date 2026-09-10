"""Local vehicle and person detection and deterministic parking occupancy calculations."""

import math
from io import BytesIO
from functools import lru_cache
from pathlib import Path
from typing import TypedDict

from weapons import DEFAULT_DINO_MODEL
from devices import resolve_device

VEHICLE_CLASSES = ('car', 'truck', 'bus', 'motorcycle')
DETECTION_CLASSES = (*VEHICLE_CLASSES, 'person')
DESCRIPTION_CLASSES = ('car', 'person')


class Detection(TypedDict):
    label: str
    confidence: float
    xyxy: list[float]


class ObjectDescription(TypedDict):
    label: str
    object_id: int
    detection_index: int
    xyxy: list[int]
    description: str


class State(TypedDict):
    question: str
    image_path: str
    parking_capacity: int | None
    yolo_model: str
    device: str
    confidence: float
    sharpen_crops: bool
    dino_model: str
    weapon_threshold: float
    weapon_text_threshold: float
    weapon_detections: list[Detection]
    detections: list[Detection]
    counts: dict[str, int]
    descriptions: list[ObjectDescription]
    total: int
    occupancy: float | None
    answer: str
    trace: list[str]
    use_llm: bool
    model: str


def initial_state(question: str, image_path: str | Path, parking_capacity: int | None = None, *,
                  use_llm: bool = True, model: str = 'claude-sonnet-4-6',
                  yolo_model: str = 'yolo26x.pt', confidence: float = 0.25,
                  sharpen_crops: bool = False, dino_model: str = DEFAULT_DINO_MODEL,
                  weapon_threshold: float = 0.35, weapon_text_threshold: float = 0.25,
                  device: str = 'auto') -> State:
    if device not in {'auto', 'mps', 'cuda', 'cpu'}:
        raise ValueError('Device must be auto, mps, cuda, or cpu.')
    if not question.strip():
        raise ValueError('Enter a nonempty question.')
    if parking_capacity is not None and (isinstance(parking_capacity, bool) or not isinstance(parking_capacity, int) or parking_capacity < 1):
        raise ValueError('Parking capacity must be a positive integer.')
    if not 0 < confidence <= 1:
        raise ValueError('Confidence must be greater than 0 and at most 1.')
    for value in (weapon_threshold, weapon_text_threshold):
        if isinstance(value, bool) or not 0 < value <= 1:
            raise ValueError('Weapon thresholds must be greater than 0 and at most 1.')
    if not dino_model.strip():
        raise ValueError('Provide a Grounding DINO model ID or local directory.')
    path = Path(image_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f'Image file does not exist: {path}')
    if path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}:
        raise ValueError('Provide a local JPG, PNG, BMP, WebP, or TIFF image.')
    return State(question=question.strip(), image_path=str(path), parking_capacity=parking_capacity,
                 yolo_model=yolo_model, device=device, confidence=confidence, sharpen_crops=sharpen_crops, detections=[], counts={}, descriptions=[],
                 total=0, occupancy=None, answer='', trace=[], use_llm=use_llm, model=model,
                 dino_model=dino_model, weapon_threshold=weapon_threshold,
                 weapon_text_threshold=weapon_text_threshold, weapon_detections=[])


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


def describe_detections(state: State) -> dict:
    """Describe each detected car or person using only its cropped image."""
    objects = [(index, detection) for index, detection in enumerate(state['detections'])
            if detection['label'] in DESCRIPTION_CLASSES]
    if not state['use_llm'] or not objects:
        reason = 'offline mode' if not state['use_llm'] else 'no cars or people detected'
        return {'descriptions': [],
                'trace': [*state['trace'], f'Skip object descriptions: {reason}']}

    from PIL import Image, ImageOps, ImageFilter
    from claude import describe_object

    descriptions = []
    try:
        # Use the same EXIF orientation as detection so coordinates match.
        with Image.open(state['image_path']) as source:
            image = ImageOps.exif_transpose(source).convert('RGB')
        ids = dict.fromkeys(DESCRIPTION_CLASSES, 0)
        for index, detection in objects:
            label = detection['label']
            ids[label] += 1
            object_id = ids[label]
            box = detection['xyxy']
            if len(box) != 4 or not all(math.isfinite(value) for value in box):
                raise ValueError(f'Invalid bounding box for {label} {object_id}.')
            x1, y1, x2, y2 = box
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f'Invalid bounding box for {label} {object_id}.')
            bounds = [max(0, math.floor(x1)), max(0, math.floor(y1)),
                      min(image.width, math.ceil(x2)), min(image.height, math.ceil(y2))]
            if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
                raise ValueError(f'Bounding box for {label} {object_id} is outside the image.')
            crop = image.crop(tuple(bounds))
            crop.thumbnail((768, 768), Image.Resampling.LANCZOS)
            buffer = BytesIO()
            crop.save(buffer, format='JPEG', quality=85)
            sharpened_jpeg = None
            if state.get('sharpen_crops', False):
                # Mild local sharpening; keep the original crop for comparison.
                sharpened = crop.filter(ImageFilter.UnsharpMask(radius=1.2, percent=125, threshold=3))
                enhanced_buffer = BytesIO()
                sharpened.save(enhanced_buffer, format='JPEG', quality=85)
                sharpened_jpeg = enhanced_buffer.getvalue()
            description = describe_object(buffer.getvalue(), state['model'], label,
                                          sharpened_jpeg=sharpened_jpeg)
            descriptions.append(ObjectDescription(label=label, object_id=object_id, detection_index=index,
                                               xyxy=bounds, description=description))
    except (OSError, ValueError, RuntimeError) as error:
        raise RuntimeError(f'Object description failed: {error}') from error
    return {'descriptions': descriptions,
            'trace': [*state['trace'], f'Claude described {len(descriptions)} car/person crop(s)']}


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
    """Summary evidence: statistics and description text, without paths or boxes."""
    evidence = {key: state[key] for key in ('counts', 'total', 'parking_capacity', 'occupancy')}
    evidence['descriptions'] = [
        {'label': item['label'], 'object_id': item['object_id'], 'description': item['description']}
        for item in state['descriptions']
    ]
    evidence['weapon_candidates'] = [
        {'label': item['label'], 'confidence': item['confidence']}
        for item in state['weapon_detections']
    ]
    return evidence


def explain(state: State) -> dict:
    if state['use_llm']:
        from claude import generate_answer
        text = generate_answer(state['question'], statistics(state), state['model'])
        event = 'Claude explains the calculated statistics'
    else:
        counts = ', '.join(f'{name}: {count}' for name, count in state['counts'].items())
        text = f"Detections: {counts}. Total vehicles: {state['total']}.\n"
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
    if state['use_llm']:
        for label, heading in (('person', 'People descriptions'), ('car', 'Car descriptions')):
            items = [item for item in state['descriptions'] if item['label'] == label]
            if items:
                details = [f"{label.capitalize()} {item['object_id']}: {item['description']}"
                           for item in items]
                text += f'\n\n{heading}:\n' + '\n'.join(details)
            elif label == 'person' and state['counts'].get('person', 0) == 0:
                text += '\n\nPeople descriptions:\nNo people detected.'
    text += '\n\nGrounding DINO weapon candidates:\n'
    if state['weapon_detections']:
        text += '\n'.join(
            f"Candidate {index}: {item['label']} (score {item['confidence']:.3f}), box {item['xyxy']}"
            for index, item in enumerate(state['weapon_detections'], 1)
        )
    else:
        text += 'No weapon candidates above the configured thresholds.'
    text += ('\nThese are unverified model matches, not confirmed weapons or calibrated probabilities. '
             'Missed detections are possible; no detections does not establish that anyone is unarmed. '
             'Candidates are not assigned to people.')
    return {'answer': text, 'trace': [*state['trace'], event]}
