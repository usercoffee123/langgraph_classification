"""Local, text-prompted weapon candidates for the LangGraph workflow."""

import math
from functools import lru_cache
from typing import TYPE_CHECKING
from devices import resolve_device

if TYPE_CHECKING:
    from steps import State

WEAPON_PROMPT = 'handgun. rifle. shotgun. knife.'
DEFAULT_DINO_MODEL = 'IDEA-Research/grounding-dino-tiny'


@lru_cache(maxsize=2)
def load_grounding_dino(model_id: str, local_files_only: bool, device: str):
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id, local_files_only=local_files_only)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_id, local_files_only=local_files_only,
    )
    return processor, model.to(device).eval()


def predict_weapons(image, model_id: str, threshold: float, text_threshold: float, *, local_files_only: bool,
                    device: str = 'auto'):
    import torch

    processor, model = load_grounding_dino(model_id, local_files_only, resolve_device(device))
    inputs = processor(images=image, text=WEAPON_PROMPT, return_tensors='pt').to(model.device)
    with torch.inference_mode():
        outputs = model(**inputs)
    result = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, threshold=threshold, text_threshold=text_threshold,
        target_sizes=[(image.height, image.width)],
    )[0]
    # Transformers 5.17 can decode an empty token batch as [''] even when
    # no boxes passed the threshold. There are no candidates in that case.
    if len(result['scores']) == 0 and len(result['boxes']) == 0:
        return []
    return [
        {'label': label, 'confidence': float(score), 'xyxy': box}
        for label, score, box in zip(result['text_labels'], result['scores'].tolist(),
                                     result['boxes'].tolist(), strict=True)
    ]


def _iou(a, b):
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return overlap / (area_a + area_b - overlap)


def detect_weapons(state: 'State') -> dict:
    """Inspect the full oriented image, independently of YOLO's person boxes."""
    from PIL import Image, ImageOps

    try:
        with Image.open(state['image_path']) as source:
            image = ImageOps.exif_transpose(source).convert('RGB')
        candidates = predict_weapons(
            image, state['dino_model'], state['weapon_threshold'], state['weapon_text_threshold'],
            local_files_only=not state['use_llm'], device=state.get('device', 'auto'),
        )
        valid = []
        for candidate in candidates:
            score, box = candidate['confidence'], candidate['xyxy']
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError('Invalid weapon detection score.')
            if len(box) != 4 or not all(math.isfinite(x) for x in box) or box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError('Invalid weapon detection box.')
            bounds = [max(0, min(image.width, box[0])), max(0, min(image.height, box[1])),
                      max(0, min(image.width, box[2])), max(0, min(image.height, box[3]))]
            if score < state['weapon_threshold'] or not candidate['label'].strip():
                continue
            if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
                continue
            valid.append({'label': candidate['label'], 'confidence': score, 'xyxy': bounds})
        # Suppress duplicate prompt matches, regardless of their text label.
        detections = []
        for candidate in sorted(valid, key=lambda item: item['confidence'], reverse=True):
            if all(_iou(candidate['xyxy'], kept['xyxy']) <= 0.5 for kept in detections):
                detections.append(candidate)
    except Exception as error:
        hint = (' Offline runs require cached model/processor files or --dino-model pointing to a local model directory.'
                if not state['use_llm'] and isinstance(error, OSError) else '')
        raise RuntimeError(f'Grounding DINO weapon detection failed: {error}.{hint}') from error
    return {'weapon_detections': detections,
            'trace': [*state['trace'], f'Grounding DINO: {len(detections)} weapon candidate(s) locally']}
