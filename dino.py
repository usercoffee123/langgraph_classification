"""On-demand, text-prompted local object detection."""

import math
from functools import lru_cache
from typing import TYPE_CHECKING
from devices import resolve_device

if TYPE_CHECKING:
    from steps import State

DEFAULT_DINO_MODEL = 'IDEA-Research/grounding-dino-tiny'


@lru_cache(maxsize=2)
def load_grounding_dino(model_id: str, local_files_only: bool, device: str):
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id, local_files_only=local_files_only)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_id, local_files_only=local_files_only,
    )
    return processor, model.to(device).eval()


def predict_objects(image, model_id: str, threshold: float, text_threshold: float, *, local_files_only: bool,
                    device: str = 'auto', query: str):
    import torch

    processor, model = load_grounding_dino(model_id, local_files_only, resolve_device(device))
    inputs = processor(images=image, text=query, return_tensors='pt').to(model.device)
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


def query_objects(state: 'State', query: str) -> dict:
    """Inspect the full oriented image, independently of YOLO's person boxes."""
    from PIL import Image, ImageOps

    if not isinstance(query, str) or not any(c.isalnum() for c in query) or len(query) > 200:
        raise ValueError("Query must contain 1–200 characters naming visible objects.")
    query = query.strip().rstrip(".") + "."

    try:
        with Image.open(state['image_path']) as source:
            image = ImageOps.exif_transpose(source).convert('RGB')
        candidates = predict_objects(
            image, state['dino_model'], state['dino_threshold'], state['dino_text_threshold'],
            local_files_only=not state['use_llm'], device=state.get('device', 'auto'), query=query,
        )
        valid = []
        for candidate in candidates:
            score, box = candidate['confidence'], candidate['xyxy']
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError('Invalid object query score.')
            if len(box) != 4 or not all(math.isfinite(x) for x in box) or box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError('Invalid object query box.')
            bounds = [max(0, min(image.width, box[0])), max(0, min(image.height, box[1])),
                      max(0, min(image.width, box[2])), max(0, min(image.height, box[3]))]
            if score < state['dino_threshold'] or not candidate['label'].strip():
                continue
            if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
                continue
            valid.append({'label': candidate['label'], 'confidence': score, 'xyxy': bounds})
        # Keep overlapping different classes (e.g. a backpack on a chair).
        detections = []
        for candidate in sorted(valid, key=lambda item: item['confidence'], reverse=True):
            if all(candidate['label'] != kept['label'] or
                   _iou(candidate['xyxy'], kept['xyxy']) <= 0.5 for kept in detections):
                detections.append(candidate)
    except Exception as error:
        hint = (' Offline runs require cached model/processor files or --dino-model pointing to a local model directory.'
                if not state['use_llm'] and isinstance(error, OSError) else '')
        raise RuntimeError(f'Grounding DINO object query failed: {error}.{hint}') from error
    return {'query': query, 'count': len(detections), 'detections': detections,
            'note': 'Unverified model matches; scores are not calibrated probabilities. Zero matches do not prove absence.'}
