"""Analyze cars, people, and weapon candidates with LangGraph, YOLO, Grounding DINO, and Claude."""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from graph import run
from weapons import DEFAULT_DINO_MODEL
from steps import initial_state
from devices import resolve_device


def main() -> None:
    load_dotenv(Path(__file__).parent / '.env', override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('question', nargs='?', default='Describe the cars and people in this parking lot.')
    parser.add_argument('--image', type=Path, required=True, help='Local parking-lot image')
    parser.add_argument('--capacity', type=int, help='Known number of spaces in the pictured lot; omit if unknown')
    parser.add_argument('--confidence', type=float, default=0.25, help='YOLO confidence threshold (0, 1]')
    parser.add_argument('--yolo-model', default='yolo26x.pt', help='YOLO detection weights')
    parser.add_argument('--device', choices=('auto', 'mps', 'cuda', 'cpu'), default='auto',
                        help='Local detector device; auto prefers the Apple GPU (MPS), then CUDA, then CPU')
    parser.add_argument('--dino-model', default=DEFAULT_DINO_MODEL, help='Grounding DINO Hugging Face model ID or local directory')
    parser.add_argument('--weapon-threshold', type=float, default=0.35, help='Weapon box score threshold (0, 1]')
    parser.add_argument('--weapon-text-threshold', type=float, default=0.25, help='Weapon text score threshold (0, 1]')
    parser.add_argument('--offline', action='store_true', help='Run both local detectors without Claude; YOLO and Grounding DINO files must be available locally')
    parser.add_argument('--sharpen-crops', action='store_true',
                        help='Send original and mildly sharpened crops to Claude')
    parser.add_argument('--model', default=os.getenv('ANTHROPIC_MODEL') or 'claude-sonnet-4-6')
    args = parser.parse_args()
    try:
        state = initial_state(args.question, args.image, args.capacity, use_llm=not args.offline,
                              model=args.model, yolo_model=args.yolo_model, confidence=args.confidence,
                              sharpen_crops=args.sharpen_crops, dino_model=args.dino_model,
                              weapon_threshold=args.weapon_threshold, weapon_text_threshold=args.weapon_text_threshold,
                              device=resolve_device(args.device))
        if args.offline and not Path(args.yolo_model).is_file():
            raise ValueError('--offline requires existing local YOLO weights; pass --yolo-model /path/to/weights.pt.')
    except ValueError as error:
        parser.error(str(error))
    if not args.offline and not os.getenv('ANTHROPIC_API_KEY', '').strip():
        parser.error('Set ANTHROPIC_API_KEY in .env next to demo.py (copy .env.example), or use --offline.')
    print(f'Question: {state["question"]}\nImage: {state["image_path"]}\nCapacity: {args.capacity if args.capacity is not None else "unknown"}')
    print(f'Local detector device: {state["device"]}', flush=True)
    try:
        result = run(state)
    except (RuntimeError, ValueError) as error:
        parser.exit(1, f'Error: {error}\n')
    for event in result['trace']:
        print(f'  {event}')
    print('\n' + result['answer'])


if __name__ == '__main__':
    main()
