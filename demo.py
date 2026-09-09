"""Analyze a local parking-lot image with YOLO, LangGraph, and Claude."""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from graph import run
from steps import initial_state


def main() -> None:
    load_dotenv(Path(__file__).parent / '.env', override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('question', nargs='?', default='How crowded is this parking lot?')
    parser.add_argument('--image', type=Path, required=True, help='Local parking-lot image')
    parser.add_argument('--capacity', type=int, help='Known number of spaces in the pictured lot; omit if unknown')
    parser.add_argument('--confidence', type=float, default=0.25, help='YOLO confidence threshold (0, 1]')
    parser.add_argument('--yolo-model', default='yolo26x.pt', help='YOLO detection weights')
    parser.add_argument('--offline', action='store_true', help='Return local statistics without calling Claude; YOLO weights must be cached')
    parser.add_argument('--model', default=os.getenv('ANTHROPIC_MODEL') or 'claude-sonnet-4-6')
    args = parser.parse_args()
    try:
        state = initial_state(args.question, args.image, args.capacity, use_llm=not args.offline,
                              model=args.model, yolo_model=args.yolo_model, confidence=args.confidence)
        if args.offline and not Path(args.yolo_model).is_file():
            raise ValueError('--offline requires existing local YOLO weights; pass --yolo-model /path/to/weights.pt.')
    except ValueError as error:
        parser.error(str(error))
    if not args.offline and not os.getenv('ANTHROPIC_API_KEY', '').strip():
        parser.error('Set ANTHROPIC_API_KEY in .env next to demo.py (copy .env.example), or use --offline.')
    print(f'Question: {state["question"]}\nImage: {state["image_path"]}\nCapacity: {args.capacity if args.capacity is not None else "unknown"}')
    try:
        result = run(state)
    except (RuntimeError, ValueError) as error:
        parser.exit(1, f'Error: {error}\n')
    for event in result['trace']:
        print(f'  {event}')
    print('\n' + result['answer'])


if __name__ == '__main__':
    main()
