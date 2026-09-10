"""Report every detected YOLO class, then ask questions using Claude and local Grounding DINO."""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from graph import run
from dino import DEFAULT_DINO_MODEL
from steps import initial_state
from devices import resolve_device
from claude import ImageConversation


def main() -> None:
    load_dotenv(Path(__file__).parent / '.env', override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('question', nargs='?', help='Ask one question after counting; omit for interactive chat')
    parser.add_argument('--image', type=Path, required=True, help='Local image')
    parser.add_argument('--capacity', type=int, help='Known parking capacity; omit if unknown')
    parser.add_argument('--confidence', type=float, default=0.25, help='YOLO confidence threshold (0, 1]')
    parser.add_argument('--yolo-model', default='yolo26x.pt')
    parser.add_argument('--device', choices=('auto', 'mps', 'cuda', 'cpu'), default='auto')
    parser.add_argument('--dino-model', default=DEFAULT_DINO_MODEL)
    parser.add_argument('--dino-threshold', type=float, default=0.35)
    parser.add_argument('--dino-text-threshold', type=float, default=0.25)
    parser.add_argument('--offline', action='store_true', help='YOLO counts only, using local weights; no Claude or DINO calls')
    parser.add_argument('--model', default=os.getenv('ANTHROPIC_MODEL') or 'claude-sonnet-4-6')
    args = parser.parse_args()
    if args.offline and args.question:
        parser.error('Questions require Claude; omit --offline, or omit the question for local counts.')
    try:
        state = initial_state(args.question or 'Count cars and people.', args.image, args.capacity,
                              use_llm=not args.offline, model=args.model, yolo_model=args.yolo_model,
                              confidence=args.confidence, dino_model=args.dino_model,
                              dino_threshold=args.dino_threshold, dino_text_threshold=args.dino_text_threshold,
                              device=resolve_device(args.device))
        if args.offline and not Path(args.yolo_model).is_file():
            raise ValueError('--offline requires existing local YOLO weights; pass --yolo-model /path/to/weights.pt.')
    except ValueError as error:
        parser.error(str(error))
    print(f'Image: {state["image_path"]}\nLocal detector device: {state["device"]}', flush=True)
    try:
        result = run(state)
    except (RuntimeError, ValueError) as error:
        parser.exit(1, f'Error: {error}\n')
    print(result['answer'], flush=True)
    if args.offline:
        return
    if not os.getenv('ANTHROPIC_API_KEY', '').strip():
        print('Set ANTHROPIC_API_KEY in .env to ask follow-up questions. YOLO counts above are local.')
        if args.question:
            parser.exit(1)
        return
    conversation = ImageConversation(result)
    if args.question:
        try:
            print(conversation.ask(args.question))
        except (RuntimeError, ValueError) as error:
            parser.exit(1, f'Error: {error}\n')
        return
    print('Ask about other objects (e.g. "Are there bicycles?"). Type quit to exit.')
    while True:
        try:
            question = input('You> ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {'quit', 'exit'}:
            break
        if not question:
            continue
        try:
            print('Claude> ' + conversation.ask(question), flush=True)
        except (RuntimeError, ValueError) as error:
            print(f'Error: {error}', file=sys.stderr)


if __name__ == '__main__':
    main()
