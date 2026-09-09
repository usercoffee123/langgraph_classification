# Parking-lot analysis with LangGraph

Run commands from this repository root. A single LangGraph workflow processes a local image:

```text
START → local YOLO detection → Python calculation → Claude explanation → END
```

YOLO records vehicle classes, confidence scores, and bounding boxes locally.
Python counts cars, trucks, buses, and motorcycles. Claude receives only the
question and aggregate statistics; the image, path, and boxes remain local.

## Run

Requires Python 3.11+ and uv:

```sh
uv sync
# On a fresh checkout, copy .env.example to .env and set ANTHROPIC_API_KEY.
uv run python demo.py --image sample_images/parking_easy.jpg
```

The app loads `.env` next to `demo.py`; shell environment variables take precedence.

```dotenv
ANTHROPIC_API_KEY=your-api-key-here
ANTHROPIC_MODEL=claude-sonnet-4-6
```

Use your own image and provide capacity only when the number of spaces in the
pictured area is known:

```sh
uv run python demo.py "What vehicles were detected?" --image /path/to/parking.jpg
uv run python demo.py --image /path/to/parking.jpg --capacity 50
```

`--capacity 50` means 50 parking spaces. Python calculates
`detected vehicles / capacity * 100`; this is a detection-to-capacity ratio,
not a verified measurement of occupied spaces. With no capacity, occupancy is
reported as unknown. Ratios over 100% are flagged rather than clamped.

The first online run downloads `yolo26x.pt` if needed. `--yolo-model` selects
other detection weights, `--confidence` changes the default 0.25 threshold, and
`--model` overrides the Claude model. Inputs must be local JPG, PNG, BMP, WebP,
or TIFF images. There is no browser upload interface.

To run the graph with a local statistical summary and no Claude request, use
existing local weights:

```sh
uv run python demo.py --image sample_images/parking_easy.jpg --offline --yolo-model ./yolo26x.pt
```

## Files

- `demo.py`: command-line entry point and environment loading.
- `graph.py`: LangGraph state graph with detect, calculate, and explain nodes.
- `steps.py`: state, local detection, calculations, and explanation node.
- `claude.py`: ChatAnthropic call using aggregate statistics.
- `sample_images/`: sample photograph and attribution.
- `test_workflows.py`, `test_claude.py`, `test_cli.py`: graph, model-integration,
  and command-line tests.

ChatAnthropic and LangChain Core provide the Claude model adapter and message
utilities. All workflow orchestration uses LangGraph.

## Accuracy limitations

The recommended `sample_images/parking_easy.jpg` has three clearly separated cars.
YOLO26x detected all three at the default settings, with confidence scores above
0.90. Its total parking capacity is unknown, so omit `--capacity`.

The original `sample_images/parking.jpg` shows a dense car-auction lot with many occluded vehicles.
YOLO26n detected only 6 cars at inference size 640 and 18 at size 1280 in a local
check, visibly undercounting the scene. Increasing resolution alone did not solve
this case. The current YOLO26x default detected 29 cars at size 640 and confidence
0.25 on the same image, which still visibly undercounts it. Its actual capacity
is unknown; do not use 50 for this sample.

Claude cannot recover missing detections because it receives only statistics.
It is instructed not to infer crowdedness or available spaces. Reliable occupancy
requires validated detections and capacity for the same pictured area, or a
parking-space occupancy detector. This example has no space segmentation, lot
boundary, or parked-versus-moving classification.

## Tests

```sh
uv run python -m unittest -v
```

Tests run the actual LangGraph and local image decoding with mocked YOLO and
Claude. They check filtering, arithmetic, unknown capacity, input validation,
failure handling, and the data sent to Claude. No API key or weights are needed.
