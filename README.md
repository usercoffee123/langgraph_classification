# Parking-lot analysis with LangGraph

Run commands from this repository root. A single LangGraph workflow processes a local image:

```text
START → detect → describe_detections → calculate → explain → END
```

YOLO records vehicle and person classes, confidence scores, and bounding boxes locally.
The `describe_detections` node crops each detection labeled `car` or `person` using its bounding box
and sends that crop to Claude for a short visual description of the car or person. Crops are JPEGs,
resized to at most 768 pixels per side; the full image and local paths are not sent.
Anything visible inside an object's bounding box is included in its crop.

Python counts cars, trucks, buses, motorcycles, and people. The vehicle total and
parking occupancy exclude people. Person crops are described using visible
clothing, posture, and activity, without attempting to identify people. The final `explain` node
requests a summary using statistics and description text, then appends every car/person description. Each description
in state includes a class label, an ID numbered within that class, a zero-based detection index, and the crop's pixel bounds
in the EXIF-oriented image. Trucks, buses, and motorcycles are counted but not described.

Online runs make one vision request per detected car or person plus one summary request.
`--offline` skips all Claude requests, including descriptions. No car or person detections
means no vision requests. A failed vision request stops the run with an error.

## Run

Requires Python 3.11+ and uv:

```sh
uv sync
# On a fresh checkout, copy .env.example to .env and set ANTHROPIC_API_KEY.
uv run python demo.py --image sample_images/parking_easy.jpg
```

The default question asks for both cars and people. Results include a separate
people-description section with clothing, colors, posture, visible activity, and
a gun-visibility assessment: "Gun visible", "Possible gun / unclear", or
"No gun visible in this crop". These are Claude's visual descriptions, not a
validated firearm detector. Hidden objects and details outside a crop are unknown.
For a simple example with **2 cars and 4 people**, all detected by YOLO26x at the
default settings:

```sh
uv run python demo.py --image sample_images/cars_and_people.jpg
```

The photo's capacity is unknown, so omit `--capacity`. Source and attribution
notes are in `sample_images/README.md`. The `parking_easy.jpg` sample contains
three cars and no people; the original `parking.jpg` is a difficult dense scene.

The app loads `.env` next to `demo.py`; shell environment variables take precedence.

```dotenv
ANTHROPIC_API_KEY=your-api-key-here
ANTHROPIC_MODEL=claude-sonnet-4-6
```

Use your own image and provide capacity only when the number of spaces in the
pictured area is known:

```sh
uv run python demo.py "Describe the cars and people." --image /path/to/parking.jpg
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

## Optional crop sharpening

```sh
uv run python demo.py --image sample_images/cctv_carjacking.jpg --sharpen-crops
```

This applies a mild Pillow UnsharpMask filter after cropping and resizing.
Claude receives the unsharpened crop and sharpened copy in the same request,
clearly labeled as the same subject. It is instructed to use the original as
primary evidence and retain uncertainty when details are ambiguous.
The original file, YOLO detections, and bounding boxes are unchanged.

Sharpening can emphasize edges but cannot restore missing pixels or prove a
blurry object is a gun. It may amplify noise; better descriptions are not
guaranteed. The option is off by default and sends twice as many crop images,
increasing image-token usage. Offline mode skips the description node's API calls.

Filter reference: [Pillow UnsharpMask](https://pillow.readthedocs.io/en/stable/reference/ImageFilter.html#PIL.ImageFilter.UnsharpMask).

## Files

- `demo.py`: command-line entry point and environment loading.
- `graph.py`: LangGraph state graph with detect, describe_detections, calculate, and explain nodes.
- `steps.py`: state, local detection, car/person crops, calculations, and explanation nodes.
- `claude.py`: ChatAnthropic calls for car/person images and aggregate statistics.
- `sample_images/`: sample photograph and attribution.
- `test_workflows.py`, `test_claude.py`, `test_descriptions.py`, `test_cli.py`: graph, model-integration,
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

Claude cannot recover missing detections because it sees only detected car/person crops
and aggregate statistics, not the full scene. Visual descriptions can also be
wrong; the prompt asks for visible color, body style, and features without guessing
make, model, or year for cars, or identity and sensitive traits for people.
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
failure handling, EXIF orientation, crop bounds, car and person descriptions, and the data sent to Claude. No API key or weights are needed.

Vision payload reference: [Claude image inputs](https://platform.claude.com/docs/en/build-with-claude/vision).
