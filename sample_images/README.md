# Sample parking-lot image

## Clear weapon example: handgun_easy.jpg

Man Aiming S&W SD9, by Noah Wulf, 12 January 2019.
Source: https://commons.wikimedia.org/wiki/File:Man_Aiming_S%26W_SD9.jpg
Original: https://upload.wikimedia.org/wikipedia/commons/f/f5/Man_Aiming_S%26W_SD9.jpg
License: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0),
https://creativecommons.org/licenses/by-sa/4.0/
Downloaded unchanged (4272 x 2848 pixels).

A close-up of hands holding a handgun against an uncluttered background.
The previous automatic weapon-scan workflow, at default thresholds on MPS, detected one person and one
weapon candidate, with score 0.394. Grounding DINO returned the combined label
`handgun rifle shotgun`, so this demonstrates localization, not reliable subtype
classification. This is a single-image check, not an accuracy benchmark.

Run: uv run python demo.py "Look for a handgun." --image sample_images/handgun_easy.jpg --device mps
The current workflow searches only when Claude requests the DINO tool.

## Original parking example: parking.jpg

`parking.jpg`: Parking lot at HAA Kobe, by Laitr Keiows (9 January 2010).

Source: https://commons.wikimedia.org/wiki/File:Parking_lot_at_HAA_Kobe.jpg
Original: https://upload.wikimedia.org/wikipedia/commons/0/0d/Parking_lot_at_HAA_Kobe.jpg
License: Creative Commons Attribution 3.0 Unported (CC BY 3.0)
https://creativecommons.org/licenses/by/3.0/

Downloaded unchanged. The actual parking capacity is not supplied by the source;
any capacity used for a demo is an assumption, not ground truth.

## Easier example: parking_easy.jpg

Parking lot at Lake Meridian Park (2025) - 0003, by Roc0ast3r, 11 October 2025.
Source: https://commons.wikimedia.org/wiki/File:Parking_lot_at_Lake_Meridian_Park_(2025)_-_0003.jpg
Original: https://upload.wikimedia.org/wikipedia/commons/c/c7/Parking_lot_at_Lake_Meridian_Park_%282025%29_-_0003.jpg
License: CC0 1.0 Universal, https://creativecommons.org/publicdomain/zero/1.0/
Downloaded unchanged (5810 x 3873 pixels).

Three cars are visible. YOLO26x at default inference size 640 and confidence
threshold 0.25 detected all three, with confidences of approximately 0.932,
0.913, and 0.906. The total parking capacity is unknown; omit --capacity.
This is a single-image sanity check, not a general accuracy benchmark.

## Cars and people: cars_and_people.jpg

Toyota Corolla Hatchback promotional photograph, hosted by DARCARS Toyota of Baltimore.
Source page: https://www.darcarstoyotaofbaltimore.com/2022-toyota-corolla-hatchback-baltimore-md
Image URL: https://www.darcarstoyotaofbaltimore.com/static/brand-toyota/vehicle/2022/Toyota/Corolla-Hatchback/MRP/08.jpg
Downloaded unchanged (800 x 400 pixels) for this local example.
This is a promotional image; an open redistribution license has not been established.

Visible objects: 2 cars and 4 people, including one partly obscured behind a car.
YOLO26x at the default settings detected both cars and all four people.
Car confidence: approximately 0.942 and 0.940.
Person confidence: approximately 0.920, 0.910, 0.854, and 0.823.
These results are a single-image check, not a general accuracy benchmark.
Parking capacity is unknown; omit --capacity.

Run: uv run python demo.py --image sample_images/cars_and_people.jpg

## Security-camera example: cctv_carjacking.jpg

Surveillance still published by WPVI/ABC in a report about an armed carjacking
at a Sunoco gas station in Germantown, Philadelphia. Report dated October 3, 2022.
The reporting describes four armed men emerging from a van and threatening a driver;
the incident context is supplied by the source, not established by this app.
Source: https://abc7chicago.com/post/caught-on-video-surveillance-camera-philadelphia-pa-carjacking-in-germantown/12292009/
Image: https://cdn.abcotvs.com/dip/images/12291609_100322-wpvi-germantown-carjacking-video-430pm-video-CC-vid.jpg?w=992
Downloaded unchanged for local testing. Source credits WPVI-TV; no open
redistribution license has been established.

Run: uv run python demo.py --image sample_images/cctv_carjacking.jpg
Capacity is unknown. YOLO reports vehicle and person counts first. Ask a follow-up
question to search for other objects with the local Grounding DINO tool. Small or
obscured objects in this CCTV still may be missed. No automatic weapon scan runs.
