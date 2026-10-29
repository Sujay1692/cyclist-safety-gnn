# Dataset: cyclist riding-safety photographs and pose keypoints (v1.0)

This is the dataset used in the paper cited in [README.md](README.md). It consists of 2970 photographs of cyclists, each labelled *safe* or *unsafe*, with the skeletal keypoints extracted from each photograph and the train/validation/test split used for every result in the paper.

| | Safe | Unsafe | Total |
|---|---|---|---|
| Training | 1063 | 1016 | 2079 |
| Validation | 228 | 218 | 446 |
| Test | 227 | 218 | 445 |
| **Total** | **1518** | **1452** | **2970** |

The split is stratified by class and was generated once by `create_split.py` with random seed 42.

## Where each part is

| Part | Location |
|---|---|
| Keypoints, one `.npy` per image | `keypoints/safe/`, `keypoints/unsafe/` in this repository |
| Labels and split | `dataset_split.csv` in this repository |
| Skeleton definition | `skeleton_edges.npy` (18 undirected edges as node indices 0–16), `skeleton_node_order.csv` |
| Images with no detected pose | `excluded_images.csv` (30 images, not part of the dataset) |
| Photographs (full resolution) | Zip files attached to the [v1.0 release](../../releases), with a `SHA256SUMS.txt` |

Each photograph has the same file stem as its keypoint array, so `image_data/unsafe/IMG_1234.jpg` corresponds to `keypoints/unsafe/IMG_1234.npy`.

## Collection

- **Riders and consent:** nine adult riders. All nine gave written consent for their photographs to be used and published.
- **Capture:** iPhone 14, five sessions between 12:00 and 18:00, at locations around Tierpark, Pankow and Kreuzberg in Berlin, Germany. This produced 9093 raw photographs.
- **Curation:** a manual review discarded failed or visibly wrong pose detections and redundant burst frames, leaving 3000 photographs. No pose was detected in 30 of these, leaving the 2970 in the dataset.
- **Privacy:** EXIF metadata (GPS location, device information, timestamps) was removed from the released photographs. The pixel data are unchanged.

## Labelling

A single annotator labelled each image against the five criteria below. An image is *safe* only if it meets the safe condition on all five.

| Criterion | Safe | Unsafe |
|---|---|---|
| Riding position | Ideal riding position | Not in a balanced, ideal riding posture |
| Gaze direction | Looks in the direction of motion | Does not look in the direction of motion |
| Hand placement | Both hands hold the handlebar | One or both hands off the handlebar |
| Foot placement | Both feet on the pedals | One or both feet off the pedals or in the air |
| Overall stability | No unusual movement or excessive twisting | Unusual movement or excessive twisting of handlebar or body |

## Keypoint format

Keypoints were extracted with MediaPipe Pose 0.10.14 (model complexity 2) by `extract_keypoints_gnn.py`. Before detection, each image was contrast-enhanced with CLAHE (clip limit 2.5, 8 × 8 tiles) on the lightness channel of LAB colour space. Images with no detection at a confidence threshold of 0.5 were retried at 0.15, which recovered 84 images.

Each `.npy` file is an array of shape (17, 3) holding raw MediaPipe output for the retained joints:

- **Columns:** `x` and `y` are normalised to image width and height (origin top-left; usually 0–1, slightly outside that range for joints near the image edge). `visibility` is MediaPipe's confidence, from 0 to 1.
- **Raw values:** the coordinates are not hip-centred or torso-scaled, and no joint angles are stored. `gnn_dataset.py` applies the paper's preprocessing (hip-midpoint centring, torso-length scaling, node-level joint angles with confidence) at load time.
- **Missing joints:** low-confidence joints are kept as detected. Nothing is interpolated or mirrored.

**Node order (rows 0–16):** nose, left eye, right eye, left ear, right ear, left shoulder, right shoulder, left elbow, right elbow, left wrist, right wrist, left hip, right hip, left knee, right knee, left ankle, right ankle.

## Known limitations

- **Narrow conditions:** daytime images only, of nine riders, from one camera, in one city.
- **Possible identity leakage across splits:** rider and session identity were not recorded, so the split is at image level. Images of the same rider, or near-duplicate burst frames, can appear in more than one split.
- **One annotator:** labels come from a single annotator, and inter-annotator agreement was not measured.
- **Side bias:** riders were photographed mostly from one side, so right-side joints are missing about 1.9 times as often as left-side joints.

## Terms of use

CC BY-NC 4.0 (see [LICENSE-DATA.md](LICENSE-DATA.md)). Please cite the paper if you use the data, and do not attempt to identify, contact or profile the people shown.

## Contact

Sujay Krishnamurthy Somayaji, SRH Berlin University of Applied Science (sujayksomayaji.blr@gmail.com).
