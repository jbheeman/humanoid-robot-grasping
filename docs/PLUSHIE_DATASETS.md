# Plushie Detector Dataset Sources

The local training target is one YOLO detection class:

```yaml
names:
  0: plushie
```

Use these sources to bootstrap training, then bias the final dataset toward Unitree-camera frames from the real room.

## Best Starting Sources

1. Roboflow Universe teddy-bear datasets
   - Search: https://universe.roboflow.com/search?q=class%3A%22teddy+bear%22
   - Exports are often available in YOLO format.
   - Check each dataset license before use.

2. Roboflow stuffed-animal dataset
   - Example: https://universe.roboflow.com/babson-college-tvbix/stuffed-animals-d4fei
   - Small but directly relevant.
   - Useful as seed data, not enough by itself.

3. COCO `teddy bear`
   - Source: https://cocodataset.org/
   - Good baseline positives and negatives.
   - Label maps to `plushie` for this project.

4. Open Images V7
   - Source: https://storage.googleapis.com/openimages/web/factsfigures_v7.html
   - Very large object-detection dataset with box annotations.
   - Mine teddy-bear / stuffed-toy-like classes if available in the label map.

5. LVIS
   - Source: https://www.lvisdataset.org/
   - Fine-grained long-tail object vocabulary.
   - Useful if it contains plush/toy categories that COCO misses.

6. Objects365
   - Source: https://www.objects365.org/
   - Large object-detection corpus.
   - Useful for negatives and broad visual pretraining, but verify matching toy/plush categories before spending time on conversion.

## Installed Locally

COCO 2017 `teddy bear` and Open Images `Teddy bear` are installed as bootstrap sources:

```text
data/plushie/images/train/*: 6322
data/plushie/images/val/*:    225
data/plushie/labels/train/*: 6322
data/plushie/labels/val/*:    225
```

The split is:

```text
COCO train:        2140 teddy-bear positives + 2140 hard negatives
COCO val:            94 teddy-bear positives +   94 hard negatives
Open Images train: 1020 teddy-bear positives + 1020 hard negatives
Open Images val:     18 teddy-bear positives +   18 hard negatives
```

The default train-only local augmentation pass adds 3 variants per source train image:

```text
augmented train images: 18966
augmented train labels: 18966
total train images:     25288
total train labels:     25288
total val images:         225
total val labels:         225
```

The augmentation intentionally leaves validation untouched and simulates Unitree G1 deployment conditions: blur, motion smear, JPEG compression, light sensor noise, gray 1280x720 floor-like canvases, and smaller apparent object scale.

Manifest:

```text
data/plushie/dataset_manifest.json
```

## Dataset Mix

For the first usable plushie model:

```text
60% Unitree-camera labeled frames
25% public teddy/stuffed-animal positives
15% hard negatives: dog, pillow, shoes, bags, hard toys, people, furniture
```

Public data should bootstrap the model, not define the deployment distribution. The robot-camera angle, lighting, motion blur, carpet/furniture background, and distractors matter more than internet image volume.

## Local Import Rule

All trained weights and downloaded pretrained weights belong under:

```text
models/
```

Datasets belong under:

```text
data/
```

Runtime captures belong under:

```text
runs/captures/
```
