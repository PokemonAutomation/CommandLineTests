"""
Measure how each capture card / OS combination distorts colors, so that test images can be
re-colored to cover the whole range of capture cards seen in the wild.

Calibration data:
  The calibration folder (by default the folder this script is in, CommandLineTests/CaptureCardCalibration/)
  has one sub-folder per console (Switch1/, Switch2/) of screenshot pairs of the user icon "Background"
  color picker page, one scrolled to the top ("Up") and one to the bottom ("Down"). The color picker
  has the same 60 flat color patches on every console, so each pair gives 60
  (reference color -> captured color) samples.

  color_profile_labels.json in that folder maps each label to its image pairs (see `load_manifest()`):
    - "Reference": pairs dumped directly from the Switch (no capture card, no compression). They
      are averaged into the ground truth colors; the script warns if they disagree.
    - Every other key is a distortion label, e.g. "Win-MYPIN" (<OS>-<Card>), with the pairs that
      measured it. A pair's console is the folder its images are in, which only selects the
      screen layout. Pairs under one label are fitted together, and the script warns if they disagree.
  details/color_profile_label_explanation.json (optional, not in git) explains each label and names
  the contributor of each pair; both are copied into details/calibration_file_details.json.
  Images in the console folders that the manifest doesn't list are reported, so new calibration
  images aren't silently ignored.

Sampling:
  The selection cursor and the "current background" outline can bleed color into the edges of a
  patch (capture cards blur and chroma-subsample), so only a smaller square in the center of each
  patch is averaged (`--sample-fraction`, default 40% of the patch size). The script warns when a
  sampled square is not flat, which usually means the patch layout does not match the image.
  Use `--debug-dir` to write copies of the images with the sampled squares drawn on them.

Distortion model:
  Each distortion label is fitted with an affine RGB transform:
      captured = clip(M @ reference + b, 0, 255)       (M is 3x3, b is 3x1, RGB in 0-255)
  This models brightness/contrast, limited vs full range mismatches, white balance, saturation,
  hue shifts and YUV matrix mismatches (BT.601 vs BT.709). Channels that are clipped to 0 or 255
  in the captured image only constrain the fit in one direction: e.g. if the captured value is 255,
  any prediction >= 255 counts as correct.
  Cross-validation on the calibration data showed that adding per-channel gamma curves or
  2nd-order polynomial terms does not generalize better than this affine model. The remaining
  errors are on the most saturated colors (pure red, pure yellow), caused by gamut clipping
  inside the capture card's YUV pipeline.

Extreme transforms:
  To keep the number of test variants small, test images are not re-colored with every distortion.
  Instead the script picks extreme transforms along the three ways real capture cards differ:
  shadows (crushed or raised blacks), highlights (dim or bright whites) and saturation (low or high).
  For each of the 2 x 2 x 2 = 8 corners it picks the measured distortion that goes furthest toward
  that corner. Real cards don't vary along the axes independently, so several corners share a card and
  the result is usually fewer than 8 transforms. See `compute_extreme_transforms()`.

Output (all in the calibration folder by default):
  - details/calibration_file_details.json (not in git): the reference colors and, per distortion
    label, its pairs and contributors, the captured colors, fitted transform, error stats and
    interpretable YCbCr parameters, plus the envelope (min/max) across all labels.
  - color_profile_to_transform.json: {label: 3x4 matrix [M | b]} for every distortion label, plus
    "Reference" (the identity). Test code uses it to undo the distortion of the setup a test image
    came from. This file is the input of `simulate_capture_card_colors.py`.
  - color_profile_extremes.json: {label: 3x4 matrix} for just the extreme labels. Which corners each
    extreme covers is written to details/calibration_file_details.json.
  - A swatch report image (default <calibration folder>/color_profile_report.png) showing,
    for each label, the captured colors above the model's predicted colors.
  - A summary table printed to stdout.

Usage:
  python3 measure_capture_card_colors.py [calibration_folder] [--manifest files.json] [--output out.json] [--report out.png]
      [--transforms-output out.json] [--extremes-output out.json] [--sample-fraction 0.4] [--debug-dir dir]
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw


# Names of the 10 rows of the color picker. The "Up" page shows the first 5 rows, the "Down" page
# shows the last 5 rows. The two rows in the middle of the list (teal and cyan) are only partially
# visible on either page and are not sampled.
UP_ROW_NAMES = ["Red", "Orange", "Yellow", "YellowGreen", "Green"]
DOWN_ROW_NAMES = ["Blue", "Purple", "Pink", "Gray", "BlueGray"]
NUM_COLUMNS = 6     # Columns go from light (0) to dark (5).

# Patch names in the same order as the sampled colors: Red0, Red1, ..., BlueGray5.
PATCH_NAMES = [f"{row}{col}" for row in UP_ROW_NAMES + DOWN_ROW_NAMES for col in range(NUM_COLUMNS)]


@dataclass
class ConsoleLayout:
    """
    Pixel position of the visible part of each color patch, in 1920x1080 coordinates.
    Images of other resolutions (e.g. 1280x720 CFW screenshots) are scaled from these.
    `up_rows` / `down_rows` are (top, bottom) pixel rows of each patch row on the Up / Down page.
    Some rows are clipped by the scroll area, so their visible height is smaller than `patch_size`.
    """
    left: float          # x of the left edge of column 0
    pitch_x: float       # distance between the left edges of neighboring columns
    patch_size: float    # width of a patch
    up_rows: List[Tuple[float, float]]
    down_rows: List[Tuple[float, float]]

    def patch_boxes(self, page: str) -> List[Tuple[float, float, float, float]]:
        """Return (x, y, width, height) of every visible patch on `page` ("Up" or "Down"), row-major."""
        rows = self.up_rows if page == "Up" else self.down_rows
        return [
            (self.left + col * self.pitch_x, top, self.patch_size, bottom - top)
            for (top, bottom) in rows
            for col in range(NUM_COLUMNS)
        ]


# Measured from the CFW screenshots (Switch 1, 1280x720, scaled by 1.5) and the Switch 2 console
# screenshots. Switch 1 uses the same patch rows on both pages; Switch 2 scrolls differently.
LAYOUTS: Dict[str, ConsoleLayout] = {
    "Switch1": ConsoleLayout(
        left=133.5, pitch_x=153, patch_size=138,
        up_rows=[(177 + 153 * r, 177 + 153 * r + 138) for r in range(5)],
        down_rows=[(177 + 153 * r, 177 + 153 * r + 138) for r in range(5)],
    ),
    "Switch2": ConsoleLayout(
        left=844, pitch_x=160, patch_size=144,
        # The last Up row (Green) is clipped at the bottom of the scroll area.
        up_rows=[(200, 344), (360, 504), (520, 664), (680, 824), (840, 968)],
        # The first Down row (Blue) is clipped at the top of the scroll area.
        down_rows=[(127, 264), (280, 424), (440, 584), (600, 744), (760, 904)],
    ),
}

# The file in the calibration folder that maps each label to its calibration image pairs.
MANIFEST_NAME = "color_profile_labels.json"

# Optional file with a description of each label and the contributor of each pair.
EXPLANATION_NAME = "details/color_profile_label_explanation.json"

# The label of the undistorted reference pairs in color_profile_labels.json. In
# color_profile_to_transform.json it maps to the identity transform, for test images dumped straight
# from the Switch.
REFERENCE_LABEL = "Reference"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

# Pairs under one label whose colors differ from the label's average by more than this RMS (in 0-255
# levels) are reported: they probably measured different distortions and need separate labels.
CONSISTENCY_WARNING_RMS = 3.0

# Two different labels whose average colors are within this RMS of each other are reported as
# measuring practically the same distortion.
SIMILAR_LABEL_RMS = 2.0

# A sampled square whose per-channel standard deviation is above this is reported as not flat.
FLATNESS_WARNING_STDDEV = 4.0

# Captured channel values at or beyond these are treated as clipped by the capture pipeline.
CLIP_LOW = 0.5
CLIP_HIGH = 254.5

# The 3x4 transform [M | b] that leaves colors unchanged.
IDENTITY_TRANSFORM = np.hstack([np.eye(3), np.zeros((3, 1))])

# Gray levels used to measure how a transform moves the shadows and the highlights.
SHADOW_GRAYS = (0, 16, 32)
HIGHLIGHT_GRAYS = (192, 224, 255)

# When picking the card for an extreme corner, a card already picked for another corner is preferred
# if its score is within this much of the best card's (on the 0-1 axis scales), so similar cards
# don't both become extremes: e.g. Win-CheapCard-HighContrast is covered by Mac-Mirabox-HighContrast, and
# Win-Shadowcast (raised blacks, normal saturation) by Mac-Mirabox-GreenShift.
EXTREME_TIE_TOLERANCE = 0.2

# The three axes of the extreme transforms, with the names of their low and high ends.
EXTREME_AXES = [
    ("shadows", "crushed", "raised"),
    ("highlights", "dim", "bright"),
    ("saturation", "low", "high"),
]


@dataclass
class Calibration:
    """One Up + Down image pair from color_profile_labels.json."""
    console: str
    up_path: Path
    down_path: Path
    contributor: str = ""
    colors: Optional[np.ndarray] = None    # (60, 3) float, sampled average color of each patch
    stddevs: Optional[np.ndarray] = None   # (60,) float, max channel stddev inside each sample

    @property
    def name(self) -> str:
        """Short name for messages, e.g. "Switch1/Kuro-Win-MYPIN" from Switch1/Kuro-Win-MYPIN-Up.png."""
        stem = self.up_path.stem
        return f"{self.up_path.parent.name}/{stem[:-3] if stem.endswith('-Up') else stem}"


@dataclass
class Distortion:
    """
    One distortion label (e.g. "Win-MYPIN") with all the calibration pairs that measured it.
    After `analyze_distortion()`, `colors` is the average captured color of each patch over the
    pairs and `fit` holds the fitted transform and stats.
    """
    label: str
    description: str
    calibrations: List[Calibration]
    colors: Optional[np.ndarray] = None
    fit: Dict = field(default_factory=dict)

    @property
    def full_name(self) -> str:
        return self.label


# ---------------------------------------------------------------------------------------------
# Loading and sampling
# ---------------------------------------------------------------------------------------------

def load_manifest(folder: Path, manifest_path: Path, explanation_path: Path) -> Tuple[List[Calibration], List[Distortion]]:
    """
    Read color_profile_labels.json and return the reference pairs and the distortion labels with
    their pairs. Format:
        {
          "Reference": [{"up": "Switch1/CFW-BrightBG-Up.png", "down": "Switch1/CFW-BrightBG-Down.png"}],
          "Win-MYPIN": [{"up": "Switch1/Kuro-Win-MYPIN-Up.png", "down": "Switch1/Kuro-Win-MYPIN-Down.png"}]
        }
    Paths are relative to `folder`. The first folder of a path is the console, which must have a
    layout in `LAYOUTS`. If `explanation_path` exists, label descriptions and pair contributors are
    read from it:
        {"labels": {"Win-MYPIN": {"description": "...", "calibrations": {"<up path>": {"contributor": "Kuro"}}}}}
    Also warns about images in the console folders that the manifest doesn't list.
    Raises ValueError for a missing file, an unknown console, a label without pairs or a missing
    "Reference" label.
    """
    with open(manifest_path, encoding="utf-8") as file:
        data = json.load(file)
    explanations: Dict = {}
    if explanation_path.is_file():
        with open(explanation_path, encoding="utf-8") as file:
            explanations = json.load(file).get("labels", {})

    def make_calibration(entry: Dict, label: str) -> Calibration:
        console = Path(entry["up"]).parts[0]
        if console not in LAYOUTS:
            raise ValueError(f"label '{label}': {entry['up']} is not in a console folder with a patch layout "
                             f"(known: {', '.join(LAYOUTS)}).")
        paths = []
        for page in ("up", "down"):
            path = folder / entry[page]
            if not path.is_file():
                raise ValueError(f"label '{label}': {path} does not exist.")
            paths.append(path)
        details = explanations.get(label, {}).get("calibrations", {}).get(entry["up"], {})
        return Calibration(console, paths[0], paths[1], details.get("contributor", ""))

    if REFERENCE_LABEL not in data:
        raise ValueError(f"{manifest_path.name} has no '{REFERENCE_LABEL}' label.")
    references = [make_calibration(entry, REFERENCE_LABEL) for entry in data[REFERENCE_LABEL]]
    distortions = []
    for label, pairs in data.items():
        if label == REFERENCE_LABEL:
            continue
        if not pairs:
            raise ValueError(f"Label '{label}' has no calibration pairs.")
        calibrations = [make_calibration(pair, label) for pair in pairs]
        distortions.append(Distortion(label, explanations.get(label, {}).get("description", ""), calibrations))

    listed = {c.up_path.resolve() for c in references} | {c.down_path.resolve() for c in references}
    for distortion in distortions:
        for c in distortion.calibrations:
            listed |= {c.up_path.resolve(), c.down_path.resolve()}
    for console in LAYOUTS:
        console_dir = folder / console
        if not console_dir.is_dir():
            continue
        for path in sorted(console_dir.iterdir()):
            if path.suffix.lower() in IMAGE_EXTENSIONS and path.resolve() not in listed:
                print(f"Warning: {path.relative_to(folder)} is not listed in {manifest_path.name}, so it is not used.")
    return references, distortions


def sample_boxes(layout: ConsoleLayout, page: str, image_size: Tuple[int, int], fraction: float) -> List[Tuple[int, int, int, int]]:
    """
    Return the pixel boxes (x0, y0, x1, y1) to average for every patch on `page` of an image of
    `image_size` (width, height). Each box is a square centered in the visible part of the patch,
    with a side of `fraction` * the smaller visible side, so that colors bled in from the cursor
    and neighboring UI are excluded.
    """
    width, height = image_size
    sx, sy = width / 1920.0, height / 1080.0
    boxes = []
    for x, y, w, h in layout.patch_boxes(page):
        cx, cy = x + w / 2, y + h / 2
        half = fraction * min(w, h) / 2
        boxes.append((
            int(round((cx - half) * sx)), int(round((cy - half) * sy)),
            int(round((cx + half) * sx)), int(round((cy + half) * sy)),
        ))
    return boxes


def load_rgb(path: Path) -> np.ndarray:
    """Load an image as an (H, W, 3) float array in RGB order, dropping any alpha channel."""
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float64)


def sample_calibration(calibration: Calibration, fraction: float, debug_dir: Optional[Path]) -> None:
    """
    Fill `calibration.colors` / `calibration.stddevs` with the average color / flatness of the 60
    patches (30 from the Up image, then 30 from the Down image). Warns when the image is not 16:9,
    since the layout would not line up.
    """
    layout = LAYOUTS[calibration.console]
    colors, stddevs = [], []
    for page, path in (("Up", calibration.up_path), ("Down", calibration.down_path)):
        pixels = load_rgb(path)
        height, width = pixels.shape[:2]
        if abs(width / height - 16 / 9) > 0.01:
            print(f"Warning: {path} is {width}x{height}, not 16:9. Patch positions are probably wrong.")
        boxes = sample_boxes(layout, page, (width, height), fraction)
        for x0, y0, x1, y1 in boxes:
            block = pixels[y0:y1, x0:x1].reshape(-1, 3)
            colors.append(block.mean(axis=0))
            stddevs.append(block.std(axis=0).max())
        if debug_dir is not None:
            write_debug_image(path, boxes, debug_dir / f"{path.parent.name}-{path.stem}.png")
    calibration.colors = np.array(colors)
    calibration.stddevs = np.array(stddevs)
    for i in np.flatnonzero(calibration.stddevs > FLATNESS_WARNING_STDDEV):
        print(f"Warning: {calibration.name} patch {PATCH_NAMES[i]} is not flat "
              f"(stddev {calibration.stddevs[i]:.1f}). Check the layout with --debug-dir.")


def write_debug_image(path: Path, boxes: List[Tuple[int, int, int, int]], out_path: Path) -> None:
    """Save a copy of the image at `path` with the sampled squares outlined in magenta."""
    with Image.open(path) as image:
        image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    for box in boxes:
        draw.rectangle(box, outline=(255, 0, 255), width=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


# ---------------------------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------------------------

def clip_aware_residual(predicted: np.ndarray, captured: np.ndarray) -> np.ndarray:
    """
    Return `predicted - captured`, except that captured values clipped at 0 / 255 only count as
    errors when the prediction is on the wrong side of the clip: e.g. captured 255 and predicted
    270 is a perfect match because the capture card would have clipped 270 to 255 too.
    """
    residual = predicted - captured
    high = captured >= CLIP_HIGH
    low = captured <= CLIP_LOW
    residual[high] = np.minimum(residual[high], 0)
    residual[low] = np.maximum(residual[low], 0)
    return residual


def fit_affine(reference: np.ndarray, captured: np.ndarray, iterations: int = 50) -> np.ndarray:
    """
    Fit `captured = clip(M @ reference + b, 0, 255)` and return the (3, 4) matrix [M | b].
    Each output channel is fitted independently, minimizing the squared `clip_aware_residual()` of
    the clipped prediction.

    The clip makes the problem piecewise linear, so it is solved with Gauss-Newton steps:
    1. Start from a plain least squares fit on the samples whose captured value is not clipped.
    2. Samples whose current prediction is inside (0, 255) have a residual that is linear in the
       coefficients; samples predicted outside have a constant residual (e.g. predicted -10 is
       clipped to 0 no matter how negative it is). Solve least squares on the first group only.
       This is the exact minimizer of the current linear piece.
    3. Step towards that solution, halving the step until the total loss decreases. Stop when no
       step improves it.
    """
    design = np.hstack([reference, np.ones((len(reference), 1))])
    matrix = np.zeros((3, 4))
    for ch in range(3):
        target = captured[:, ch]

        def loss(c: np.ndarray) -> float:
            return float((clip_aware_residual(np.clip(design @ c, 0, 255), target) ** 2).sum())

        unclipped = (target > CLIP_LOW) & (target < CLIP_HIGH)
        coef, *_ = np.linalg.lstsq(design[unclipped], target[unclipped], rcond=None)
        current = loss(coef)
        for _ in range(iterations):
            predicted = design @ coef
            linear = (predicted > 0) & (predicted < 255)
            if linear.sum() < design.shape[1]:
                break
            candidate, *_ = np.linalg.lstsq(design[linear], target[linear], rcond=None)
            step = 1.0
            while step > 1e-3 and loss(coef + step * (candidate - coef)) >= current:
                step /= 2
            if step <= 1e-3:
                break
            coef = coef + step * (candidate - coef)
            current = loss(coef)
        matrix[ch] = coef
    return matrix


def apply_affine(matrix: np.ndarray, colors: np.ndarray) -> np.ndarray:
    """Apply a (3, 4) affine color transform to (N, 3) RGB colors in 0-255 and clip to 0-255."""
    return np.clip(colors @ matrix[:, :3].T + matrix[:, 3], 0, 255)


# ---------------------------------------------------------------------------------------------
# Color metrics and interpretable parameters
# ---------------------------------------------------------------------------------------------

def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert (N, 3) sRGB colors in 0-255 to CIELAB (D65), for perceptual Delta E."""
    c = rgb / 255.0
    linear = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = linear @ np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]).T
    xyz /= np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], axis=1)


# Full-range BT.709 RGB -> YCbCr, with RGB and Y in 0-1 and Cb/Cr in -0.5..0.5.
RGB_TO_YCBCR = np.array([
    [0.2126, 0.7152, 0.0722],
    [-0.1146, -0.3854, 0.5],
    [0.5, -0.4542, -0.0458],
])


def ycbcr_parameters(matrix: np.ndarray) -> Dict[str, float]:
    """
    Re-express an RGB affine transform in BT.709 YCbCr space, where the usual video controls are
    (approximately) separate terms. The transform in YCbCr is K @ ycbcr + k with
    K = T M T^-1 and k = T b. From that:
      - contrast:   K[Y, Y], how much luma is scaled (limited->full range expansion gives ~1.16).
      - brightness: k[Y] * 255, luma offset in 0-255 units (limited->full range gives ~-18.6).
      - saturation: sqrt(|det| of the CbCr block), the average chroma scale.
      - hue_degrees: rotation of the CbCr plane.
      - saturation_anisotropy: ratio of the CbCr block's singular values (1 = uniform saturation).
      - cb_offset / cr_offset: white balance tint in 0-255 units.
      - luma_from_chroma: magnitude of K[Y, CbCr], how much chroma leaks into luma, which is the
        signature of a BT.601 vs BT.709 matrix mismatch.
    """
    T = RGB_TO_YCBCR
    K = T @ matrix[:, :3] @ np.linalg.inv(T)
    k = T @ (matrix[:, 3] / 255.0)
    chroma = K[1:, 1:]
    singular_values = np.linalg.svd(chroma, compute_uv=False)
    return {
        "contrast": float(K[0, 0]),
        "brightness": float(k[0] * 255),
        "saturation": float(np.sqrt(abs(np.linalg.det(chroma)))),
        "hue_degrees": float(np.degrees(np.arctan2(chroma[1, 0] - chroma[0, 1], chroma[0, 0] + chroma[1, 1]))),
        "saturation_anisotropy": float(singular_values[0] / singular_values[1]),
        "cb_offset": float(k[1] * 255),
        "cr_offset": float(k[2] * 255),
        "luma_from_chroma": float(np.linalg.norm(K[0, 1:])),
    }


def analyze_distortion(distortion: Distortion, reference: np.ndarray) -> None:
    """
    Fit the affine model for a distortion label against the `reference` colors and store the
    transform and stats in `distortion.fit`. All the label's calibration pairs are fitted together
    (each pair adds 60 samples). `distortion.colors` becomes the average captured colors, and
    "calibration_rms" records how far each pair is from that average (in 0-255 levels). A large
    value means the pairs measured different distortions; see CONSISTENCY_WARNING_RMS.
    """
    stacked = np.vstack([c.colors for c in distortion.calibrations])
    stacked_reference = np.tile(reference, (len(distortion.calibrations), 1))
    distortion.colors = np.mean([c.colors for c in distortion.calibrations], axis=0)
    calibration_rms = [float(np.sqrt(((c.colors - distortion.colors) ** 2).mean())) for c in distortion.calibrations]
    for c, rms in zip(distortion.calibrations, calibration_rms):
        if rms > CONSISTENCY_WARNING_RMS:
            print(f"Warning: {c.name} differs from the average of label '{distortion.label}' by RMS {rms:.1f}. "
                  "It probably measured a different distortion; consider giving it its own label.")

    delta = stacked - stacked_reference
    delta_e = np.linalg.norm(srgb_to_lab(stacked) - srgb_to_lab(stacked_reference), axis=1)
    matrix = fit_affine(stacked_reference, stacked)
    residual = clip_aware_residual(apply_affine(matrix, stacked_reference), stacked)
    worst = int(np.abs(residual).max(axis=1).argmax()) % len(PATCH_NAMES)
    distortion.fit = {
        "matrix": matrix,
        "calibration_rms": calibration_rms,
        "raw_delta_min": delta.min(axis=0),
        "raw_delta_max": delta.max(axis=0),
        "raw_max_abs": float(np.abs(delta).max()),
        "raw_rms": float(np.sqrt((delta ** 2).mean())),
        "delta_e_mean": float(delta_e.mean()),
        "delta_e_max": float(delta_e.max()),
        "delta_e_worst_patch": PATCH_NAMES[int(delta_e.argmax()) % len(PATCH_NAMES)],
        "fit_rms": float(np.sqrt((residual ** 2).mean())),
        "fit_max_abs": float(np.abs(residual).max()),
        "fit_worst_patch": PATCH_NAMES[worst],
        "ycbcr": ycbcr_parameters(matrix),
    }


def find_similar_labels(distortions: List[Distortion]) -> List[Tuple[str, str, float]]:
    """Return (label, label, RMS) for every pair of labels whose average colors are within SIMILAR_LABEL_RMS."""
    similar = []
    for i, a in enumerate(distortions):
        for b in distortions[i + 1:]:
            rms = float(np.sqrt(((a.colors - b.colors) ** 2).mean()))
            if rms <= SIMILAR_LABEL_RMS:
                similar.append((a.label, b.label, rms))
    return similar


# ---------------------------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------------------------

def rounded(values, digits: int = 2):
    """Round numpy arrays / floats for compact JSON output."""
    return np.round(np.asarray(values, dtype=float), digits).tolist()


def build_json(references: List[Calibration], reference: np.ndarray, distortions: List[Distortion], fraction: float,
               extremes: List[Dict]) -> Dict:
    """
    Assemble details/calibration_file_details.json: reference colors, per-label fits, the envelope
    over all labels, and the extreme labels with the axis ranges and corners they cover.
    """
    raw = np.stack([extreme_axis_features(d.fit["matrix"]) for d in distortions], axis=1)
    captured = np.stack([d.colors for d in distortions])     # (labels, 60, 3)
    parameter_names = list(distortions[0].fit["ycbcr"].keys())
    return {
        "description": (
            "Capture card color distortions measured from the Switch icon background color picker, one per "
            "distortion label. Each label's transform is captured = clip(M @ reference + b, 0, 255) on RGB in "
            "0-255, stored as the 3x4 row-major matrix [M | b]. captured_colors is the average over the label's "
            "calibration pairs; calibration_rms is how far each pair is from that average."
        ),
        "sample_fraction": fraction,
        "reference_files": [[c.up_path.name, c.down_path.name] for c in references],
        "patch_names": PATCH_NAMES,
        "reference_colors": rounded(reference),
        "profiles": [
            {
                "label": d.label,
                "description": d.description,
                "calibrations": [
                    {
                        "console": c.console,
                        "contributor": c.contributor,
                        "files": [c.up_path.name, c.down_path.name],
                        "calibration_rms": round(rms, 2),
                    }
                    for c, rms in zip(d.calibrations, d.fit["calibration_rms"])
                ],
                "transform": rounded(d.fit["matrix"], 5),
                "captured_colors": rounded(d.colors),
                "raw_delta_min": rounded(d.fit["raw_delta_min"]),
                "raw_delta_max": rounded(d.fit["raw_delta_max"]),
                "raw_max_abs": round(d.fit["raw_max_abs"], 2),
                "raw_rms": round(d.fit["raw_rms"], 2),
                "delta_e_mean": round(d.fit["delta_e_mean"], 2),
                "delta_e_max": round(d.fit["delta_e_max"], 2),
                "delta_e_worst_patch": d.fit["delta_e_worst_patch"],
                "fit_rms": round(d.fit["fit_rms"], 2),
                "fit_max_abs": round(d.fit["fit_max_abs"], 2),
                "fit_worst_patch": d.fit["fit_worst_patch"],
                "ycbcr_parameters": {k: round(v, 4) for k, v in d.fit["ycbcr"].items()},
            }
            for d in distortions
        ],
        "envelope": {
            "per_patch_min": rounded(captured.min(axis=0)),
            "per_patch_max": rounded(captured.max(axis=0)),
            "per_patch_max_distance": rounded(np.linalg.norm(captured - reference, axis=2).max(axis=0)),
            "ycbcr_parameter_ranges": {
                name: [
                    round(min(d.fit["ycbcr"][name] for d in distortions), 4),
                    round(max(d.fit["ycbcr"][name] for d in distortions), 4),
                ]
                for name in parameter_names
            },
        },
        "extremes": {
            "axes": {
                axis: {"low": low_name, "high": high_name, "range": rounded([raw[a].min(), raw[a].max()], 3)}
                for a, (axis, low_name, high_name) in enumerate(EXTREME_AXES)
            },
            "labels": {e["label"]: {k: v for k, v in e.items() if k not in ("label", "transform")} for e in extremes},
        },
    }


def write_report(reference: np.ndarray, profiles: List[Distortion], out_path: Path) -> None:
    """
    Save a swatch image: the first line is the reference colors, then for each distortion label one
    line of captured colors (upper half of each swatch) over the model's predictions (lower half).
    """
    swatch, gap, label_width = 22, 6, 300
    line_height = swatch + gap
    width = label_width + len(PATCH_NAMES) * swatch
    height = (len(profiles) + 1) * line_height + gap
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    def draw_line(y: int, label: str, top: np.ndarray, bottom: np.ndarray) -> None:
        draw.text((4, y + swatch // 2 - 6), label, fill=(0, 0, 0))
        for i in range(len(PATCH_NAMES)):
            x = label_width + i * swatch
            draw.rectangle((x, y, x + swatch - 1, y + swatch // 2 - 1), fill=tuple(int(round(v)) for v in top[i]))
            draw.rectangle((x, y + swatch // 2, x + swatch - 1, y + swatch - 1), fill=tuple(int(round(v)) for v in bottom[i]))

    draw_line(gap, "Reference", reference, reference)
    for i, profile in enumerate(profiles):
        label = f"{profile.full_name}  (fit rms {profile.fit['fit_rms']:.1f})"
        draw_line(gap + (i + 1) * line_height, label, profile.colors, apply_affine(profile.fit["matrix"], reference))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def print_summary(distortions: List[Distortion], similar: List[Tuple[str, str, float]]) -> None:
    """
    Print one line per distortion label: number of calibration pairs and how much they disagree,
    raw error, model fit error and YCbCr parameters. Then the parameter ranges, and the labels that
    measure practically the same distortion.
    """
    print()
    print(f"{'Label':30s} {'pairs':>5s} {'spread':>6s} {'raw max':>7s} {'raw rms':>7s} {'dE avg':>6s} {'dE max':>6s} "
          f"{'fit rms':>7s} {'fit max':>7s} {'contr':>6s} {'bright':>6s} {'sat':>5s} {'hue':>6s} {'Cb':>6s} {'Cr':>6s}")
    for d in distortions:
        f, y = d.fit, d.fit["ycbcr"]
        print(f"{d.label:30s} {len(d.calibrations):5d} {max(f['calibration_rms']):6.1f} "
              f"{f['raw_max_abs']:7.1f} {f['raw_rms']:7.1f} {f['delta_e_mean']:6.1f} {f['delta_e_max']:6.1f} "
              f"{f['fit_rms']:7.2f} {f['fit_max_abs']:7.1f} "
              f"{y['contrast']:6.3f} {y['brightness']:6.1f} {y['saturation']:5.2f} {y['hue_degrees']:6.2f} "
              f"{y['cb_offset']:6.1f} {y['cr_offset']:6.1f}")
    print("(spread: largest RMS of a calibration pair from its label's average)")
    print()
    print("Parameter ranges over all labels:")
    for name in distortions[0].fit["ycbcr"]:
        values = [d.fit["ycbcr"][name] for d in distortions]
        print(f"  {name:22s} [{min(values):8.3f}, {max(values):8.3f}]")
    if similar:
        print()
        print(f"Labels measuring practically the same distortion (RMS <= {SIMILAR_LABEL_RMS}):")
        for a, b, rms in similar:
            print(f"  {a} ~ {b} (RMS {rms:.1f})")


# ---------------------------------------------------------------------------------------------
# Label mapping and extreme transforms
# ---------------------------------------------------------------------------------------------

def build_label_transforms(distortions: List[Distortion]) -> Dict[str, np.ndarray]:
    """
    Map every distortion label to its 3x4 transform [M | b], for color_profile_to_transform.json.
    Test code looks up the label of the setup a test image was captured with, and undoes that
    transform before applying the extreme transforms. REFERENCE_LABEL maps to the identity, for
    test images dumped straight from the Switch.
    """
    transforms = {REFERENCE_LABEL: IDENTITY_TRANSFORM}
    for d in distortions:
        transforms[d.label] = d.fit["matrix"]
    return transforms


def write_transform_json(path: Path, transforms: Dict[str, np.ndarray]) -> None:
    """
    Write {label: 3x4 matrix} as JSON with one matrix row per line, e.g.
        {
          "Win-MYPIN": [
            [0.98086, -0.05106, -0.00127, -4.94545],
            ...
          ]
        }
    Numbers are rounded to 5 decimals and written without exponents.
    """
    lines = ["{"]
    for i, (label, matrix) in enumerate(transforms.items()):
        rows = ",\n".join(
            "    [" + ", ".join(f"{v:.5f}".rstrip("0").rstrip(".") or "0" for v in row) + "]"
            for row in rounded(matrix, 5)
        )
        lines.append(f"  {json.dumps(label)}: [\n{rows}\n  ]" + ("," if i < len(transforms) - 1 else ""))
    lines.append("}")
    with open(path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


def extreme_axis_features(matrix: np.ndarray) -> np.ndarray:
    """
    Measure where a 3x4 transform sits on the three axes of the extreme transforms:
      - shadows: average luma change of the dark grays in SHADOW_GRAYS, in 0-255 levels.
        Negative means crushed blacks (e.g. cheap cards stretching limited range: about -26),
        positive means raised blacks (e.g. Mac capture: about +7).
      - highlights: average luma change of the light grays in HIGHLIGHT_GRAYS. Positive means
        brighter whites (cheap cards: up to +44), negative means dimmer (MYPIN: about -22).
      - saturation: average chroma gain, the mean of the Cb->Cb and Cr->Cr terms of the transform
        in BT.709 YCbCr space (1 = unchanged).
    Luma is measured before clipping. All three features are linear in the transform's
    coefficients, so the features of a weighted average of transforms are the same weighted
    average of their features. `compute_extreme_transforms()` relies on this.
    """
    T = RGB_TO_YCBCR

    def luma_change(level: float) -> float:
        gray = np.full(3, float(level))
        return float(T[0] @ (matrix[:, :3] @ gray + matrix[:, 3]) - level)

    chroma = (T @ matrix[:, :3] @ np.linalg.inv(T))[1:, 1:]
    return np.array([
        np.mean([luma_change(v) for v in SHADOW_GRAYS]),
        np.mean([luma_change(v) for v in HIGHLIGHT_GRAYS]),
        (chroma[0, 0] + chroma[1, 1]) / 2,
    ])


def corner_name(corner: np.ndarray) -> str:
    """Name a 0/1 corner of the extreme axes, e.g. (0, 1, 1) -> "shadows_crushed_highlights_bright_saturation_high"."""
    return "_".join(
        f"{axis}_{high_name if corner[a] else low_name}"
        for a, (axis, low_name, high_name) in enumerate(EXTREME_AXES)
    )


def compute_extreme_transforms(profiles: List[Distortion]) -> List[Dict]:
    """
    Pick the extreme transforms along the three ways real capture cards differ (see
    `extreme_axis_features()`): shadows crushed/raised x highlights dim/bright x saturation low/high.
    A test image re-colored with these covers the measured range without one variant per card.

    How it works:
    1. Measure the three features of every profile and scale each to 0-1 over the profiles, so
       0 is the lowest value any card has on that axis and 1 the highest.
    2. For each of the 8 corners of that box, score every card by how far it goes in the corner's
       direction from the center: score = (corner - 0.5) . position. E.g. for (0, 1, 1) a card scores
       high if it crushes shadows, brightens highlights and boosts saturation.
    3. The extreme for that corner is the card with the highest score. If a card already picked for
       another corner scores within EXTREME_TIE_TOLERANCE of the best, that card is used instead.
    4. Cards picked for several corners are listed once, with all the corners they cover.

    Why real cards and not blends: the features are linear in the transform, so blending transforms
    blends their features. A blend can never score higher in a direction than the best card in the
    blend, so the furthest point of the measured range in any direction is always a real card. Blends
    aimed at a corner no card reaches end up in the middle of the range, which is a milder test.
    Real cards also don't vary along the three axes independently: crushed blacks, bright whites and
    high saturation come together in the cheap cards. So several corners share a card.

    Returns one dict per picked distortion label, in the order of the first corner each covers
    (shadows changes fastest), with the label, the corners it covers, its position on the axes and
    its 3x4 transform.
    """
    raw = np.stack([extreme_axis_features(p.fit["matrix"]) for p in profiles], axis=1)   # (3, K)
    low, high = raw.min(axis=1), raw.max(axis=1)
    positions = (raw - low[:, None]) / (high - low)[:, None]

    corners = [np.array([(i >> axis) & 1 for axis in range(3)], dtype=float) for i in range(8)]
    scores = [(corner - 0.5) @ positions for corner in corners]
    picks = [int(np.argmax(score)) for score in scores]

    # Swap near-ties to cards that are already picked, until nothing changes.
    changed = True
    while changed:
        changed = False
        for c, score in enumerate(scores):
            for k in sorted(set(picks), key=lambda k: -score[k]):
                if k != picks[c] and score[k] >= score[picks[c]] - EXTREME_TIE_TOLERANCE and picks.count(picks[c]) == 1:
                    picks[c] = k
                    changed = True
                    break

    extremes = []
    for k in dict.fromkeys(picks):
        extremes.append({
            "label": profiles[k].label,
            "corners": [corner_name(corner) for corner, pick in zip(corners, picks) if pick == k],
            "position": rounded(positions[:, k], 3),
            "features": {axis: round(float(v), 3) for (axis, _, _), v in zip(EXTREME_AXES, raw[:, k])},
            "transform": rounded(profiles[k].fit["matrix"], 5),
        })
    return extremes


def print_extremes(extremes: List[Dict]) -> None:
    """Print each extreme card, the corners it covers, its position on the axes and its output for a few grays."""
    levels = np.array([0, 32, 64, 128, 192, 224, 255], dtype=float)
    grays = np.repeat(levels[:, None], 3, axis=1)
    print()
    print(f"Extreme transforms: {len(extremes)} labels cover the 8 corners "
          "(position = shadows, highlights, saturation on 0-1 scales):")
    for extreme in extremes:
        position = ", ".join(f"{v:.2f}" for v in extreme["position"])
        print(f"  {extreme['label']:30s} position ({position})")
        out = apply_affine(np.array(extreme["transform"]), grays)
        print("      grays " + ", ".join(str(int(v)) for v in levels) + " -> "
              + " ".join("(" + ",".join(f"{int(round(v))}" for v in color) + ")" for color in out))
        for corner in extreme["corners"]:
            print(f"      covers {corner}")


def default_calibration_dir() -> Path:
    """
    The calibration folder used when none is given: the folder this script lives in,
    CommandLineTests/CaptureCardCalibration/, next to the Switch1/ and Switch2/ image folders.
    """
    return Path(__file__).resolve().parent


def main() -> int:
    default_folder = default_calibration_dir()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", nargs="?", type=Path, default=default_folder,
                        help=f"calibration image folder (default: {default_folder})")
    parser.add_argument("--manifest", type=Path, help=f"label -> calibration pairs (default: <folder>/{MANIFEST_NAME})")
    parser.add_argument("--explanation", type=Path, help=f"optional label descriptions (default: <folder>/{EXPLANATION_NAME})")
    parser.add_argument("--output", type=Path, help="measurement details JSON path (default: <folder>/details/calibration_file_details.json)")
    parser.add_argument("--report", type=Path, help="swatch report image path (default: <folder>/color_profile_report.png)")
    parser.add_argument("--transforms-output", type=Path,
                        help="label -> transform JSON path (default: <folder>/color_profile_to_transform.json)")
    parser.add_argument("--extremes-output", type=Path,
                        help="extreme transforms JSON path (default: <folder>/color_profile_extremes.json)")
    parser.add_argument("--sample-fraction", type=float, default=0.4,
                        help="side of the sampled square as a fraction of the patch size (default: 0.4)")
    parser.add_argument("--debug-dir", type=Path, help="write images with the sampled squares drawn into this folder")
    args = parser.parse_args()

    folder: Path = args.folder
    if not folder.is_dir():
        print(f"Error: calibration folder {folder} does not exist.")
        return 1

    try:
        references, distortions = load_manifest(folder, args.manifest or folder / MANIFEST_NAME,
                                                args.explanation or folder / EXPLANATION_NAME)
    except (OSError, KeyError, ValueError) as e:
        print(f"Error reading the calibration manifest: {e!r}")
        return 1
    if not references or not distortions:
        print("Error: the manifest needs at least one reference pair and one distortion label.")
        return 1
    for calibration in references + [c for d in distortions for c in d.calibrations]:
        sample_calibration(calibration, args.sample_fraction, args.debug_dir)

    reference = np.mean([c.colors for c in references], axis=0)
    reference_spread = max(np.abs(c.colors - reference).max() for c in references)
    print(f"Reference colors: average of {', '.join(c.name for c in references)} "
          f"(max deviation {reference_spread:.1f}).")
    if reference_spread > 1.0:
        print("Warning: the reference image pairs disagree. Check that they are undistorted screenshots.")

    for distortion in distortions:
        analyze_distortion(distortion, reference)
    similar = find_similar_labels(distortions)

    extremes = compute_extreme_transforms(distortions)

    output = args.output or folder / "details" / "calibration_file_details.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as file:
        json.dump(build_json(references, reference, distortions, args.sample_fraction, extremes), file, indent=2)
    report = args.report or folder / "color_profile_report.png"
    write_report(reference, distortions, report)

    transforms_output = args.transforms_output or folder / "color_profile_to_transform.json"
    write_transform_json(transforms_output, build_label_transforms(distortions))

    extremes_output = args.extremes_output or folder / "color_profile_extremes.json"
    write_transform_json(extremes_output, {e["label"]: np.array(e["transform"]) for e in extremes})

    print_summary(distortions, similar)
    print_extremes(extremes)
    print()
    for path in (output, report, transforms_output, extremes_output):
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
