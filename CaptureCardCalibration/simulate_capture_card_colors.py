"""
Re-color test images as if they were captured through different capture cards, using the profiles
measured by `measure_capture_card_colors.py`.

This is the Python prototype of the capture card distortion framework described in README.md.
A test image goes through a pipeline of stages:
  1. Color transform: one of the measured profiles (captured = clip(M @ rgb + b)), a random
     mixture of the measured profiles (optionally pushed further away from the identity with
     `--extrapolate` to cover cards that distort more than the ones measured so far), or one of
     the extreme transforms (`--extremes`) that the C++ tests are planned to use.
  2. Optional spatial / compression stages that capture cards add on top of the color shift:
     4:2:0 chroma subsampling (color bleeding at edges), Gaussian blur, JPEG compression and noise.

Usage examples:
  # One output per distortion label:
  python3 simulate_capture_card_colors.py test.png --out-dir out/

  # Only the labels that contain "Shadowcast" or "MYPIN":
  python3 simulate_capture_card_colors.py test.png --out-dir out/ --profile Shadowcast --profile MYPIN

  # 10 random mixtures of the measured profiles, 20% stronger than measured, with MJPEG-like artifacts:
  python3 simulate_capture_card_colors.py test.png --out-dir out/ --random 10 --seed 1 --extrapolate 1.2 \\
      --chroma-subsample --jpeg-quality 80

  # The extreme transforms, with a contact sheet for a quick visual check:
  python3 simulate_capture_card_colors.py test.png --out-dir out/ --extremes --sheet
"""

import argparse
import io
import json
import re
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


IDENTITY = np.hstack([np.eye(3), np.zeros((3, 1))])


# A color stage: takes an (H, W, 3) uint8 RGB image and returns the re-colored image.
ColorFunction = Callable[[np.ndarray], np.ndarray]

def default_calibration_dir() -> Path:
    """
    The calibration folder used when none is given: the folder this script lives in,
    CommandLineTests/CaptureCardCalibration/, next to the Switch1/ and Switch2/ image folders.
    """
    return Path(__file__).resolve().parent


# The calibration folder that `measure_capture_card_colors.py` writes its JSON files into by default.
DEFAULT_CALIBRATION_DIR = default_calibration_dir()


def load_profiles(path: Path, name_filters: List[str]) -> List[Tuple[str, np.ndarray]]:
    """
    Load (distortion label, 3x4 transform) pairs from color_profile_to_transform.json, skipping the
    identity "Reference" label. `name_filters` keeps only labels that contain one of the strings
    (case-insensitive); an empty list keeps all. Raises ValueError if nothing is left.
    """
    with open(path, encoding="utf-8") as file:
        data = json.load(file)
    profiles = []
    for label, transform in data.items():
        if label == "Reference":
            continue
        if name_filters and not any(f.lower() in label.lower() for f in name_filters):
            continue
        profiles.append((label, np.array(transform, dtype=np.float64)))
    if not profiles:
        raise ValueError(f"No profiles in {path} match the filters {name_filters}.")
    return profiles


def random_mixtures(profiles: List[Tuple[str, np.ndarray]], count: int, concentration: float,
                    rng: np.random.Generator) -> List[Tuple[str, np.ndarray]]:
    """
    Make `count` new transforms, each a random weighted average of the measured transforms, with
    weights drawn from a Dirichlet distribution. A weighted average of affine color maps is itself
    an affine color map that lies "between" the measured cards, so these cover the measured range
    without inventing unrealistic distortions.
    `concentration` is the Dirichlet parameter: 1 spreads the weight over all profiles, which pulls
    every mixture towards the average card; small values like 0.3 put most of the weight on one to
    three profiles, so the mixtures also reach the edges of the measured range.
    """
    matrices = np.stack([m for _, m in profiles])
    mixtures = []
    for i in range(count):
        weights = rng.dirichlet(np.full(len(profiles), concentration))
        mixtures.append((f"random{i:03d}", np.tensordot(weights, matrices, axes=1)))
    return mixtures


def extrapolate(matrix: np.ndarray, factor: float) -> np.ndarray:
    """
    Scale how far a transform is from the identity: 0 gives the identity, 1 gives the transform
    unchanged, 1.2 gives a distortion 20% stronger in every term (gain, offset, cross-talk).
    """
    return IDENTITY + factor * (matrix - IDENTITY)


def apply_color_transform(rgb: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 3x4 affine transform to an (H, W, 3) uint8 RGB image, rounding and clipping to uint8."""
    out = rgb.astype(np.float32) @ matrix[:, :3].T.astype(np.float32) + matrix[:, 3].astype(np.float32)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def load_extremes(path: Path) -> List[Tuple[str, ColorFunction]]:
    """
    Load color_profile_extremes.json ({label: 3x4 matrix}, written by `measure_capture_card_colors.py`)
    as (name, color function) pairs; the name is "extreme_<label>".
    """
    with open(path, encoding="utf-8") as file:
        data = json.load(file)
    return [
        (f"extreme_{label}", lambda rgb, m=np.array(matrix): apply_color_transform(rgb, m))
        for label, matrix in data.items()
    ]


def chroma_subsample(rgb: np.ndarray) -> np.ndarray:
    """
    Simulate 4:2:0 chroma subsampling, which most capture cards use over USB: convert to YCbCr,
    average Cb/Cr over 2x2 blocks, scale them back up with bilinear filtering and convert back.
    Sharp color edges (e.g. the selection cursor next to a color patch) bleed by about a pixel.
    """
    image = Image.fromarray(rgb).convert("YCbCr")
    y, cb, cr = image.split()
    size = image.size
    half = ((size[0] + 1) // 2, (size[1] + 1) // 2)
    cb = cb.resize(half, Image.BOX).resize(size, Image.BILINEAR)
    cr = cr.resize(half, Image.BOX).resize(size, Image.BILINEAR)
    return np.asarray(Image.merge("YCbCr", (y, cb, cr)).convert("RGB"))


def jpeg_round_trip(rgb: np.ndarray, quality: int) -> np.ndarray:
    """Encode and decode as JPEG at `quality` (1-95), similar to MJPEG capture cards."""
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return np.asarray(decoded.convert("RGB"))


def add_noise(rgb: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Add Gaussian noise with standard deviation `sigma` (in 0-255 units) to every channel."""
    noisy = rgb.astype(np.float32) + rng.normal(0, sigma, rgb.shape).astype(np.float32)
    return np.clip(np.rint(noisy), 0, 255).astype(np.uint8)


def simulate(rgb: np.ndarray, color_function: ColorFunction, args: argparse.Namespace, rng: np.random.Generator) -> np.ndarray:
    """Run the full pipeline (color transform, then the optional spatial stages) on an RGB image."""
    out = color_function(rgb)
    if args.chroma_subsample:
        out = chroma_subsample(out)
    if args.blur > 0:
        out = np.asarray(Image.fromarray(out).filter(ImageFilter.GaussianBlur(args.blur)))
    if args.jpeg_quality:
        out = jpeg_round_trip(out, args.jpeg_quality)
    if args.noise > 0:
        out = add_noise(out, args.noise, rng)
    return out


def safe_name(name: str) -> str:
    """Turn a profile name like "Switch1/Kuro-Win-MYPIN" into a file name part "Switch1_Kuro-Win-MYPIN"."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def write_sheet(tiles: List[Tuple[str, np.ndarray]], out_path: Path, tile_width: int = 480) -> None:
    """Save a grid of labeled, downscaled variants for a quick visual comparison."""
    columns = 4
    height, width = tiles[0][1].shape[:2]
    tile_height = round(tile_width * height / width)
    label_height = 16
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_width, rows * (tile_height + label_height)), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for i, (label, rgb) in enumerate(tiles):
        x = (i % columns) * tile_width
        y = (i // columns) * (tile_height + label_height)
        sheet.paste(Image.fromarray(rgb).resize((tile_width, tile_height), Image.BILINEAR), (x, y + label_height))
        draw.text((x + 2, y + 2), label, fill=(0, 0, 0))
    sheet.save(out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="+", type=Path, help="test images to re-color")
    parser.add_argument("--out-dir", type=Path, required=True, help="folder to write the variants into")
    parser.add_argument("--profiles", type=Path, default=DEFAULT_CALIBRATION_DIR / "color_profile_to_transform.json",
                        help="label -> transform JSON from measure_capture_card_colors.py (default: %(default)s)")
    parser.add_argument("--extremes", action="store_true",
                        help="use the extreme transforms instead of the measured profiles")
    parser.add_argument("--extremes-file", type=Path, default=DEFAULT_CALIBRATION_DIR / "color_profile_extremes.json",
                        help="extreme transforms JSON from measure_capture_card_colors.py (default: %(default)s)")
    parser.add_argument("--profile", action="append", default=[],
                        help="only use distortion labels that contain this text; can be repeated")
    parser.add_argument("--random", type=int, default=0, help="use N random mixtures of the profiles instead of the profiles themselves")
    parser.add_argument("--concentration", type=float, default=0.3,
                        help="Dirichlet concentration for --random; smaller values stay closer to single profiles (default: 0.3)")
    parser.add_argument("--extrapolate", type=float, default=1.0,
                        help="scale each distortion away from the identity, e.g. 1.2 for 20%% stronger (default: 1.0)")
    parser.add_argument("--seed", type=int, default=0, help="random seed for --random and --noise (default: 0)")
    parser.add_argument("--chroma-subsample", action="store_true", help="simulate 4:2:0 chroma subsampling")
    parser.add_argument("--blur", type=float, default=0.0, help="Gaussian blur radius in pixels (default: off)")
    parser.add_argument("--jpeg-quality", type=int, help="JPEG round trip at this quality, 1-95 (default: off)")
    parser.add_argument("--noise", type=float, default=0.0, help="Gaussian noise sigma in 0-255 units (default: off)")
    parser.add_argument("--sheet", action="store_true", help="also write <image>__sheet.png with all variants side by side")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    try:
        if args.extremes:
            if args.random > 0 or args.extrapolate != 1.0 or args.profile:
                print("Error: --extremes can't be combined with --random, --extrapolate or --profile.")
                return 1
            transforms = load_extremes(args.extremes_file)
        else:
            profiles = load_profiles(args.profiles, args.profile)
            matrices = random_mixtures(profiles, args.random, args.concentration, rng) if args.random > 0 else profiles
            transforms = [
                (name, lambda rgb, m=extrapolate(matrix, args.extrapolate): apply_color_transform(rgb, m))
                for name, matrix in matrices
            ]
    except (OSError, ValueError) as e:
        print(f"Error: {e}")
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for image_path in args.images:
        with Image.open(image_path) as image:
            alpha: Optional[Image.Image] = image.getchannel("A") if image.mode == "RGBA" else None
            rgb = np.asarray(image.convert("RGB"))
        tiles = [("original", rgb)]
        for name, color_function in transforms:
            out = simulate(rgb, color_function, args, rng)
            out_image = Image.fromarray(out)
            if alpha is not None:
                out_image.putalpha(alpha)
            out_path = args.out_dir / f"{image_path.stem}__{safe_name(name)}.png"
            out_image.save(out_path)
            tiles.append((name, out))
        print(f"{image_path}: wrote {len(transforms)} variants to {args.out_dir}")
        if args.sheet:
            sheet_path = args.out_dir / f"{image_path.stem}__sheet.png"
            write_sheet(tiles, sheet_path)
            print(f"  contact sheet: {sheet_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
