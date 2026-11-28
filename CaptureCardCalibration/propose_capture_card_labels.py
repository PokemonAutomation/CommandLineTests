"""Find unlisted calibration pairs and propose labels using their measured colors."""

import argparse
import html
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

import measure_capture_card_colors as measure


SAMPLE_FRACTION = 0.4  # Match the default in measure_capture_card_colors.py.
MINIMUM_MATCH_MARGIN = 1.0  # RMS levels between the best and second-best matches.
SCREENSHOT_LABELS = {measure.REFERENCE_LABEL, "Console-Screenshot"}


def discover_pairs(folder, manifest):
    # -Up & -Down pairs
    listed = {
        Path(pair[page]).as_posix()
        for pairs in manifest.values()
        for pair in pairs
        for page in ("up", "down")
    }

    # Uses measure_capture_card_colors.py
    calibrations = []
    for console in measure.LAYOUTS:
        console_dir = folder / console
        if not console_dir.is_dir():
            continue
        pairs = {}
        for path in sorted(console_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in measure.IMAGE_EXTENSIONS:
                continue
            if path.relative_to(folder).as_posix() in listed:
                continue

            # Parse file name for key and pair "Dolphin-Win-CheapOLED-Up" -> stem: Dolphin-Win-CheapOLED, page: "Up"
            stem, _, page = path.stem.rpartition("-")
            if not stem or page not in ("Up", "Down"):
                raise ValueError(f"{path.name}: expected matching -Up and -Down filenames.")
            pair = pairs.setdefault(stem, {})
            if page in pair:
                raise ValueError(f"{console}/{stem}: more than one {page} image.")
            pair[page] = path

        # Ensure both Up and Down screenshots are found.    
        for stem, pair in pairs.items():
            for page in ("Up", "Down"):
                if page not in pair:
                    raise ValueError(f"{console}/{stem}: missing an unlisted -{page} image.")
            calibrations.append(measure.Calibration(console, pair["Up"], pair["Down"]))
    return calibrations


def sample_new_pair(calibration):
    # Validate aspect ratio
    for path in (calibration.up_path, calibration.down_path):
        with Image.open(path) as image:
            if abs(image.width / image.height - 16 / 9) > 0.01:
                raise ValueError(f"{path.name}: expected a full 16:9 frame, got {image.width}x{image.height}.")

    measure.sample_calibration(calibration, SAMPLE_FRACTION, None)
    colors = calibration.colors
    if not np.isfinite(colors).all() or not np.isfinite(calibration.stddevs).all():
        raise ValueError(f"{calibration.name}: could not sample all 60 color patches.")
    if np.linalg.matrix_rank(colors - colors.mean(axis=0)) < 3:
        raise ValueError(f"{calibration.name}: sampled patches lack the expected range of colors.")


def profile_averages(folder):
    references, distortions = measure.load_manifest(
        folder, folder / measure.MANIFEST_NAME, folder / measure.EXPLANATION_NAME
    )
    if not references:
        raise ValueError("The manifest must contain at least one reference pair.")
    profiles = {measure.REFERENCE_LABEL: references}
    for distortion in distortions:
        profiles[distortion.label] = distortion.calibrations
    averages = {}
    for label, pairs in profiles.items():
        for pair in pairs:
            measure.sample_calibration(pair, SAMPLE_FRACTION, None)
        averages[label] = np.mean([pair.colors for pair in pairs], axis=0)

    return averages


def match_profile(calibration, averages):
    """Return the matching label (or None), the closest profiles, and patch flatness."""
    matches = []
    for label, colors in averages.items():
        rms = np.sqrt(np.mean((calibration.colors - colors) ** 2))
        matches.append((rms, label))
    matches.sort()

    best_rms, label = matches[0]
    flat_patches = np.all(calibration.stddevs <= measure.FLATNESS_WARNING_STDDEV)
    clear_match = (
        label not in SCREENSHOT_LABELS
        and best_rms <= measure.SIMILAR_LABEL_RMS
        and matches[1][0] - best_rms >= MINIMUM_MATCH_MARGIN
        and flat_patches
    )
    return label if clear_match else None, matches[:3], flat_patches


def propose_labels(folder, *, dry_run):
    manifest_path = folder / measure.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    calibrations = discover_pairs(folder, manifest)
    if not calibrations:
        return ["All calibration pairs are already listed in the manifest."], False

    # Compare every new pair against the same set of previously labeled captures.
    averages = profile_averages(folder)
    summary = []
    needs_review = False
    for calibration in calibrations:
        sample_new_pair(calibration)
        label, matches, flat_patches = match_profile(calibration, averages)
        if label is not None:
            decision = "Existing label"
        else:
            needs_review = True
            decision = "Provisional label for review"
            base_label = calibration.name.replace("/", "-", 1)
            label = base_label
            suffix = 2
            while label in manifest:
                label = f"{base_label}-{suffix}"
                suffix += 1

        manifest.setdefault(label, []).append({
            "up": calibration.up_path.relative_to(folder).as_posix(),
            "down": calibration.down_path.relative_to(folder).as_posix(),
        })
        candidates = ", ".join(
            f"<code>{html.escape(profile)}</code> ({rms:.2f} RMS)" for rms, profile in matches
        )
        summary.append(f"- <code>{html.escape(calibration.name)}</code> -> <code>{html.escape(label)}</code>. "
                       f"{decision}. Closest profiles: {candidates}.")
        if not flat_patches:
            summary.append("  Some sampled patches are uneven. Check the screenshot framing and swatch report.")

    if not dry_run:
        write_manifest(manifest_path, manifest)
    return summary, needs_review


def write_manifest(path, manifest):
    profiles = []
    for label, pairs in manifest.items():
        entries = ",\n".join(f"    {json.dumps(pair)}" for pair in pairs)
        profiles.append(f"  {json.dumps(label)}: [\n{entries}\n  ]")
    path.write_text("{\n" + ",\n".join(profiles) + "\n}\n", encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, nargs="?", default=measure.default_calibration_dir())
    parser.add_argument("--dry-run", action="store_true", help="validate and preview without writing the manifest")
    parser.add_argument("--summary", type=Path, help="write a Markdown report of the proposed labels")
    args = parser.parse_args()

    try:
        summary, needs_review = propose_labels(args.folder, dry_run=args.dry_run)
    except (OSError, ValueError, KeyError, Image.DecompressionBombError) as error:
        parser.exit(1, f"Calibration error: {error}\n")

    report = "### Proposed calibration labels\n\n" + "\n".join(summary) + "\n"
    if args.summary:
        args.summary.write_text(report, encoding="utf-8")
    else:
        print(report, end="")

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as output:
            print(f"needs_review={str(needs_review).lower()}", file=output)


if __name__ == "__main__":
    main()
