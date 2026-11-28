# Contributing

Use the SerialPrograms screenshot button for test and calibration images. Put test
images in the folder for the relevant detector and give them descriptive filenames.

## First contribution or new capture setup

Using the same setup as your test images, capture two full-frame screenshots of
the Switch user icon's **Edit Background** color picker. Take one at the top of
the palette (`-Up.png`) and one at the bottom (`-Down.png`). Add the pair to
[Switch1](CaptureCardCalibration/Switch1) or [Switch2](CaptureCardCalibration/Switch2).

| Switch 2 at the top of the palette                                                                                                       | Switch 2 at the bottom of the palette                                                                                                         |
| ---------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| <img src="CaptureCardCalibration/Switch2/Dolphin-Win-HD60Pro-Up.png" width="320" alt="Edit Background color picker scrolled to the top"> | <img src="CaptureCardCalibration/Switch2/Dolphin-Win-HD60Pro-Down.png" width="320" alt="Edit Background color picker scrolled to the bottom"> |

Use the same filename stem for both screenshots. In your pull request, include
the capture card make and model, your operating system (OS), and the Switch model.
After the screenshots merge, GitHub Actions opens or updates a follow-up PR with
the proposed labels and transforms for maintainers to review. See the
[calibration guide](CaptureCardCalibration/README.md) for the review steps.

## Returning contributors with the same setup

Add each new image path and its existing capture profile label to
[`test_image_labels.json`](CaptureCardCalibration/test_image_labels.json), or identify
your setup in the pull request so maintainers can label the images. If you used
multiple setups, say which images came from each.
