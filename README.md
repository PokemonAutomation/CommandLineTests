# CommandLineTests

Unit test files for the codebase [Arduino-Source](https://github.com/PokemonAutomation/Arduino-Source)
and the Pokémon automation PC program `SerialPrograms` built by it.

## Layout

Top-level folders are grouped by game or library folders in the codebase, and the next level down is
the various detector/reader classes being tested:

```
CommandLineTests/
├── CommonFramework/      # framework-level tests (e.g. BlackBorderDetector)
├── NintendoSwitch/       # console-level screens (check online, update menu, ...)
├── OCR/                  # individual glyphs/sentences for the OCR engine
├── PokemonSwSh/          # Pokémon Sword & Shield visual detectors
├── PokemonHome/          # Pokémon HOME visual detectors
└── ...                   # Many other games
```

## Naming conventions

We used to put expected test results into image filenames for easy test addition.
We've now moved on to adding them into the codebase. So pick a natural filename
that describes the content of your screenshot. It is technically fine
to save with a name like `20261025-200328573913.png` but we prefer sth more meaningful like
`AnshaDonutMenuEng-20261025-200328.png`.


## How the tests find these files

In [Arduino-Source](https://github.com/PokemonAutomation/Arduino-Source),
the test framework first looks for the test root folder (aka this repo) named
`UnitTestResources/` (or the legacy name `CommandLineTests/`) next to the program.
If not found, it can search up to 5 directories relative to where the program runs.
See `SerialPrograms/Source/CommonFramework/GlobalAutoPaths.cpp`.

Once the test root folder is found, each unit test containing the image paths
relative to the test root folder is executed.

So to run the tests on your end, you can clone this repository next to your
SerialPrograms installation:
```
MyGithubRepos/
├── Arduino-Source/...
└── CommandLineTests/     # this repo
```

## Running the tests

- **Command line:** launch the program with the `--command-line-test-mode`
  argument, or set
  `"20-GlobalSettings": {"COMMAND_LINE_TESTS": "RUN"}` to `true` in
  `SerialPrograms-Settings.json`. This runs every registered test case in
  parallel (bounded by memory and thread limits) and returns a non-zero exit
  code if anything fails.
- **GUI:** the "Unit Test Runner" program can run a single test, the tests
  matching a substring, or all tests.

## Adding a new test case

To take a screenshot of the Switch, use the screenshot button on the `SerialPrograms` GUI.
Don't use your phone to take a photo or use your PC to take a screenshot. The unit tests
need the exact image size and color distortion as the program sees it during automation.

Save your screenshot in the appropriate folder (usually `Game/DetectorName/`) and rename it
to not collide with other files while describing what it shows.

Then write your unit test cases in [Arduino-Source](https://github.com/PokemonAutomation/Arduino-Source).

### Capture card and Switch video distortion calibration

If you haven't done so, provide a pair of calibration screenshots to record the color distortion
from your capture card + Switch video setting:

Following the other developers' calibration images at `CaptureCardCalibration/Switch1/` or
`CaptureCardCalibration/Switch2/`, use the screenshot button to capture a pair of your
edit-icon screens and save them there.

Give a name of your color setting as your color distortion label and save it inside
`CaptureCardCalibration/color_profile_labels.json`. Follow the format of the existing labels
in the JSON.

Then run
```
cd CaptureCardCalibration/
python3 measure_capture_card_colors.py
```
You may need to install some Python libraries to run the script above.

The script will measure how much distortion your calibration images have comparing to
the reference images.

If the script warns you that your label's distortion transform is too close to another
label, your color distortion is already present, you can then revert your changes in
`CaptureCardCalibration/` and remember the distortion label name of your setting.
 
Place your new screenshot file path and its distortion label in 
`CaptureCardCalibration/test_image_labels.json` so that the test framework can
apply the correct color distortion simulation on it to widen the image test coverage,
making our visual detectors more robust against color distortions.