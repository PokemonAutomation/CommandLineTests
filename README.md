# CommandLineTests

Screenshot fixtures (test images) for the unit tests in
[Arduino-Source](https://github.com/PokemonAutomation/Arduino-Source).

This repository contains **images only, no code**. The test cases that load
these images live in the Arduino-Source repository, next to each detector or
reader they exercise.

## Layout

Top-level folders are grouped by game or platform, and the next level down is
the detector/reader class being tested:

```
CommandLineTests/
├── CommonFramework/      # framework-level tests (e.g. BlackBorderDetector)
├── NintendoSwitch/       # console-level screens (check online, update menu, ...)
├── OCR/                  # individual glyphs/sentences for the OCR engine
├── PokemonFRLG/          # FRLG (GBA) detector/reader tests
├── PokemonHome/          # Pokémon HOME
├── PokemonLA/            # Legends Arceus
├── PokemonLZA/           # Legends Z-A
├── PokemonSV/            # Scarlet & Violet
└── PokemonSwSh/          # Sword & Shield
```

Some detectors have subfolders per capture device or video mode (for example
`macOS_bright`, `WinElgato`, `WinMyPin`, `WinShadowCast`, `WinPowcxy`,
`WinHD60S`, `WinMirabox`, `Switch`) because images look different depending on
the capture setup. FRLG reader tests also include a language code in the file
name (`eng`, `fra`, `jpn`, `deu`, `spa`, `ita`).

## Naming conventions

- The expected result is **not** parsed from the file name — each test case
  receives it when it is registered (e.g.
  `database.add<Test_BlackOutDetector>(".../BlackOut1_True.png", true)`).
  Older fixtures sometimes encode it in the name
  (`BlackOut1_True.png`, `Icelands_0.png`,
  `eng_snorunt_NotShiny_NotAlpha_Female.png`) because the test code used to
  rely on that for simplicity. It is no longer required and there is no
  readability benefit — for new screenshots just use whatever is natural for
  the file name.
- **Readers** that produce several values from one image use a *golden file*:
  a `.txt` file in the same folder named with a leading underscore and the
  same stem as the image, containing one expected value per line, in the same
  order as the labels passed to the test, e.g.
  `bulbasaur_1_eng.png` is paired with `_bulbasaur_1_eng.txt`
  (see `check_against_golden_file()` in `PokemonFRLG/PokemonFRLG_Tests.cpp`).
- Files are usually `.png`, some are `.jpg`/`.jpeg` (older captures).

## How the tests find these files

When a test case is constructed, its image path is loaded relative to
`UNIT_TEST_RESOURCE_PATH()`. The program looks for a folder named
`UnitTestResources/` — or, falling back, the legacy name `CommandLineTests/` —
next to the program, searching up to 5 directory levels up
(`SerialPrograms/Source/CommonFramework/GlobalAutoPaths.cpp`).

So to run the tests, clone this repository next to your SerialPrograms
installation, either as:

```
MyInstall/
├── SerialPrograms/...
└── CommandLineTests/     # this repo (legacy name, still works)
```

or rename the checkout to `UnitTestResources/` (preferred).

## Running the tests

- **Command line:** launch the program with the `--command-line-test-mode`
  argument, or set
  `"20-GlobalSettings": {"COMMAND_LINE_TESTS": "RUN"}` to `true` in
  `SerialPrograms-Settings.json`. This runs every registered test case in
  parallel (bounded by memory and thread limits) and returns a non-zero exit
  code if anything fails.
- **GUI:** the "Unit Test Runner" program can run a single test, the tests
  matching a substring, or all tests.

Each test case is a class inheriting `UnitTest`
(`Common/Cpp/TestRunners/UnitTest.h`) that implements `run()` and returns a
`UnitTestResult` (passed, failed with a message, skipped, or out of memory).
Modules register their cases in an `add_tests(UnitTestDatabase&)` function,
for example:

```cpp
database.add<Test_SummaryReader_Numbers>("PokemonHome/SummaryScreen/squirtle_Shiny.png", 7, 700052, 1);
database.add<Test_StatsReaderPage1>("PokemonFRLG/StatsReader/Page1/bulbasaur_1_eng.png");
```

`ComputerPrograms/UnitTestRunner.cpp` collects all of them.

## Adding a new test case

1. Take a screenshot of the relevant screen (note which capture device and
   video mode it came from, and put it in the matching subfolder if one
   exists).
2. Save it in `Game/DetectorName/` in this repository — use whatever is
   natural for the screenshot file name, no need to encode the expected
   result in it — or, for a reader, also add the matching `_<stem>.txt`
   golden file.
3. Register the case in the `add_tests_*` function in the corresponding source
   file in Arduino-Source (usually the detector/reader `.cpp` itself).
4. Run the tests (command-line mode or the Unit Test Runner) to confirm the new
   case passes.
