This directory contains various bash scripts used in running our experiments.

* `setup/` contains scripts to setup the environment for various datasets or models.
* `training/` contains scripts for training new models used in our experiments.
* `ece/` contains scrips for evaluating ECE and accuracy on each of our main datasets; for validating classifiers.
* `generator/` contains scripts for generating samples from expensive models before running the main missing value experiments.
* `mvce/` contains scripts for evaluating MVCE across an entire dataset, with an optional calibration mapping.
* `calibration/` contains scripts to calibrate each dataset.
* `experiment/` contains scripts related to some older experiments.

All of these scripts are designed to be run from the root directory via `bash scripts/<directory>/<script>.sh`.