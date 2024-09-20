This directory contains various bash scripts used in running our experiments.

* `setup/` contains scripts to setup the environment for various datasets or models.
* `training/` contains scripts for training new models used in our experiments.
* `generator/` contains scripts for generating samples from expensive models before running the main missing value experiments.
* `/experiment/` contains scripts that run our final experiments.

All of these scripts are designed to be run from the root directory via `bash scripts/<directory>/<script>.sh`.