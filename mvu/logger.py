import json
import logging
import os
import sys
from argparse import Namespace
from datetime import datetime
from typing import Optional


def setupLogging(verbosity: int, outputDir: str = None, outputName: str = None) -> None:
    """
    Sets up standard script logging
    :param verbosity:  1 for info, 2 for debug
    :param outputDir:  Folder to save log files, if none does not save
    :param outputName: Name of the log file, used as a prefix for the date
    """

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)-5.5s] [%(filename)s:%(funcName)s:%(lineno)d]\t %(message)s"
    )
    root = logging.getLogger()

    # always log to console
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(formatter)
    root.addHandler(consoleHandler)

    # optionally log to file
    if outputDir is not None:
        os.makedirs(outputDir, exist_ok=True)
        date = datetime.now().strftime("%Y%m%d-%H%M%S")
        if outputName is not None:
            date = f"{outputName}-{date}"
        fileHandler = logging.FileHandler(os.path.join(outputDir, f"{date}.log"))
        fileHandler.setFormatter(formatter)
        root.addHandler(fileHandler)

    if verbosity == 1:
        root.setLevel(logging.INFO)
    elif verbosity == 2:
        root.setLevel(logging.DEBUG)


def dumpArgs(args: Namespace, path: str = None) -> None:
    """
    Dumps the argument to the script to both the log and to a JSON file at the specified path
    :param args: Arguments object
    :param path: Path to dump the JSON file
    """
    jsonArgs = json.dumps(vars(args))
    logging.info("Starting with arguments\n%s", jsonArgs)
    if path is not None:
        with open(path, 'w') as f:
            f.write(jsonArgs)
