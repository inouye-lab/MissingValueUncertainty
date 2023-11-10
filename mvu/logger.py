import json
import logging
import os
import sys
from argparse import Namespace
from datetime import datetime
from types import TracebackType
from typing import Type, Optional


def handleException(excType: Type[BaseException], excValue: BaseException, excTraceback: Optional[TracebackType],
                    message: str = "Uncaught Exception") -> None:
    """Function passed to python internals to handle exceptions by the logger"""
    if issubclass(excType, KeyboardInterrupt):
        sys.__excepthook__(excType, excValue, excTraceback)
    else:
        logging.critical(message, exc_info=(excType, excValue, excTraceback))


def setupLogging(verbosity: int, outputDir: str = None, outputName: str = None, args: Namespace = None) -> str:
    """
    Sets up standard script logging
    :param verbosity:  1 for info, 2 for debug
    :param outputDir:  Folder to save log files, if none does not save
    :param outputName: Name of the log file, used as a prefix for the date
    :param args: If defined, dumps the arguments next to the namespace
    :return:  Date used for output files.
    """

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)-5.5s] [%(filename)s:%(funcName)s:%(lineno)d]\t %(message)s"
    )
    root = logging.getLogger()

    # always log to console
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(formatter)
    root.addHandler(consoleHandler)

    if verbosity == 1:
        root.setLevel(logging.INFO)
    elif verbosity == 2:
        root.setLevel(logging.DEBUG)

    # add exception handler so those get logged
    sys.excepthook = handleException
    logging.captureWarnings(True)

    # optionally log to file
    date = datetime.now().strftime("%Y%m%d-%H%M%S")
    if outputDir is not None:
        os.makedirs(outputDir, exist_ok=True)
        name = date
        if outputName is not None:
            name = f"{outputName}-{date}"
        fileHandler = logging.FileHandler(os.path.join(outputDir, f"{name}.log"))
        fileHandler.setFormatter(formatter)
        root.addHandler(fileHandler)

        if args is not None:
            dumpArgs(args, os.path.join(outputDir, f"{name}-args.json"))
    return date


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
