import argparse
import logging
import os.path

import pandas as pd

from mvu.logger import setupLogging

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("name", type=str, nargs='*', help='List of files to combine')
    parser.add_argument("--output", type=str, help='Name for output CSV')
    parser.add_argument("--type", type=str, default='legacy',
                        help="Calibration type, can be 'soft', 'hard', or 'legacy")

    parser.add_argument('-v', '--verbose', type=int, nargs='?', default=1, help='Logging verbosity level')

    args = parser.parse_args()
    # start logging
    folder = os.path.dirname(args.output)
    date = setupLogging(args.verbose, os.path.join(folder, "log"), args=args)

    # select MVCE key, allows us to choose soft or hard consistency for calibration
    mvceKey: str
    if args.type == 'soft':
        mvceKey = 'Soft MVCE Mean'
    elif args.type == 'hard':
        mvceKey = 'Hard MVCE Mean'
    elif args.type == 'legacy':
        mvceKey = 'MVCE Mean'
    else:
        raise ValueError(f"Unknown calibration type: {args.type}")
    logging.info(f"Calibrating using {args.type} using key {mvceKey}")

    # load in data frames
    logging.info(f"Loading in {args.name} data frames from {folder}")
    dataframes = [pd.read_csv(name) for name in args.name]

    # ensure all dataframes contain the relevant key
    for i, df in enumerate(dataframes):
        if not mvceKey in df.columns:
            raise ValueError(f"Missing {mvceKey} in dataframe {i}: {args.name[i]}")

    # merge into 1 dataframe
    logging.info("Combining dataframes")
    combinedInput = pd.concat(dataframes, axis=0, ignore_index=True)

    # average over masks if needed
    logging.info("Averaging over masks")
    perMethodScale = (combinedInput.groupby(['Method', 'Scale'])[mvceKey]
                      .mean().reset_index())

    # find best scale for the method
    logging.info("Finding best method")
    bestScale = (perMethodScale.loc[perMethodScale.groupby('Method')[mvceKey].idxmin()]
                 .reset_index(drop=True)
                 .sort_values('Method'))

    # save results to CSV
    outputPath = f"{args.output}.csv"
    perMethodPath = f"{args.output}-all-scales.csv"
    logging.info(f"Saving results to {outputPath} and {perMethodPath}")
    bestScale.to_csv(outputPath, index=False)
    perMethodScale.to_csv(perMethodPath, index=False)
