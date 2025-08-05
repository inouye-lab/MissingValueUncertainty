import argparse
import logging
import os.path

import pandas as pd

from mvu.logger import setupLogging

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("name", type=str, nargs='*', help='List of files to combine')
    parser.add_argument("--output", type=str, help='Name for output CSV')

    parser.add_argument('-v', '--verbose', type=int, nargs='?', default=1, help='Logging verbosity level')

    args = parser.parse_args()
    # start logging
    folder = os.path.dirname(args.output)
    date = setupLogging(args.verbose, os.path.join(folder, "log"), args=args)

    # load in data frames
    logging.info(f"Loading in {args.name} data frames from {folder}")
    dataframes = [pd.read_csv(name) for name in args.name]

    # merge into 1 dataframe
    logging.info("Combining dataframes")
    combinedInput = pd.concat(dataframes, axis=0, ignore_index=True)

    # average over masks if needed
    logging.info("Averaging over masks")
    perMethodScale = (combinedInput.groupby(['Method', 'Scale'])['MVCE Mean']
                      .mean().reset_index())

    # find best scale for the method
    logging.info("Finding best method")
    bestScale = (perMethodScale.loc[perMethodScale.groupby('Method')['MVCE Mean'].idxmin()]
                 .reset_index(drop=True)
                 .sort_values('Method'))

    # save results to CSV
    outputPath = f"{args.output}.csv"
    perMethodPath = f"{args.output}-all-scales.csv"
    logging.info(f"Saving results to {outputPath} and {perMethodPath}")
    bestScale.to_csv(outputPath, index=False)
    perMethodScale.to_csv(perMethodPath, index=False)
