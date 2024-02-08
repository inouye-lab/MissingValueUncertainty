import argparse
import json

from mvu.dataset.loader import getDatasetSplits

if __name__ == '__main__':
    """
    This is a quick and simple script to download the starcraft data for a given machine.
    Presently, we automatically download the dataset if the cache is not detected, however this is a time intensive
    process so its convenient to do during downtime instead of before a planned experiment.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str, help='Name of the dataset to parse')
    parser.add_argument("dataset", type=json.loads, default=dict(), help='Parameters to load the dataset')
    args = parser.parse_args()

    # just running this file will ensure its downloaded
    getDatasetSplits(args.name, **args.dataset)
    print(f"Done loading {args.name}")
