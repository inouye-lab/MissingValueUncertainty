# This is a simple script to migrate saved binaries after a class rename.
# It's not a common problem but has been useful on several occasions so figured its worth keeping around

import argparse
import gzip
import pickle
import sys

import mvu.model.regressor

if __name__ == '__main__':
    # if an entire package gets renamed, you can use sys.modules to redirect the old package to the new one
    # for class renames, simply ensure the old import is valid, e.g. using an assignment statement
    sys.modules['mvu.regressor'] = mvu.model.regressor

    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str, help='Name of the file to migrate')
    args = parser.parse_args()

    # assuming the system modules and imports are setup correctly, this reading should be valid
    # if so, Python will automatically redirect any class references to their current location
    with gzip.open(args.name, 'rb') as f:
        data = pickle.load(f)
    # saving the object again will save it using the new class references
    # after doing this, you can freely remove the now invalid imports
    with gzip.open(args.name, 'wb') as f:
        pickle.dump(data, f)
