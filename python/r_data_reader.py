import pyreadr
import pandas as pd
import argparse
import sys
from pathlib import Path

class DataLoader:

    def __init__(self, filename):
        self.filename = filename
        self.path = str((Path().cwd() / 'data' / self.filename).resolve())
        print(self.path)
    
    def read_file(self):
        results = pyreadr.read_r(self.path)
        print(results.keys())
        df = results[""]
        print(df)


        



def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--filename", help="The file in ./data to load", default=None
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    data_loader = DataLoader(args.filename)
    try:
        data_loader.read_file()
    except Exception as e:
        print(e)
        sys.exit(1)
        
