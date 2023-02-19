import os
import sys
import argparse

utils_path = os.path.join(os.path.dirname(__file__),'utils')
sys.path.append(utils_path)

from fetch_database import DataSet



def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", dest="url", type=str, help="url to kaagle dataset")
    parser.add_argument("--data_dir", dest="data_dir", type=str, help= "dataset destination directory path")
    args = parser.parse_args()
    dataset = DataSet(url=args.url, data_dir=args.data_dir)
    dataset.fech_data()


run()
