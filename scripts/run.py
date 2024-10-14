from helper import process_dataset
from hypertuning import create_hypermodel
from stats import create_statistics
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='filename', required=True,
                        help='name of config file')
    args = parser.parse_args()

    process_dataset(config_filename=args.config)
    for i in range(10):
        create_hypermodel(config_filename=args.config)
        create_statistics(i, config_filename=args.config)
