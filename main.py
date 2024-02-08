import argparse
import os
from source.run_experiment import run_experiment


def main():
    parser = argparse.ArgumentParser(description='Run the experiment')
    parser.add_argument('--dataset_name', type=str, default="boston", help='The name of the dataset')
    parser.add_argument('--loss', type=str, default="QR", help='The name ofthe loss function')
    parser.add_argument('--coverage', type=float, default=0.9, help='The targeted coverage level')
    parser.add_argument('--random_seed', type=int, default=42, help='The random seed')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='The test ratio')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='The validation ratio')
    parser.add_argument('--dropout', type=float, default=0.1, help='The dropout')
    parser.add_argument('--epochs', type=int, default=1000, help='The number of epochs')
    parser.add_argument('--lr', type=float, default=0.0005, help='The learning rate')
    parser.add_argument('--penalty', type=float, default=0., help='The penalty')
    parser.add_argument('--scheduler1', type=float, default=1, help='The scheduler1')
    parser.add_argument('--scheduler2', type=float, default=1, help='The scheduler2')
    parser.add_argument('--batch_size', type=int, default=64, help='The batch size')
    args = parser.parse_args()
    
    config = vars(args)
    
    run_experiment(config)
    
if __name__ == "__main__":
    main()