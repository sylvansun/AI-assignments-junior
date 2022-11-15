import argparse


def make_parser():
    parser = argparse.ArgumentParser(description="Task2 Configurations")

    parser.add_argument("--bs", type=int, default=64, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--wd", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--ne", type=int, default=200, help="number of epochs")
    parser.add_argument("--debug", action="store_true", help="run in debug mode")


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    print(args)
