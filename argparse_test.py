import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-l','--list',nargs="+",type=str)
    args = parser.parse_args()

    for key in args.list:
        print(key)
