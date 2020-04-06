import argparse

def main(args):
    print("path:{0}".format(args.path))
    print("verbose: {0}".format(args.v))

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='./', help = 'what is the path of the file')
parser.add_argument('-v', type=bool, default=False, help='verbose')
args = parser.parse_args()
main(args)