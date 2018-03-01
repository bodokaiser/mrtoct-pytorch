import argparse


def main(args):
  pass


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('inputs-path')
  parser.add_argument('targets-path')

  main(parser.parse_args())
