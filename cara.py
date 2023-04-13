import numpy as np
import numpy.random as npr
import pandas as pd
import scipy
import caffeine
from tqdm import tqdm

import math
import argparse
import sys
import subprocess
from subprocess import STDOUT, PIPE
import os.path
from os.path import isfile
import itertools
import time

from cara.auction import Auction
from cara.auction_parser import Parser

import cara


argparser = argparse.ArgumentParser()
argparser.add_argument('input_files', nargs='+', help='Input files')
argparser.add_argument('-f', '--full', action="store_true", default=False, help='Write full results and support.')
argparser.add_argument('-c', '--caffeine', action="store_true", default=False, help='Stops system from sleeping.')
argparser.add_argument('-m', '--more', action="store_true", default=False, help='Write intermediate representation.')
argparser.add_argument('-t', '--timing', action="store_true", default=False, help='Write timing data.')
argparser.add_argument('-o', '--output', choices=['a', 'w'], required=False, help='SOMETHING')
args = argparser.parse_args()


print('\nAll input files are:\n' + '\n'.join(args.input_files))

npr.seed(0)

already_exists = False
for input_file in args.input_files:
    output_file = input_file[:-4] + '-res'
    if isfile(output_file+'.csv') and args.full and not args.output:
        print(f"WARNING: {output_file}.csv exists.")
        already_exists = True
    if isfile(output_file+'-spa.csv') and args.full and not args.output:
        print(f"WARNING: {output_file}-spa.csv exists.")
        already_exists = True
    if isfile(output_file+'-int.txt') and args.more and not args.output:
        print(f"WARNING: {output_file}-int.txt exists.")
        already_exists = True
    if isfile(output_file+'-time-pre.csv') and args.timing and not args.output:
        print(f"WARNING: {output_file}-time-pre.csv exists.")
        already_exists = True
    if isfile(output_file+'-time-gen.csv') and args.timing and not args.output:
        print(f"WARNING: {output_file}-time-gen.csv exists.")
        already_exists = True
if already_exists:
    print("Set -o [--output] to 'a' (append) or 'w' (write) to silence warning.")
    sys.exit()

if args.caffeine:
    caffeine.on(display=False)
else: 
    caffeine.off()

for i, input_file in enumerate(args.input_files):
    print(f'\n{input_file} results:')

    output_file = input_file[:-4] + '-res'

    with open(input_file, "r") as f:
        input = f.read()
    parsed = Parser(input)
    auction = Auction(parsed)
    auction.run(args, output_file)

print()

