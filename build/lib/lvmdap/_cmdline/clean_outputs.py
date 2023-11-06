
import sys, os
import time
import numpy as np
import argparse
from copy import deepcopy as copy
from pprint import pprint

from pyFIT3D.common.io import remove_previous


CWD = os.path.abspath(".")

def _no_traceback(type, value, traceback):
  print(value)

def _main(cmd_args=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description="Removes the previous runs for the given labels"
    )
    parser.add_argument(
        "labels", metavar="label", nargs="*",
        help="an arbitrary number of labels to remove from previous runs of the lvm-dap script"
    )
    parser.add_argument(
        "-o", "--output-path", metavar="path",
        help=f"path to the outputs. Defaults to '{CWD}'",
        default=CWD
    )
    parser.add_argument(
        "-v", "--verbose",
        help="if given, shows information about the progress of the script. Defaults to false.",
        action="store_true"
    )
    parser.add_argument(
        "-d", "--debug",
        help="debugging mode. Defaults to false.",
        action="store_true"
    )
    args = parser.parse_args(cmd_args)

    if not args.debug:
        sys.excepthook = _no_traceback
    else:
        pprint("COMMAND LINE ARGUMENTS")
        pprint(f"{args}\n")

    for label in args.labels:
        remove_previous(
            os.path.join(args.output_path, label),
            os.path.join(args.output_path, f"elines_{label}"),
            os.path.join(args.output_path, f"single_{label}"),
            os.path.join(args.output_path, f"coeffs_{label}"),
            os.path.join(args.output_path, f"output.{label}.fits.gz")
        )
