"""
Main module.
Usage: python3 main.py <input USB dump file (BIN)> <output file name> (<network name>)
"""

import logging
import os
import sys
from argparse import ArgumentParser

from interpret_blob import interpret_blob


def main():
    """
    Main method.
    Processes command line arguments and performs basic validation.
    Then, it reads the input file and hands over execution to the BLOB interpreter.
    """

    parser = ArgumentParser(description='Process NCS2 USB BLOB files.')

    parser.add_argument(
        '-i', '--in',
        metavar='in',
        dest='infile',
        type=str,
        help='Path to input file.',
        required=True)

    parser.add_argument(
        '-o', '--out',
        metavar='out',
        dest='outfile',
        type=str,
        help='Name of output files (without extension).',
        required=True)

    parser.add_argument(
        '-n', '--name',
        metavar='name',
        dest='netname',
        type=str,
        help='Name of resulting network (default: reverse_net).',
        default='reverse_net',
        required=False)

    parser.add_argument(
        '-l', '--logging',
        metavar='ERROR/WARN/INFO/DEBUG',
        dest='loglevel',
        help='Sets loglevel.',
        choices=('ERROR', 'WARN', 'INFO', 'DEBUG'),
        default='INFO',
        required=False
    )

    parser.add_argument(
        '-f', '--force-overwrite',
        action='store_true',
        dest='force_overwrite',
        help='Do not ask to overwrite files. Use with caution!',
        required=False
    )

    parser.add_argument(
        '--interactive',
        action='store_true',
        dest='interactive',
        help='Decode file interactively. Enables stepping through the decoding process.',
        required=False)

    parser.add_argument(
        '--mnist',
        action='store_true',
        dest='mnist',
        help='Enables special MNIST mode needed for that model. Only use for the MNIST model.',
        required=False)

    args = parser.parse_args()

    logging.getLogger('root').setLevel(level=args.loglevel)


    raw_data = None
    if os.path.isfile(args.infile):
        with open(args.infile, 'rb') as bin_file:
            raw_data = bin_file.read()
    else:
        print(f"Input file ({args.infile} does not exist!")
        print("Quitting.")
        sys.exit(1)

    if os.path.isfile(args.outfile + '.xml') and not args.force_overwrite:
        choice = input(f"Output file ({args.outfile}.xml) already exists."
                        "Do you want to overwrite it? [y/N] ")
        choice = choice.lower()
        if choice not in ('y', 'yes'):
            print("Quitting.")
            sys.exit(0)

    if os.path.isfile(args.outfile + '.bin') and not args.force_overwrite:
        choice = input(f"Output file ({args.outfile}.bin) already exists."
                        "Do you want to overwrite it? [y/N] ")
        choice = choice.lower()
        if choice not in ('y', 'yes'):
            print("Quitting.")
            sys.exit(0)

    net_name = args.netname if args.netname else 'revnet'
    if not (net_name[0].isalpha() and net_name.isprintable()):
        print(f"Invalid network name: {net_name}")
        print("Only printable characters are allowed.")
        print("Additionally, the first character most be alphabetic.")
        print("Quitting.")
        sys.exit(1)


    verbose = args.loglevel == 'DEBUG'

    interpret_blob(
        data=raw_data,
        verbose=verbose,
        outfile=args.outfile,
        net_name=net_name,
        interactive=args.interactive,
        mnist=args.mnist)

if __name__ == '__main__':
    main()
