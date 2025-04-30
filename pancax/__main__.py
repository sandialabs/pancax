from importlib import import_module
from pathlib import Path
import argparse
import jax
import os
import sys


code_name = """
██████╗  █████╗ ███╗   ██╗ ██████╗ █████╗ ██╗  ██╗
██╔══██╗██╔══██╗████╗  ██║██╔════╝██╔══██╗╚██╗██╔╝
██████╔╝███████║██╔██╗ ██║██║     ███████║ ╚███╔╝
██╔═══╝ ██╔══██║██║╚██╗██║██║     ██╔══██║ ██╔██╗
██║     ██║  ██║██║ ╚████║╚██████╗██║  ██║██╔╝ ██╗
╚═╝     ╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝

Developed by Craig M. Hamel

MIT License

Copyright (c) 2024 Sandia National Laboratories

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""


parser = argparse.ArgumentParser(
    # prog='pancax',
    prog=code_name,
    description='Physics Informed Neural Network library written in jax',
    epilog='Reach out to chamel@sandia.gov for help'
)
parser.add_argument(
    '-d', '--debug-nans',
    # default='off',
    action='store_true',
    default=False,
    help='Flag to debug nans. Options are on or off.'
)
parser.add_argument(
    '-i', '--input-file',
    help='Input file for pancax to run.'
)
parser.add_argument(
    '-l', '--log-file',
    default=os.path.join(os.getcwd(), 'pancax.log'),
    help='Log file for pancax to write stdout and stderr to.'
)
parser.add_argument(
    '-p', '--precision',
    default='double',
    help='Floating point precision to use. Options are single or double.'
)
parser.add_argument(
    '-v', '--verbose',
    action='store_true',
    default=False,
    help='Whether or not to print to console.'
)
args = parser.parse_args()

print(code_name)

# switch on debug state
# if args.debug_nans == 'on':
if args.debug_nans:
    print('Debugging NaNs')
    jax.config.update('jax_debug_nans', True)
else:
    jax.config.update('jax_debug_nans', False)

# switch on different precision levels
if args.precision == 'double':
    print('Using double precision')
    jax.config.update("jax_enable_x64", True)
elif args.precision == 'single':
    print('Using single precision')
else:
    raise ValueError('Precision needs to be \'single\' or \'double\'.')

# read in the input file and make sure it exists
input_file = Path(args.input_file)
if not input_file.is_file():
    raise FileNotFoundError(f'Input file {input_file} does not exist.')

log_file = Path(args.log_file)

# switch over file types, e.g. python or yaml in the future
print(f'Running {input_file}')
print(f'Writing output to {log_file}')
# NOTE below will only work on linux/mac

with open(log_file, 'w') as log:
    # direct output to log
    if not args.verbose:
        sys.stdout = log
        sys.stderr = log
    print(code_name, flush=True)
    # read input file and run it
    if input_file.suffix == '.py':
        try:
            # with jax.profiler.trace("/tmp/jax-trace",
            # create_perfetto_link=True):
            import_module(str(input_file).replace('/', '.'))
        except ModuleNotFoundError:
            # things bug out currently if we use above in the bootstrapper
            print('Finished running')
    else:
        raise ValueError('Only python files are supported currently!')

    # Reset stdout and stderr to their original values
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
