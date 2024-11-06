from importlib import import_module
from pathlib import Path
import argparse
import jax


parser = argparse.ArgumentParser(
    prog='pancax',
    description='Physics Informed Neural Network library written in jax',
    epilog='Reach out to chamel@sandia.gov for help'
)
parser.add_argument(
    '--debug-nans',
    default='off', help='Flag to debug nans. Can be either on or off'
)
parser.add_argument(
    '-i', '--input-file',
    help='Input file for pancax to run'
)
parser.add_argument(
    '-p', '--precision', 
    default='double', help='Floating point precision to use'
)
args = parser.parse_args()

# switch on debug state
if args.debug_nans == 'on':
    print('Debugging NaNs')
    jax.config.update('jax_debug_nans', True)
elif args.debug_nans == 'off':
    jax.config.update('jax_debug_nans', False)
else:
    raise ValueError('debug nans can only be on or off')

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

# switch over file types, e.g. python or yaml in the future
print(f'Running {input_file}')
# NOTE below will only work on linux/mac
if input_file.suffix == '.py':
    try:
        # with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
        import_module(str(input_file).replace('/', '.'))
    except ModuleNotFoundError:
        # things bug out currently if we use above in the bootstrapper
        print('Finished running')
else:
    raise ValueError('Only python files are supported currently!')
