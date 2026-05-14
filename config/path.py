import os

# Determine the root directory (parent of this config file)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA = os.path.join(ROOT)

RESULTS = os.path.join(ROOT, 'results')

# Create results directory if it doesn't exist
if not os.path.exists(RESULTS):
    os.makedirs(RESULTS)


def get_data_file(filename):
    return os.path.join(DATA, filename)
