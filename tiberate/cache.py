import glob
import os

CACHE_FOLDER = os.path.abspath(__file__).replace("cache.py", "resources")


def clean_cache(path=None):
    if path is None:
        path = CACHE_FOLDER
    files = glob.glob(os.path.join(path, "*.pkl"))
    for file in files:
        try:
            os.unlink(file)
        except Exception as e:
            print(e)
            pass


def generate_cache(path=None):
    if path is None:
        path = CACHE_FOLDER
    # Read in pre-calculated high-quality primes.
    from tiberate.prim import generate_primes

    _ = generate_primes.generate_message_primes(cache_folder=path)
    _ = generate_primes.generate_scale_primes(cache_folder=path)
