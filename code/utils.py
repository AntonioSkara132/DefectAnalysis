import glob

def get_file_paths(directory_path: 'str') -> list:
    """Return a list of paths matching a pathname pattern."""
    return glob.glob(directory_path)

def addFrequencies(freqs, list):
    """Counts index values"""
    for i in list:
        freqs[i] += 1
    return freqs

