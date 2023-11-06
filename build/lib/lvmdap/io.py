
import gzip
import pandas as pd

def read_iso(filename):
    with gzip.open(filename, "r") as f:
        line = f.readline().decode("utf-8")
        while line.startswith("#"):
            hdr = line.replace("#","").split()
            line = f.readline().decode("utf-8")
    df = pd.read_csv(filename, sep="\s+", header=None, skiprows=31)
    df.columns = hdr
    return df
