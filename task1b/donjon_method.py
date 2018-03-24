import numpy as np
from pandas import read_csv, DataFrame


# Define exercise functions
def read_csv_to_matrix(filename):
    return read_csv(filename, header=None).as_matrix()


def write_to_csv_from_vector(filename, vec):
    return DataFrame(vec).to_csv(filename, index=False, header=False)


# Import Data
weights5 = read_csv_to_matrix("sample_franz_5.csv")
weights10 = read_csv_to_matrix("sample_franz_10.csv")
weights900 = read_csv_to_matrix("sample_franz_900.csv")

mean = (weights5 + weights10 + weights900)/3
print(mean)
write_to_csv_from_vector("sample_franz_donjon_method.csv", mean)
