import matplotlib.pyplot as plt

from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import \
    create_windows_from_events, create_fixed_length_windows
from braindecode.preprocessing import preprocess, Preprocessor


dataset = MOABBDataset(dataset_name="Nakanishi2015", subject_ids=[1])

dataset.description