import os
from .dataset_rgb import DataLoaderVal_real


def get_validation_data_real(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderVal_real(rgb_dir, None)

