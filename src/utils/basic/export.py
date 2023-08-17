from typing import List

import pandas as pd


def dict_to_hdf(data: dict, save_path: str, index: List[str] = None):
    data = pd.DataFrame.from_dict(data=data)
    if index is not None:
        data.index = index
        save_index = True
    else:
        save_index = False
    data.to_hdf(save_path, index=save_index, key="data", mode="w")
