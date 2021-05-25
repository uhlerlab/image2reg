from typing import List

import pandas as pd


def dict_to_csv_gz(data: dict, save_path: str, index: List[str] = None):
    data = pd.DataFrame.from_dict(data=data)
    if index is not None:
        data.index = index
    data.to_csv(save_path, compression="gzip", header=True, index=False)
