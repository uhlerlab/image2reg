import os
from typing import List

import pandas as pd


def get_file_list(
    root_dir: str,
    absolute_path: bool = True,
    file_ending: bool = True,
    file_type_filter: str = None,
) -> List:

    assert os.path.exists(root_dir)
    list_of_data_locs = []
    for (root_dir, dirname, filename) in os.walk(root_dir):
        for file in filename:
            if file_type_filter is not None and file_type_filter not in file:
                continue
            else:
                if not file_ending:
                    file = file[: file.index(".")]
                if absolute_path:
                    list_of_data_locs.append(os.path.join(root_dir, file))
                else:
                    list_of_data_locs.append(file)
    return sorted(list_of_data_locs)


def get_genesets_from_gmt_file(file):
    data = pd.read_csv(file, sep="\t", index_col=0, header=None)
    data = data.iloc[:, 1:]
    geneset_dict = {}
    for i in data.index:
        geneset_dict[i] = list(data.loc[i].loc[data.loc[i].notnull()])
    return geneset_dict
