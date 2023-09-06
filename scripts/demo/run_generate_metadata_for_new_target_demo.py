import sys
import pandas as pd
import numpy as np

sys.path.append(".")

from src.utils.basic.io import get_file_list

img_dir = "./test_data/UNKNOWN/images/raw/plate"

all_files = get_file_list(root_dir = img_dir, absolute_path=False)
columns= ['Image_Metadata_Plate', 'Image_Metadata_Well', 'Image_FileName_IllumHoechst', 'Image_Count_Nuclei', 'Image_Metadata_GeneID', 'Image_Metadata_GeneSymbol', 'Image_Metadata_IsLandmark', 'Image_Metadata_AlleleDesc', 'Image_Metadata_ExpressionVector', 'Image_Metadata_FlaggedForToxicity', 'Image_Metadata_IE_Blast_noBlast', 'Image_Metadata_IntendedOrfMismatch', 'Image_Metadata_OpenOrClosed', 'Image_Metadata_RNAiVirusPlateName', 'Image_Metadata_Site', 'Image_Metadata_Type', 'Image_Metadata_Virus_Vol_ul', 'Image_Metadata_TimePoint_Hours', 'Image_Metadata_ASSAY_WELL_ROLE']
metadata = pd.DataFrame(np.zeros((len(all_files), len(columns))), columns=columns)
metadata["Image_FileName_IllumHoechst"] = np.array(all_files)
metadata["Image_Metadata_Plate"] = "plate"
metadata["Image_Metadata_Well"] = "X"
metadata["Image_Metadata_GeneSymbol"] = "UNKNOWN"

metadata.to_csv("./test_data/UNKNOWN/images/metadata/images_md.csv")
























