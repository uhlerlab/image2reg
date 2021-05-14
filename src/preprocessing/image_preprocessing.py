import logging
import os
from shutil import copyfile
from typing import List

import pandas as pd
import numpy as np
from tifffile import tifffile
from tqdm import tqdm

from skimage.io import imread
from skimage.measure import regionprops
from skimage.morphology import remove_small_objects

from src.utils.basic.segmentation import get_label_image_from_outline


class ImageDatasetPreprocessor:
    def __init__(
        self,
        image_input_dir: str,
        metadata_file: str,
        output_dir: str,
        raw_image_col_name: str = "Image_FileName_OrigHoechst",
        illum_image_col_name: str = "Image_FileName_IllumHoechst",
        plate_col_name: str = "Image_Metadata_Plate",
        well_col_name: str = "Image_Metadata_Well",
    ):
        self.image_input_dir = image_input_dir
        self.output_dir = output_dir

        self.raw_image_col_name = raw_image_col_name
        self.illum_image_col_name = illum_image_col_name
        self.plate_col_name = plate_col_name
        self.well_col_name = well_col_name

        self.metadata = pd.read_csv(metadata_file)
        self.processed_image_metadata = None
        self.nuclei_metadata = None

    def add_image_illumination_col(self, posfix: str = "illum_corrected"):
        orig_image_file_names = list(self.metadata[self.raw_image_col_name])
        illum_corrected_image_file_names = []
        for orig_image_file_name in orig_image_file_names:
            idx = orig_image_file_name.index(".")
            illum_corrected_image_file_name = (
                orig_image_file_name[:idx] + posfix + orig_image_file_name[idx:]
            )
            illum_corrected_image_file_names.append(illum_corrected_image_file_name)
        self.metadata[self.illum_image_col_name] = illum_corrected_image_file_names

    def filter_out_qc_flagged_images(
        self,
        blurry_col="Image_Metadata_QCFlag_isBlurry",
        saturated_col="Image_Metadata_QCFlag_isSaturated",
    ):
        self.metadata = self.metadata.loc[
            (self.metadata[blurry_col] == 0) & (self.metadata[saturated_col] == 0)
        ]

    def remove_outlier_images(
        self,
        outlier_plates: List = None,
        outlier_plate_wells: List = None,
        outlier_wells: List = None,
    ):
        if outlier_plates is not None:
            for outlier_plate in outlier_plates:
                self.metadata = self.metadata.loc[
                    self.metadata[self.plate_col_name] != outlier_plate
                ]
        if outlier_plate_wells is not None:
            for outlier_plate_well in outlier_plate_wells:
                self.metadata = self.metadata.loc[
                    (self.metadata[self.plate_col_name] != outlier_plate_well[0])
                    | (self.metadata[self.well_col_name] != outlier_plate_well[1])
                ]
        if outlier_wells is not None:
            for outlier_well in outlier_wells:
                self.metadata = self.metadata.loc[
                    self.metadata[self.well_col_name] != outlier_well
                ]

    def save_filtered_images(
        self,
        plate_col="Image_Metadata_Plate",
        illum_file_col="Image_FileName_IllumHoechst",
    ):
        output_dir = os.path.join(self.output_dir, "filtered")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        n = len(self.metadata)
        for i in tqdm(range(n), desc="Copying images"):
            plate = self.metadata.iloc[i, list(self.metadata.columns).index(plate_col)]
            filename = self.metadata.iloc[
                i, list(self.metadata.columns).index(illum_file_col)
            ]

            plate_output_dir = os.path.join(output_dir, str(plate))
            plate_input_dir = os.path.join(self.image_input_dir, str(plate))
            if not os.path.exists(plate_output_dir):
                os.makedirs(plate_output_dir)

            plate_output_file = os.path.join(plate_output_dir, filename)
            plate_input_file = os.path.join(plate_input_dir, filename)

            copyfile(plate_input_file, plate_output_file)
        logging.debug(
            "Images (n={}) copied to {}.".format(len(self.metadata), output_dir)
        )

    def segment_all_images(
        self,
        outline_input_dir,
        nuclei_outline_col_name: str = "Image_FileName_NucleiOutlines",
        nuclei_count_col_name: str = "Image_Count_Nuclei",
        min_area: int = None,
        max_area: int = None,
        max_bbarea: int = None,
        max_eccentricity: float = None,
    ):
        nuclei_metadata = []
        image_metadata = self.metadata.copy()

        metadata_cols = list(self.metadata.columns)
        max_width = 0
        max_length = 0
        output_dir = os.path.join(self.output_dir, "segmented_nuclei")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for i in tqdm(range(len(self.metadata))):
            plate = str(self.metadata.iloc[i, :][self.plate_col_name])

            image_file_name = self.metadata.iloc[i, :][self.illum_image_col_name]
            image_file_path = os.path.join(self.image_input_dir, plate, image_file_name)
            image = tifffile.imread(image_file_path)

            outline_file_name = self.metadata.iloc[i, :][nuclei_outline_col_name]
            outline_file_path = os.path.join(
                outline_input_dir, plate, outline_file_name
            )
            outline_image = imread(outline_file_path)

            label_image = get_label_image_from_outline(outline_image)

            label_image = remove_small_objects(label_image, min_size=min_area)

            image_metadata.iloc[i, metadata_cols.index(nuclei_count_col_name)] = len(
                np.unique(label_image)
            )

            regions = regionprops(label_image=label_image, intensity_image=image)
            for region in regions:
                width, length = region.image.shape

                fname_start = image_file_name[: image_file_name.index(".")]
                fname_ending = image_file_name[image_file_name.index(".") :]
                plate_output_dir = os.path.join(output_dir, plate)

                if not os.path.exists(plate_output_dir):
                    os.makedirs(plate_output_dir)

                if (
                    (max_area is None or region.area < max_area)
                    and (
                        max_eccentricity is None
                        or region.eccentricity < max_eccentricity
                    )
                    and (max_bbarea is None or width * length < max_bbarea)
                ):
                    max_width = max(max_width, width)
                    max_length = max(max_length, length)

                    output_file_name = os.path.join(
                        plate_output_dir,
                        fname_start + "_{}".format(region.label) + fname_ending,
                    )

                    cropped = region.intensity_image

                    tifffile.imsave(output_file_name, cropped)
                    nucleus_metadata = list(self.metadata.iloc[i, :])
                    nucleus_metadata[
                        metadata_cols.index(self.illum_image_col_name)
                    ] = os.path.split(output_file_name)[1]
                    nuclei_metadata.append(nucleus_metadata)
        nuclei_metadata = pd.DataFrame(np.array(nuclei_metadata), columns=metadata_cols)
        selected_cols = [
            "Image_Metadata_Plate",
            "Image_Metadata_Well",
            "Image_FileName_IllumHoechst",
            "Image_Metadata_GeneID",
            "Image_Metadata_GeneSymbol",
            "Image_Metadata_IsLandmark",
            "Image_Metadata_AlleleDesc",
            "Image_Metadata_ExpressionVector",
            "Image_Metadata_FlaggedForToxicity",
            "Image_Metadata_IE_Blast_noBlast",
            "Image_Metadata_IntendedOrfMismatch",
            "Image_Metadata_OpenOrClosed",
            "Image_Metadata_RNAiVirusPlateName",
            "Image_Metadata_Site",
            "Image_Metadata_Type",
            "Image_Metadata_Virus_Vol_ul",
            "Image_Metadata_TimePoint_Hours",
            "Image_Metadata_ASSAY_WELL_ROLE",
        ]
        new_selected_cols = [
            "plate",
            "well",
            "image_file",
            "gene_id",
            "gene_symbol",
            "is_landmark",
            "allele",
            "expr_vec",
            "toxicity",
            "ie_blast",
            "intended_orf_mismatch",
            "open_closed",
            "rnai_plate",
            "site",
            "type",
            "virus_vol",
            "timepoint",
            "assay_well_role",
        ]
        nuclei_metadata = nuclei_metadata.loc[:, selected_cols]
        nuclei_metadata.columns = new_selected_cols

        image_metadata = image_metadata.loc[
            :, selected_cols + [nuclei_outline_col_name, nuclei_count_col_name]
        ]
        image_metadata.columns = new_selected_cols + [
            "nuclei_outline_file",
            "nuclei_count",
        ]
        logging.debug(
            "Nuclei segmentation complete: max dimensions ({}, {})".format(
                max_width, max_length
            )
        )
        self.nuclei_metadata = nuclei_metadata
        self.processed_image_metadata = image_metadata
