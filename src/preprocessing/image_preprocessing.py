import logging
import os
from shutil import copyfile
from typing import List, Tuple

import cv2
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tifffile import tifffile
from tqdm import tqdm

from skimage.io import imread
from skimage.measure import regionprops, label
from skimage.morphology import remove_small_objects, remove_small_holes

from src.utils.basic.io import get_file_list
from src.utils.basic.segmentation import get_mask_image_from_outline, pad_image


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

        self.nuclei_metadata_file = None
        self.processed_image_metadata_file = None

        self.nuclei_dir = None
        self.pad_size = None

    def add_image_illumination_col(self, posfix: str = "_illum_corrected"):
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
        self.metadata.to_csv(
            os.path.join(self.output_dir, "filtered_image_metadata.csv")
        )

    def save_filtered_images(self,):
        output_dir = os.path.join(self.output_dir, "filtered")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        n = len(self.metadata)
        for i in tqdm(range(n), desc="Copying filtered images"):
            plate = self.metadata.iloc[
                i, list(self.metadata.columns).index(self.plate_col_name)
            ]
            filename = self.metadata.iloc[
                i, list(self.metadata.columns).index(self.illum_image_col_name)
            ]

            plate_output_dir = os.path.join(output_dir, str(plate))
            plate_input_dir = os.path.join(self.image_input_dir, str(plate))
            if not os.path.exists(plate_output_dir):
                os.makedirs(plate_output_dir)

            plate_output_file = os.path.join(plate_output_dir, filename)
            plate_input_file = os.path.join(plate_input_dir, filename)

            try:
                copyfile(plate_input_file, plate_output_file)
            except FileNotFoundError as e:
                logging.error(e)
        logging.debug(
            "Images (n={}) copied to {}.".format(len(self.metadata), output_dir)
        )

    def segment_all_images_given_outlines(
        self,
        outline_input_dir,
        nuclei_outline_col_name: str = "Image_FileName_NucleiOutlines",
        min_area: int = None,
        fill_holes: int = 16,
    ):
        output_dir = os.path.join(self.output_dir, "label_images")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for i in tqdm(range(len(self.metadata)), desc="Segment images using outlines"):
            plate = str(self.metadata.iloc[i, :][self.plate_col_name])

            image_file_name = self.metadata.iloc[i, :][self.illum_image_col_name]

            outline_file_name = self.metadata.iloc[i, :][nuclei_outline_col_name]
            outline_file_path = os.path.join(
                outline_input_dir, plate, outline_file_name
            )
            outline_image = imread(outline_file_path)
            plate_output_dir = os.path.join(output_dir, plate)

            if not os.path.exists(plate_output_dir):
                os.makedirs(plate_output_dir)

            label_image_file_path = os.path.join(plate_output_dir, image_file_name)

            mask_image = get_mask_image_from_outline(outline_image)

            mask_image = remove_small_holes(mask_image, area_threshold=fill_holes)
            mask_image = remove_small_objects(mask_image, min_size=min_area)
            label_image = label(mask_image)

            tifffile.imsave(label_image_file_path, label_image)

    def get_nuclear_crops(
        self,
        label_image_input_dir: str = None,
        output_dir: str = None,
        nuclei_count_col_name: str = "Image_Count_Nuclei",
        min_area: int = None,
        max_area: int = None,
        max_bbarea: int = None,
        max_eccentricity: float = None,
        min_solidity: float = None,
        min_aspect_ratio: float = None,
        convex_crop:bool=True,
    ):

        nuclei_metadata = []
        nuclei_counts = []
        image_metadata = self.metadata.copy()

        nuclei_widths = []
        nuclei_heights = []

        metadata_cols = list(self.metadata.columns)

        if output_dir is None:
            output_dir = os.path.join(self.output_dir, "nuclei_images")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for i in tqdm(range(len(self.metadata)), desc="Save nuclear crops"):
            nuclei_count = 0
            plate = str(self.metadata.iloc[i, :][self.plate_col_name])

            image_file_name = self.metadata.iloc[i, :][self.illum_image_col_name]
            label_image_file_name = image_file_name

            image_file_path = os.path.join(self.image_input_dir, plate, image_file_name)
            label_image_file_path = os.path.join(
                label_image_input_dir, plate, label_image_file_name
            )

            image = tifffile.imread(image_file_path)
            label_image = tifffile.imread(label_image_file_path)

            image_metadata.iloc[i, metadata_cols.index(nuclei_count_col_name)] = len(
                np.unique(label_image)
            )

            regions = regionprops(label_image=label_image, intensity_image=image)
            for region in regions:
                width, height = region.image.shape

                fname_start = image_file_name[: image_file_name.index(".")]
                fname_ending = image_file_name[image_file_name.index(".") :]
                plate_output_dir = os.path.join(output_dir, plate)

                if not os.path.exists(plate_output_dir):
                    os.makedirs(plate_output_dir)

                if (
                    (min_area is None or region.area >= min_area)
                    and (max_area is None or region.area <= max_area)
                    and (
                        max_eccentricity is None
                        or region.eccentricity <= max_eccentricity
                    )
                    and (max_bbarea is None or width * height <= max_bbarea)
                    and (min_solidity is None or region.solidity >= min_solidity)
                    and (
                        min_aspect_ratio is None
                        or (region.minor_axis_length / region.major_axis_length)
                        >= min_aspect_ratio
                    )
                ):
                    nuclei_widths.append(width)
                    nuclei_heights.append(height)

                    output_file_name = os.path.join(
                        plate_output_dir,
                        fname_start + "_{}".format(region.label) + fname_ending,
                    )

                    # returns convex crop of the segmented object.
                    if convex_crop:
                        xmin, ymin, xmax, ymax = region.bbox
                        cropped = image[xmin:xmax, ymin:ymax] * region.convex_image
                    else:
                        cropped = region.intensity_image

                    tifffile.imsave(output_file_name, cropped)
                    nucleus_metadata = list(self.metadata.iloc[i, :])
                    nucleus_metadata[
                        metadata_cols.index(self.illum_image_col_name)
                    ] = os.path.split(output_file_name)[1]
                    nuclei_metadata.append(nucleus_metadata)

                    nuclei_count += 1
            nuclei_counts.append(nuclei_count)

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
        nuclei_widths = np.array(nuclei_widths)
        nuclei_heights = np.array(nuclei_heights)

        nuclei_metadata = nuclei_metadata.loc[:, selected_cols]
        nuclei_metadata.columns = new_selected_cols
        nuclei_metadata["bb_width"] = nuclei_widths
        nuclei_metadata["bb_height"] = nuclei_heights

        max_width, max_height = nuclei_widths.max(), nuclei_heights.max()

        image_metadata = image_metadata.loc[:, selected_cols + [nuclei_count_col_name]]
        image_metadata.columns = new_selected_cols + [
            "nuclei_count",
        ]

        image_metadata.loc[:, "nuclei_count"] = nuclei_counts
        logging.debug("Nuclei segmentation complete.")
        logging.debug(
            "Maximum image dimensions: ({}, {})".format(max_width, max_height)
        )
        self.pad_size = max_width + 1, max_height + 1
        self.nuclei_dir = output_dir
        self.nuclei_metadata = nuclei_metadata
        self.processed_image_metadata = image_metadata
        self.nuclei_metadata_file = os.path.join(self.output_dir, "nuclei_metadata.csv")
        self.processed_image_metadata_file = os.path.join(
            self.output_dir, "image_metadata.csv"
        )

        self.nuclei_metadata.to_csv(self.nuclei_metadata_file)
        self.processed_image_metadata.to_csv(self.processed_image_metadata_file)

    def save_padded_images(
        self,
        nuclei_metadata_file: str,
        target_size: Tuple[int] = None,
        input_dir: str = None,
        image_file_col: str = "image_file",
    ):

        nuclei_metadata = pd.read_csv(nuclei_metadata_file, index_col=0)
        if input_dir is None:
            input_dir = self.nuclei_dir
        if target_size is None:
            target_size = self.pad_size
        file_list = get_file_list(input_dir)
        output_dir = os.path.join(self.output_dir, "padded_nuclei")
        n_skipped = 0
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for i in tqdm(range(len(file_list)), desc="Save padded images"):
            file = file_list[i]
            image = tifffile.imread(file)
            dir_name, file_name = os.path.split(file)

            width, height = image.shape
            if width > target_size[0] or height > target_size[1]:
                n_skipped += 1
                nuclei_metadata = nuclei_metadata.loc[
                    nuclei_metadata[image_file_col] != file_name, :
                ]
            else:
                plate = os.path.split(dir_name)[1]
                plate_output_dir = os.path.join(output_dir, plate)
                if not os.path.exists(plate_output_dir):
                    os.makedirs(plate_output_dir)
                padded_image = pad_image(image, target_size)
                padded_image = padded_image.astype(np.uint16)
                padded_image = padded_image - padded_image.min()
                padded_image = padded_image / padded_image.max()
                padded_image = np.clip(padded_image, 0, 1)
                padded_image = (padded_image * 255).astype(np.uint8)
                tifffile.imsave(os.path.join(plate_output_dir, file_name), padded_image)

        logging.debug(
            "Padding complete: {} image were skipped as they exceeded the target"
            " dimensions.".format(n_skipped)
        )
        self.nuclei_metadata_file = os.path.join(
            output_dir, "padded_nuclei_metadata.csv"
        )
        nuclei_metadata.to_csv(self.nuclei_metadata_file)

    def add_gene_label_column_to_metadata(
        self, nuclei_metadata_file: str = None, label_col: str = "gene_symbol"
    ):
        if nuclei_metadata_file is None:
            nuclei_metadata_file = self.nuclei_metadata_file
        nuclei_metadata = pd.read_csv(nuclei_metadata_file, index_col=0)
        nuclei_metadata["gene_label"] = LabelEncoder().fit_transform(
            np.array(nuclei_metadata.loc[:, label_col])
        )
        nuclei_metadata.to_csv(nuclei_metadata_file)

    def resize_and_save_images(self, target_size: Tuple[int], input_dir: str = None):
        if input_dir is None:
            input_dir = self.nuclei_dir
        file_list = get_file_list(input_dir)
        output_dir = os.path.join(self.output_dir, "resized_images")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for i in tqdm(range(len(file_list)), desc="Save resized images"):
            file = file_list[i]
            image = tifffile.imread(file)
            dir_name, file_name = os.path.split(file)
            plate = os.path.split(dir_name)[1]
            plate_output_dir = os.path.join(output_dir, plate)
            if not os.path.exists(plate_output_dir):
                os.makedirs(plate_output_dir)
            resized_image = cv2.resize(image, dsize=target_size)
            tifffile.imsave(os.path.join(plate_output_dir, file_name), resized_image)
