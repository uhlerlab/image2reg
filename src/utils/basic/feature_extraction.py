# -*- coding: utf-8 -*-
import copy
from typing import List

from skimage.measure._regionprops import RegionProperties
from skimage.morphology import erosion
from tifffile import imread
import pandas as pd
from skimage import measure
import numpy as np
from nmco.nuclear_features import (
    Boundary_global as BG,
    Img_texture as IT,
    Int_dist_features as IDF,
    Boundary_local_curvature as BLC,
)
from tqdm import tqdm
from mahotas.features import zernike_moments


def compute_carpenter_profiles(
    label_image: np.ndarray, intensity_image: np.ndarray
) -> pd.DataFrame:
    raw_image = copy.deepcopy(intensity_image)
    raw_image -= raw_image.min()
    raw_image = raw_image / raw_image.max()
    raw_image = (raw_image * 255).astype(int)
    # Get features for the individual nuclei in the image
    props = measure.regionprops(label_image, raw_image)

    propstable = measure.regionprops_table(
        label_image,
        raw_image,
        cache=True,
        properties=[
            "label",
            "area",
            "perimeter",
            "equivalent_diameter",
            "major_axis_length",
            "minor_axis_length",
            "eccentricity",
            "orientation",
            "solidity",
        ],
    )
    propstable = pd.DataFrame(propstable)

    propstable = pd.DataFrame(propstable)

    areashape_features = compute_area_shape_features(props)
    granularity_features = compute_granularity_features(props)
    intensity_features = compute_intensity_features(props)
    neighbor_features = compute_neighbor_features(props)
    texture_features = compute_texture_features(props)
    location_features = compute_location_features(props)

    all_features = pd.concat(
        [
            propstable,
            areashape_features,
            granularity_features,
            intensity_features,
            neighbor_features,
            texture_features,
            location_features,
        ]
    )
    return all_features


def compute_area_shape_features(regions: List[RegionProperties]):
    min_feret_diameter = []
    max_feret_diameter = []
    compacteness = []
    extent = []
    max_radius = []
    min_radius = []
    mean_radius = []
    median_radius = []
    formfactor = []
    zernike_features = []

    for region in regions:
        max_feret_diameter.append(region.feret_diameter_max)
        # Todo implement minimum feret diameter
        extent.append(np.sum(region.image) / region.bbox_area)
        distances = np.argwhere(region.image)
        distances[0, :] -= region.centroid[0]
        distances[1, :] -= region.centroid[1]
        distances = np.apply_along_axis(distances, func1d=np.linalg.norm, axis=0)
        compacteness.append(np.mean(distances) / region.area)
        edge = region.image ^ erosion(region.image)
        radii = np.argwhere(edge)
        radii[0, :] -= region.centroid[0]
        radii[1, :] -= region.centroid[1]
        radii = np.apply_along_axis(arr=radii, func1d=np.linalg.norm, axis=0)
        min_radius.append(min(radii))
        max_radius.append(max(radii))
        mean_radius.append(np.mean(radii))
        median_radius.append(np.median(radii))
        formfactor.append(4 * np.pi * region.area / (region.perimeter ** 2))
        zernike_features.append(zernike_moments(region.intensity_image, 10)[:30])
    columns = [
        "max_feret_diameter",
        "extent",
        "compactness",
        "min_radius",
        "max_radius",
        "mean_radius",
        "median_radius",
        "formfactor",
    ].extend(["zernike_{}".format(i) for i in range(30)])
    area_shape_features = np.concatenate(
        [
            max_feret_diameter,
            extent,
            compacteness,
            min_radius,
            max_radius,
            mean_radius,
            median_radius,
            formfactor,
            zernike_features,
        ],
        axis=1,
    )
    area_shape_features = pd.DataFrame(area_shape_features, columns=columns)
    return area_shape_features


def compute_granularity_features(regions: List[RegionProperties]):
    pass


def compute_intensity_features(regions: List[RegionProperties]):
    min_edge_int = []
    max_edge_int = []
    mean_edge_int = []
    std_edge_int = []
    med_edge_int = []

    min_int = []
    max_int = []
    mean_int = []
    std_int = []
    med_int = []
    lower_quart_int = []
    upper_quart_int = []

    for region in regions:
        edges = np.logical_xor(erosion(region.image), region.image)
        edge_int_img = region.intensity_image[edges]
        min_edge_int.append(np.min(edge_int_img))
        max_edge_int.append(np.max(edge_int_img))
        mean_edge_int.append(np.mean(edge_int_img))
        std_edge_int.append(np.std(edge_int_img))
        med_edge_int.append(np.median(edge_int_img))

        min_int.append(np.min(region.intensity_image))
        max_int.append(np.max(region.intensity_image))
        mean_int.append(np.mean(region.intensity_image))
        std_int.append(np.std(region.intensity_image))
        med_int.append(np.median(region.intensity_image))
        lower_quart_int.append(np.quantile(region.intensity_image, q=0.25))
        upper_quart_int.append(np.quantile(region.intensity_image, q=0.75))

    columns = [
        "min_int",
        "max_int",
        "mean_int",
        "med_int",
        "std_int",
        "q25_int",
        "q75_int",
        "min_edge_int",
        "max_edge_int",
        "mean_edge_int",
        "median_edge_int",
        "std_edge_int",
    ]
    int_features = np.concatenate(
        [
            min_int,
            max_int,
            mean_int,
            med_int,
            std_int,
            lower_quart_int,
            upper_quart_int,
            min_edge_int,
            max_edge_int,
            mean_edge_int,
            med_edge_int,
            std_edge_int,
        ]
    )
    int_features = pd.DataFrame(int_features, columns=columns)
    return int_features


def compute_texture_features(regions: List[RegionProperties]):
    pass


def compute_neighbor_features(regions: List[RegionProperties]):
    angle_between_closest_neighbors = []
    closest_neighbor = []
    closest_neighbor_dist = []
    second_closest_neighbor = []
    second_closest_neighbor_dist = []
    pass


def compute_location_features(regions: List[RegionProperties]):
    center_x = []
    center_y = []
    max_int_x = []
    max_int_y = []
    for region in regions:
        center_x.append(region.centroid[0])
        center_y.append(region.centroid[1])
        max_int_xy = np.where(region.intensity_image == np.max(region.intensity_image))
        max_int_x.append(max_int_xy[0])
        max_int_y.append(max_int_xy[1])
    columns = ["centroid_x", "centroid_y", "max_int_x", "max_int_y"]
    location_features = np.concatenate([center_x, center_y, max_int_x, max_int_y])
    location_features = pd.DataFrame(location_features, columns=columns)
    return location_features


def compute_nuclear_chromatin_features(
    label_image: np.ndarray, intensity_image: np.ndarray
) -> pd.DataFrame:
    """
    Function that reads in the raw and segmented/labelled images for a field of view and computes nuclear features.
    Note this has been used only for DAPI stained images
    Args:
        raw_image_path: path pointing to the raw image
        labelled_image_path: path pointing to the segmented image
        output_dir: path where the results need to be stored
    """
    raw_image = copy.deepcopy(intensity_image)
    raw_image -= raw_image.min()
    raw_image = raw_image / raw_image.max()
    raw_image = (raw_image * 255).astype(int)
    # Get features for the individual nuclei in the image
    props = measure.regionprops(label_image, raw_image)

    # Measure scikit's built in features
    propstable = measure.regionprops_table(
        label_image,
        raw_image,
        cache=True,
        properties=[
            "label",
            "area",
            "perimeter",
            "bbox_area",
            "convex_area",
            "equivalent_diameter",
            "major_axis_length",
            "minor_axis_length",
            "eccentricity",
            "weighted_moments",
            "weighted_moments_normalized",
            "weighted_moments_central",
            "weighted_moments_hu",
            "moments",
            "moments_normalized",
            "moments_central",
            "moments_hu",
        ],
    )
    propstable = pd.DataFrame(propstable)

    # measure other inhouse features
    all_features = pd.concat(
        [
            BLC.curvature_features(props[0].image, step=5).reset_index(drop=True),
            BG.boundary_features(
                props[0].image, centroids=props[0].local_centroid
            ).reset_index(drop=True),
            IDF.intensity_features(
                props[0].image, props[0].intensity_image
            ).reset_index(drop=True),
            IT.texture_features(props[0].image, props[0].intensity_image),
            pd.DataFrame([1], columns=["label"]),
        ],
        axis=1,
    )
    for i in range(1, len(props)):
        all_features = all_features.append(
            pd.concat(
                [
                    BLC.curvature_features(props[i].image, step=5).reset_index(
                        drop=True
                    ),
                    BG.boundary_features(
                        props[i].image, centroids=props[i].local_centroid
                    ).reset_index(drop=True),
                    IDF.intensity_features(
                        props[i].image, props[i].intensity_image
                    ).reset_index(drop=True),
                    IT.texture_features(props[i].image, props[i].intensity_image),
                    pd.DataFrame([i + 1], columns=["label"]),
                ],
                axis=1,
            ),
            ignore_index=True,
        )

    # Add in other related features for good measure
    features = pd.merge(all_features, propstable, on="label")
    features["concavity"] = (features["convex_area"] - features["area"]) / features[
        "convex_area"
    ]
    features["solidity"] = features["area"] / features["convex_area"]
    features["a_r"] = features["minor_axis_length"] / features["major_axis_length"]
    features["shape_factor"] = (features["perimeter"] ** 2) / (
        4 * np.pi * features["area"]
    )
    features["area_bbarea"] = features["area"] / features["bbox_area"]
    features["center_mismatch"] = np.sqrt(
        (features["weighted_centroid-0"] - features["centroid-0"]) ** 2
        + (features["weighted_centroid-1"] - features["centroid-1"]) ** 2
    )
    features["smallest_largest_calliper"] = (
        features["min_calliper"] / features["max_calliper"]
    )
    features["frac_peri_w_posi_curvature"] = (
        features["len_posi_curvature"] / features["perimeter"]
    )
    features["frac_peri_w_neg_curvature"] = (
        features["len_neg_curvature"].replace(to_replace="NA", value=0)
        / features["perimeter"]
    )
    features["frac_peri_w_polarity_changes"] = (
        features["npolarity_changes"] / features["perimeter"]
    )

    return features
