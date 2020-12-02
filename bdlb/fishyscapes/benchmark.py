# Copyright 2019 BDL Benchmarks Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""BDLB Adapter for Fishyscapes Validation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from typing import List
from tqdm import tqdm
import tensorflow_datasets as tfds

from ..core.levels import Level
from ..core.constants import DATA_DIR
from ..core.benchmark import Benchmark, DataSplits
from ..core.benchmark import BenchmarkInfo
from ..core import transforms
from .fishyscapes_tfds import _CITATION, _DESCRIPTION, Fishyscapes
from tensorflow_datasets.image.cityscapes import Cityscapes


class FishyscapesValidation(Benchmark):

    def __init__(self, level='realworld', download_and_prepare=True, **kwargs):
        if level is None:
            level = 'realworld'
        if not level == 'realworld':
            raise UserWarning("Fishyscapes only provides data on realworld level.")
        if download_and_prepare:
            for config in ['LostAndFound', 'Static']:
                self.download_and_prepare(config)

    @classmethod
    def download_and_prepare(cls, config, register_checksums=False):
        """Downloads and prepares necessary datasets for benchmark."""
        if register_checksums:
            dl_config = tfds.download.DownloadConfig(register_checksums=True)
            Fishyscapes(config=config).download_and_prepare(download_config=dl_config)
        else:
            Fishyscapes(config=config).download_and_prepare()

    @classmethod
    def load(cls, config_str):
        """
        Returns the specified dataset.

        Args:
            config_str: config string as defined by tensorflow datasets, without the
                dataset name , e.g. Static:1.0.0 for dataset Fishyscapes/Static:1.0.0
        """
        if hasattr(tfds.core.registered, '_dataset_name_and_kwargs_from_name_str'):
            name_kwargs_func = tfds.core.registered._dataset_name_and_kwargs_from_name_str
        else:
            name_kwargs_func = tfds.core.load._dataset_name_and_kwargs_from_name_str
        _, builder_args = name_kwargs_func('fishyscapes/{}'.format(config_str))

        if builder_args['config'] == 'Static':
            # Make sure that the cityscapes dataset is ready.
            Cityscapes(config='semantic_segmentation').download_and_prepare()
        ds = Fishyscapes(**builder_args)
        # Fishyscapes has no trainset, and we need to wait for the PR on tfds to return
        # the cityscapes dataset as training set.
        return DataSplits(None, ds.as_dataset(split='validation'), None)

    @classmethod
    def get_dataset(cls, config_str):
        return cls.load(config_str)[1]

    @property
    def info(self):
        """Description of the benchmark."""
        return BenchmarkInfo(
            description=_DESCRIPTION,
            urls="https://fishyscapes.com/",
            setup="",
            citation=_CITATION)

    @property
    def level(self):
        """Hardness level of the benchmark."""
        return Level.from_str('realworld')

    @classmethod
    def evaluate(cls, estimator, dataset=None, name=None, num_points=50):
        """Evaluates an `estimator` for anomaly detection on the given dataset.

        Optimized to use as little memory as possible, takes around 40GB of memory for
        1000 images.
        The implementation is based on sklearn ranking metrics:
        https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/metrics/ranking.py

        Args:
        estimator: `lambda x: uncertainty`, an uncertainty estimation
            function, which returns a matrix `uncertainty` with an uncertainty value for
            each pixel.
        dataset: `tf.data.Dataset`, on which dataset to performance evaluation.
            Defaults to the FS Lost & Found Validation dataset.
            The dataset requires properties 'image_left' and 'mask'.
        output_dir: (optional) `str`, directory to save figures.
        name: (optional) `str`, the name of the method.
        num_points: (optional) number of points to save for PR curves
        """
        if dataset is None:
            dataset = cls.get_dataset()

        # predict uncertainties over the dataset
        labels = []
        uncertainties = []
        for batch in tqdm(dataset):
            labels.append(batch['mask'].numpy())
            uncertainties.append(estimator(batch['image_left']).numpy())

        return calculate_metrics_perpixAP(
            labels, 
            uncertainties, 
            num_points=num_points,
        )
     

def calculate_metrics_perpixAP(labels : List[np.ndarray], uncertainties : List[np.ndarray], num_points=50):

    # concatenate lists for labels and uncertainties together
    if (labels[0].shape[-1] > 1 and np.ndim(labels[0]) > 2) or \
            (labels[0].shape[-1] == 1 and np.ndim(labels[0]) > 3):
        # data is already in batches
        labels = np.concatenate(labels)
        uncertainties = np.concatenate(uncertainties)
    else:
        labels = np.stack(labels)
        uncertainties = np.stack(uncertainties)
    labels = labels.squeeze()
    uncertainties = uncertainties.squeeze()

    # NOW CALCULATE METRICS
    pos = labels == 1
    valid = np.logical_or(labels == 1, labels == 0)  # filter out void
    gt = pos[valid]
    del pos
    uncertainty = uncertainties[valid].reshape(-1).astype(np.float32, copy=False)
    del valid

    # Sort the classifier scores (uncertainties)
    sorted_indices = np.argsort(uncertainty, kind='mergesort')[::-1]
    uncertainty, gt = uncertainty[sorted_indices], gt[sorted_indices]
    del sorted_indices

    # Remove duplicates along the curve
    distinct_value_indices = np.where(np.diff(uncertainty))[0]
    threshold_idxs = np.r_[distinct_value_indices, gt.size - 1]
    del distinct_value_indices, uncertainty

    # Accumulate TPs and FPs
    tps = np.cumsum(gt, dtype=np.uint64)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    del threshold_idxs

    # Compute Precision and Recall
    precision = tps / (tps + fps)
    precision[np.isnan(precision)] = 0
    recall = tps / tps[-1]
    # stop when full recall attained and reverse the outputs so recall is decreasing
    sl = slice(tps.searchsorted(tps[-1]), None, -1)
    precision = np.r_[precision[sl], 1]
    recall = np.r_[recall[sl], 0]
    average_precision = -np.sum(np.diff(recall) * precision[:-1])

    # select num_points values for a plotted curve
    interval = 1.0 / num_points
    curve_precision = [precision[-1]]
    curve_recall = [recall[-1]]
    idx = recall.size - 1
    for p in range(1, num_points):
        while recall[idx] < p * interval:
            idx -= 1
        curve_precision.append(precision[idx])
        curve_recall.append(recall[idx])
    curve_precision.append(precision[0])
    curve_recall.append(recall[0])
    del precision, recall

    if tps.size == 0 or fps[0] != 0 or tps[0] != 0:
        # Add an extra threshold position if necessary
        # to make sure that the curve starts at (0, 0)
        tps = np.r_[0., tps]
        fps = np.r_[0., fps]

    # Compute TPR and FPR
    tpr = tps / tps[-1]
    del tps
    fpr = fps / fps[-1]
    del fps

    # Compute AUROC
    auroc = np.trapz(tpr, fpr)

    # Compute FPR@95%TPR
    fpr_tpr95 = fpr[np.searchsorted(tpr, 0.95)]

    return {
        'auroc': auroc,
        'AP': average_precision,
        'FPR@95%TPR': fpr_tpr95,
        'recall': np.array(curve_recall),
        'precision': np.array(curve_precision),
    }

