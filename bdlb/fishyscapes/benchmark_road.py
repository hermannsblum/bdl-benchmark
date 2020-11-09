
# Benchmark for road obstacles
# In the LostAndFound dataset, the obstacles are located on the road in front of the car.
# Using this prior knowledge, we can ignore the non-road part of the image - we require the method to find obstacles within the road area only.
# In practice, we limit the evaluation to pixels marked as "free space" or "obstacle" in the original LAF labels.

import tensorflow as tf
import numpy as np
from tqdm import tqdm

from ..core.benchmark import Benchmark, DataSplits
from ..core.levels import Level
from .benchmark import calculate_metrics_perpixAP
from .fishyscapes_tfds import Fishyscapes
from .road_obstacles_tfds import RoadObstaclesTFDS


class FishyscapesLafOnRoad(Benchmark):
    level = Level.REALWORLD
    
    IMAGE_KEY = 'image_left'

    def __init__(self, download_and_prepare=True, data_dir=None, **kwargs):
        if download_and_prepare:
            self.download_and_prepare()

    def download_and_prepare(self):
        Fishyscapes(config='OriginalLostAndFound').download_and_prepare()

    @classmethod
    def load(cls):
        ds = Fishyscapes(config='OriginalLostAndFound').as_dataset(split='validation')
        # map the values in the mask to match other Fishyscapes data
        def value_mapper(blob):
            values = np.ones([255])
            values[0] = 255
            values[1] = 0
            blob['mask'] = tf.gather_nd(values,
                                        tf.cast(blob['mask'], tf.int32))[..., tf.newaxis]
            return blob

        return DataSplits(None, ds.map(value_mapper), None)

    @classmethod
    def get_dataset(cls):
        return cls.load()[1]

    def evaluate(self, estimator, dataset=None, name=None, num_points=50):
        """
        Args:
        estimator: `lambda x: uncertainty`, an uncertainty estimation
            function, which returns a matrix `uncertainty` with an uncertainty value for
            each pixel.
        dataset: `tf.data.Dataset`, on which dataset to performance evaluation.
            Defaults to the FS Lost & Found Validation dataset.
            The dataset requires properties 'image_left' and 'mask'.
        name: (optional) `str`, the name of the method.
        num_points: (optional) number of points to save for PR curves
        """
        if dataset is None:
            dataset = self.get_dataset()

        # predict uncertainties over the dataset
        labels = []
        uncertainties = []
        for batch in tqdm(dataset):
            labels.append(batch['mask'].numpy())
        
            uncertainties.append(estimator(batch[self.IMAGE_KEY]).numpy())

        return calculate_metrics_perpixAP(
            labels,
            uncertainties,
            num_points=num_points,
        )


class FishyscapesOnRoad_RODataset(FishyscapesLafOnRoad):

    info = 'Benchmark with RoadObstacle dataset.'

    IMAGE_KEY = 'image'

    def __init__(self, download_and_prepare=True, **_):
        if download_and_prepare:
            self.download_and_prepare()

    def download_and_prepare(self):
        self.get_dataset_spec().download_and_prepare()

    @classmethod
    def get_dataset_spec(cls):
        return RoadObstaclesTFDS(config='RoadObstacles2048', version='0.3.2048')

    @classmethod
    def get_dataset(cls):
        return cls.get_dataset_spec().as_dataset(split='full')
