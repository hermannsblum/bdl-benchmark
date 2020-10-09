
# Benchmark for road obstacles
# In the LostAndFound dataset, the obstacles are located on the road in front of the car.
# Using this prior knowledge, we can ignore the non-road part of the image - we require the method to find obstacles within the road area only.
# In practice, we limit the evaluation to pixels marked as "free space" or "obstacle" in the original LAF labels.

import json
from pathlib import Path

import numpy as np
import tensorflow_datasets
from tqdm import tqdm

from ..core.benchmark import Benchmark
from ..core.levels import Level
from .benchmark import calculate_metrics_perpixAP

# Extracting frame IDs
# FishyLAF-validation is a subset of 100 frames of LostAndFound
# We want to retrieve those frames but now from the LostAndFound dataset `tensorflow_datasets.load('lost_and_found')`.
# We establish the correspondence by the frame names such as `04_Maurener_Weg_8_000000_000030`.

def list_frame_ids(tfdset, id_field_name='image_id'):
    return [str(fr[id_field_name], 'utf8') for fr in tqdm(tfdset.as_numpy_iterator())]

def extract_tfdsLAF_ids(data_dir = None):
    frame_ids = dict()
    
    for split in ('train', 'test'):
        ds = tensorflow_datasets.load('lost_and_found', split=split, data_dir = data_dir)
        frame_ids[split] = list_frame_ids(ds)
    
    with Path('tfdsLAF_index.json').open('w') as file_index:
        json.dump(frame_ids, file_index, indent='	')

    return frame_ids

def extract_FishyLAF_ids(data_dir = None, splits=('validation',) ):
    data_builder = Fishyscapes(data_dir = data_dir, config='LostAndFound')
    data_builder.download_and_prepare()
    
    frame_ids = dict()
    
    for split in splits:
        ds = data_builder.as_dataset(split=split)
        frame_ids[split] = list_frame_ids(dset, id_field_name='basedata_id')
    
    with Path('FishyLAF_index.json').open('w') as file_index:
        json.dump(frame_ids, file_index, indent='	')
    
    return frame_ids


class FishyLAF_OrigLafLabels_Dataset:
        
    def __init__(self, split='validation', tfds_data_dir = None):
        
        self.split = split
        self.tfds_data_dir = tfds_data_dir
        
    def download_and_prepare(self):
        this_dir = Path(__file__).parent

        # Load Fishyscapes-LAF ids
        id_list_FishyLAF = json.loads((this_dir / 'FishyLAF_index.json').read_text())
        self.id_list_FishyLAF = id_list_FishyLAF[self.split]
        
        # Map to tfdsLAF indices
        id_list_tfdsLAF = json.loads((this_dir / 'tfdsLAF_index.json').read_text())
        fid_to_split_and_idx = dict()
        for split in ('train', 'test'):
            fid_to_split_and_idx.update({
                fid: (split, idx)
                for (idx, fid) in enumerate(id_list_tfdsLAF[split])
            })
        self.fid_to_split_and_idx_tfdsLAF = fid_to_split_and_idx

        # Load LAF dsets
        self.tfdsLAF = {
            split: tensorflow_datasets.load(
                'lost_and_found', 
                split = split, 
                data_dir = self.tfds_data_dir,
            )
            for split in ('train', 'test')
        }
    
    def __len__(self):
        return self.id_list_FishyLAF.__len__()

    def __iter__(self):
        for i in range(self.__len__()):
            yield self[i]
    
    @staticmethod
    def tfds_get_nth_frame(ds, n):
        return list(ds.skip(n).take(1).as_numpy_iterator())[0]
    
    @staticmethod
    def convert_labels_LAF_to_Fishy(labels_laf):
        """
        In the LafRoi split we use LAF's free-space label (including the anomaly area) as the ROI
        This free-space label is a convervative coarse road area

        We convert:
        LAF labels 1=road, 2+=obstacle
        to 
        Fishy labels 1=obstacle 255=out of roi
        """
    
        # drop extraneous dimension
        if labels_laf.shape.__len__() > 2:
            labels_laf = labels_laf[:, :, 0]
        
        labels_fishy = np.full_like(labels_laf, fill_value=255, dtype=np.uint8)
        labels_fishy[labels_laf > 0] = 0 # road area
        labels_fishy[labels_laf > 1] = 1 # obstacle
        return labels_fishy
    
    def __getitem__(self, idx):
        
        fid = self.id_list_FishyLAF[idx]
        laf_split, laf_idx = self.fid_to_split_and_idx_tfdsLAF[fid]
        
        frame = self.tfds_get_nth_frame(
            ds = self.tfdsLAF[laf_split],
            n = laf_idx,
        )
        
        fid_extracted = str(frame['image_id'], 'utf8')
        frame['image_id'] = fid_extracted
        if fid_extracted != fid:
            raise AssertionError(f'LostAndFound-{laf_split}[{laf_idx}] is {fid_extracted} but we wanted {fid}')
        
        frame['mask'] = self.convert_labels_LAF_to_Fishy(frame['segmentation_label'])
        
        return frame
    


class FishyscapesLafOnRoad(Benchmark):
    level = Level.REALWORLD

    def __init__(self, download_and_prepare=True, data_dir=None, **kwargs):
        self.dataset = FishyLAF_OrigLafLabels_Dataset(tfds_data_dir = data_dir)

        if download_and_prepare:
            self.dataset.download_and_prepare()

    def evaluate(self, estimator, name=None, num_points=50):
        """
        Args:
        estimator: `lambda x: uncertainty`, an uncertainty estimation
            function, which returns a matrix `uncertainty` with an uncertainty value for 
            each pixel.
        name: (optional) `str`, the name of the method.
        num_points: (optional) number of points to save for PR curves
        """
        import tensorflow
        
        # predict uncertainties over the dataset
        labels = []
        uncertainties = []

        for frame in tqdm(self.dataset):
            labels.append(frame['mask'])

            img = frame['image_left']
            img_tf = tensorflow.constant(img)
            prediction = estimator(img_tf)
            if not isinstance(prediction, np.ndarray):
                prediction = prediction.numpy()
            uncertainties.append(prediction)

        return calculate_metrics_perpixAP(
            labels, 
            uncertainties, 
            num_points=num_points,
        )
     

# Example usage

def example_evaluation():
    import bdlb
    import tensorflow as tf
    import matplotlib.pyplot as plt

    fs = bdlb.load(benchmark="fishyscapes-road", download_and_prepare=True, data_dir="/cvlabsrc1/cvlab/tensorflow_datasets")

    def estimator(image):
        """Assigns a random uncertainty per pixel."""
        uncertainty = tf.random.uniform(image.shape[:-1])
        return uncertainty

    metrics = fs.evaluate(estimator)

    plt.plot(metrics['recall'], metrics['precision'])
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.show()
