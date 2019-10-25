"""Fishyscapes Lost & Found Validation Dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from itertools import chain
from os import path
import re

import tensorflow_datasets as tfds
from tensorflow_datasets.core import api_utils
from .lost_and_found import LostAndFound, LostAndFoundConfig

_CITATION = """
@article{blum2019fishyscapes,
  title={The Fishyscapes Benchmark: Measuring Blind Spots in Semantic Segmentation},
  author={Blum, Hermann and Sarlin, Paul-Edouard and Nieto, Juan and Siegwart, Roland and Cadena, Cesar},
  journal={arXiv preprint arXiv:1904.03215},
  year={2019}
}
"""

_DESCRIPTION = """
Benchmark of anomaly detection for semantic segmentation in urban driving images.
"""

class FishyscapesConfig(tfds.core.BuilderConfig):
  '''BuilderConfig for Fishyscapes

    Args:
  '''
  @api_utils.disallow_positional_args
  def __init__(self, base_data='lost_and_found', **kwargs):
    super().__init__(**kwargs)

    assert base_data in ['lost_and_found']
    self.base_data = base_data


class Fishyscapes(tfds.core.GeneratorBasedBuilder):
  """Fishyscapes Lost & Found Validation Dataset"""

  VERSION = tfds.core.Version('1.0.0')

  BUILDER_CONFIGS = [
      FishyscapesConfig(
        name='Lost and Found',
        description='Validation set based on LostAndFound images.',
        version=VERSION,
      )]

  def _info(self):
    # TODO(fishyscapes): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        # This is the description that will appear on the datasets page.
        description=_DESCRIPTION,
        # tfds.features.FeatureConnectors
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'image_id': tfds.features.Text(),
            'basedata_id': tfds.features.Text(),
            'image_left': tfds.features.Image(shape=(1024, 2048, 3),
                                              encoding_format='png'),
            'mask': tfds.features.Image(shape=(1024, 2048, 1),
                                        encoding_format='png'),
        }),
        supervised_keys=('image_left', 'mask'),
        # Homepage of the dataset for documentation
        urls=[],
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    # download the data
    dl_paths = dl_manager.download({
        'mask': 'http://robotics.ethz.ch/~asl-datasets/Fishyscapes/fishyscapes_lostandfound.zip',
    })
    dl_paths = dl_manager.extract(dl_paths)

    if self.builder_config.base_data == 'lost_and_found':
      base_builder = LostAndFound(config=LostAndFoundConfig(
          name='fishyscapes',
          description='Config to generate images for the Fishyscapes dataset.',
          version='1.0.0',
          right_images=False,
          segmentation_labels=False,
          instance_ids=False,
          disparity_maps=False,
          use_16bit=False))

    # manually force a downlaod and split generation for the base dataset
    # There is no tfds-API that allows for getting images by id, so this is the only
    # option.
    splits = base_builder._split_generators(dl_manager)
    generators = [base_builder._generate_examples(**split.gen_kwargs)
                  for split in splits]

    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION,
            # These kwargs will be passed to _generate_examples
            gen_kwargs={'fishyscapes_path': dl_paths['mask'],
                        'base_images': {key: features for key, features in chain(*generators)}},
        ),
    ]

  def _generate_examples(self, fishyscapes_path, base_images):
    """Yields examples."""
    for filename in tf.io.gfile.listdir(fishyscapes_path):
      fs_id, cityscapes_id = _get_ids_from_labels_file(filename)
      features = {
        'image_id': fs_id,
        'basedata_id': cityscapes_id,
        'mask': path.join(fishyscapes_path, filename),
        'image_left': base_images[cityscapes_id]['image_left'],
      }
      yield fs_id, features

# Helper functions

IDS_FROM_FILENAME = re.compile(r'([0-9]+)_(.+)_labels.png')

def _get_ids_from_labels_file(labels_file):
  '''Returns the ids (fishyscapes and cityscapes format) from the filename of a labels
  file. Used to associate a fishyscapes label file with the corresponding cityscapes
  image.

  Example:
    '0000_04_Maurener_Weg_8_000000_000030_labels.png' -> '0000', '04_Maurener_Weg_8_000000_000030'
  '''
  match = IDS_FROM_FILENAME.match(labels_file)
  return match.group(1), match.group(2)
