from pathlib import Path
import json
from os import environ
import numpy as np
from PIL import Image

import tensorflow_datasets as tfds
from tensorflow_datasets.core import BuilderConfig, DatasetInfo, GeneratorBasedBuilder, SplitGenerator, Version


def imread(path):
    return np.asarray(Image.open(path))


class RoadObstaclesTFDS(GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        BuilderConfig(
            name = 'RoadObstacles', 
            version = Version('0.0.2', 'sample batch 2'),
            description = """
                Photos and annotations of obstacles placed on the road, 
                featuring a variety of new objects and road surfaces.
            """
        )
    ]

    def _info(self):
        w, h = 2000, 1500
        img_x3 = tfds.features.Image(shape=(h, w, 3))
        img_x1 = tfds.features.Image(shape=(h, w, 1))

        return DatasetInfo(
            builder = self,
            features = tfds.features.FeaturesDict(dict(
                frame_id = tfds.features.Text(),
                image = img_x3,
                # Labels source: 
                labels_source = img_x1, # 0 = background, 1 = road, 2 = anomaly, 255 = ignore
                instances = img_x1,
                # Labels for on-road benchmark:
                mask = img_x1, # 0 = road, 1 = anomaly, 255 = ignore
            )),
        )

    CHANNEL_FILE_TEMPLATES = {
        'image': '{data_dir}/frames/{fid}.webp',
        'labels_source': '{data_dir}/frames/{fid}.labels/labels_semantic.png',
        'instances': '{data_dir}/frames/{fid}.labels/labels_instance.png',
    }

    @staticmethod
    def convert_labels_for_onroad_benchmark(labels_source):
        """
        Labels source: 
            0 = background, 1 = road, 2 = anomaly, 255 = ignore
        Labels for on-road benchmark:
            0 = road, 1 = anomaly, 255 = ignore
        
        Ignore everything outside of road area.
        """
        labels = np.full_like(labels_source, fill_value=255, dtype=np.uint8)
        labels[labels_source == 1] = 0
        labels[labels_source == 2] = 1
        return labels

    @classmethod
    def get_frame(cls, fid : str, data_dir : Path):
        # loading files
        channels = {
            ch_name: imread(ch_tmpl.format(fid=fid, data_dir=data_dir))
            for ch_name, ch_tmpl in cls.CHANNEL_FILE_TEMPLATES.items()
        }

        channels['frame_id'] = fid

        # adapt labels for on-road benchmark
        channels['mask'] = cls.convert_labels_for_onroad_benchmark(channels['labels_source'])

        # TFDS wants an additional 1-sized dimension at the end
        for k in ['mask', 'labels_source', 'instances']:
            channels[k] = channels[k][:, :, None]

        return channels

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Download the data and define splits."""
        
        download_server = environ.get('ROAD_OBSTACLE_URL')
        if download_server is None:
            raise RuntimeError('Please specify server URL as ROAD_OBSTACLE_URL env variable.')

        download_url = download_server + "/dataset_RoadObstacle_0.0.2.zip"
        download_dir = dl_manager.download_and_extract(download_url)

        data_dir = Path(download_dir) / 'dataset_RoadObstacle'

        splits = json.loads((data_dir / 'splits.json').read_text())

        make_split_entry = lambda name, key: SplitGenerator(
            name=name, 
            gen_kwargs = dict(data_dir=str(data_dir), split=key)
        )

        return [
            make_split_entry(tfds.Split.TEST, 'full')
        ] + [
            make_split_entry(k, k)
            for k in sorted(splits.keys())
        ]

    def _generate_examples(self, data_dir, split): #-> Iterator[Tuple[Key, Example]]:
        data_dir = Path(data_dir)
        index_file = data_dir / 'splits.json'
        frame_ids_in_split = json.loads(index_file.read_text())[split]

        for fid in frame_ids_in_split:
            yield (fid, self.get_frame(fid=fid, data_dir=data_dir))
