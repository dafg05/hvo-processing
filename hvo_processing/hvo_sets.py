import pickle
import os
import hvo_sequence

from hvo_sequence.hvo_seq import HVO_Sequence

from typing import List

TRAIN_DIR = "GrooveMIDI_processed_train"
TEST_DIR = "GrooveMIDI_processed_test"
VALIDATION_DIR = "GrooveMIDI_processed_validation"

class HVOSetRetriever():
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.train_dir = f"{root_dir}/{TRAIN_DIR}"
        self.test_dir = f"{root_dir}/{TEST_DIR}"
        self.validation_dir = f"{root_dir}/{VALIDATION_DIR}"

        # we're only gonna check if the train dir is a valid dir. If this is true,
        # this implies that root_dir is a valid dir, and most likely means that both
        # test and validation dirs are valid dirs.

        if not os.path.isdir(self.train_dir):
            raise FileNotFoundError(f"{self.train_dir} is not a directory!")

    @staticmethod
    def __load_dataset_from_dir__(data_dir: str):
        with open(f"{data_dir}/hvo_sequence_data.obj", "rb") as hvo_file:
            data_set = pickle.load(hvo_file)
            print(f"Loaded {len(data_set)} hvo_seqs from {data_dir}")
            return data_set

    def get_train_set(self) -> List[HVO_Sequence]:
        return HVOSetRetriever.__load_dataset_from_dir__(self.train_dir)
    
    def get_test_set(self) -> List[HVO_Sequence]:
        return HVOSetRetriever.__load_dataset_from_dir__(self.test_dir)

    def get_validation_set(self) -> List[HVO_Sequence]:
        return HVOSetRetriever.__load_dataset_from_dir__(self.validation_dir)