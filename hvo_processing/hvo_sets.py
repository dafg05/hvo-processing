import pickle
import os
import hvo_sequence
import pandas as pd

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
        
    @staticmethod
    def __load_metadata__from_dir__(data_dir: str):
        return pd.read_csv(f"{data_dir}/metadata.csv")

    def get_trainset_and_metadata(self) -> tuple[List[HVO_Sequence], pd.DataFrame]:
        train_set = HVOSetRetriever.__load_dataset_from_dir__(self.train_dir)
        meta_data = HVOSetRetriever.__load_metadata__from_dir__(self.train_dir)
        return train_set, meta_data
    
    def get_testset_and_metadata(self) -> List[HVO_Sequence]:
        test_set = HVOSetRetriever.__load_dataset_from_dir__(self.test_dir)
        meta_data = HVOSetRetriever.__load_metadata__from_dir__(self.test_dir)
        return test_set, meta_data

    def get_validationset_and_metadata(self) -> List[HVO_Sequence]:
        validation_set = HVOSetRetriever.__load_dataset_from_dir__(self.validation_dir)
        meta_data = HVOSetRetriever.__load_metadata__from_dir__(self.validation_dir)
        return validation_set, meta_data
