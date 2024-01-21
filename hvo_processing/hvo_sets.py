import pickle

TRAIN_DIR = "GrooveMIDI_processed_train"
TEST_DIR = "GrooveMIDI_processed_test"
VALIDATION_DIR = "GrooveMIDI_processed_validation"

class HVOSets():
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.train_dir = f"{root_dir}/{TRAIN_DIR}"
        self.test_dir = f"{root_dir}/{TEST_DIR}"
        self.validation_dir = f"{root_dir}/{VALIDATION_DIR}"

    @staticmethod
    def __load_dataset_from_dir__(data_dir: str):
        with open(f"{data_dir}/hvo_sequence_data.obj", "rb") as hvo_file:
            data_set = pickle.load(hvo_file)
            print(f"Loaded {len(data_set)} hvo_seqs from {data_dir}")
            return data_set

    def get_train_set(self):
        return HVOSets.__load_dataset_from_dir__(self.train_dir)
    
    def get_test_set(self):
        return HVOSets.__load_dataset_from_dir__(self.test_dir)

    def get_validation_set(self):
        return HVOSets.__load_dataset_from_dir__(self.validation_dir)