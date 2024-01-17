import sys
import os

from hvo_processing import tools
from pathlib import Path

SPLIT_DIR = "split"
PROCESSED_DATA_DIR = "processed"
PARTITION_DIR = "partitioned"

def process(sourceDir=SPLIT_DIR, processedDir=PROCESSED_DATA_DIR, partitionDir=PARTITION_DIR, augParams=None, debug=False):

    if len(os.listdir(processedDir)) > 0:
        raise Exception(f"Processed data directory {processedDir} is not empty. Please clear it before processing.")
    
    partitionDir = Path(processedDir).parent.absolute().as_posix() + "/partitioned"
    
    training_dir = partitionDir + "/training"
    test_dir = partitionDir + "/test"
    validation_dir = partitionDir + "/validation"

    # populate partitioned directory
    try:
        if not os.path.exists(training_dir):
            os.mkdir(training_dir)
        if not os.path.exists(test_dir):
            os.mkdir(test_dir)
        if not os.path.exists(validation_dir):
            os.mkdir(validation_dir)
        tools.partitionData(sourceDir, training_dir, test_dir, validation_dir)
    except Exception as e:
        raise Exception(f"An error occurred when partitioning data: {e}")

    # serialize data from partitioned directory
    try:
        augSuffix = "_aug" if augParams else ""
        tools.serialize_hvo_pairs(training_dir, f"{processedDir}/training{augSuffix}.pkl", debug)
        tools.serialize_hvo_pairs(test_dir, f"{processedDir}/test{augSuffix}.pkl", debug)
        tools.serialize_hvo_pairs(validation_dir, f"{processedDir}/validation{augSuffix}.pkl", debug)
        if augParams:
            # document augmentation scheme metadata
            name = f"{processedDir}/augParams.txt"
            f = open(name, "x")
            f.write(str(augParams))
            
    except Exception as e:
        raise Exception(f"An error occurred when serializing data: {e}")

    print("Pre processing done!!")
        
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please specify a split data source directory")
        exit(1)
    sourceDir = sys.argv[1]
    process(sourceDir)