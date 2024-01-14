from hvo_sequence import io_helpers, drum_mappings, hvo_seq
from note_seq import midi_io

import os
import numpy as np
import pickle
import random
import mido

TIME_STEPS = 32
FEATURES = 27

# Pipeline functions ----------------------------------------------------------
# -----------------------------------------------------------------------------

def serialize_hvo_pairs(midi_dir, output_path):
    # hvo pairs consist of a monotonic groove sequence and a full-groove sequence in that order
    hvo_pairs = []

    errors = 0
    # Iterate through each MIDI file in the directory
    for midi_file in os.listdir(midi_dir):
        if midi_file.endswith('.mid'):
            midi_path = os.path.join(midi_dir, midi_file)
            try:
                hvo_sequence = midi_to_hvo_seq(midi_path)
                hvo_array = hvo_seq_to_array(hvo_sequence)
                monotonic_array = hvo_sequence.flatten_voices()

                assert len(hvo_array) == len(monotonic_array), f"len(hvo_array) != len(monotonic_array). Midi path: {midi_path}. hvo_array: {len(hvo_array)}, monotonic_array: {len(monotonic_array)}"

                # pad both arrays, if needed
                if len(hvo_array) < TIME_STEPS:
                    hvo_array = pad_hvo_timesteps(hvo_array, TIME_STEPS)
                if len(monotonic_array) < TIME_STEPS:
                    monotonic_array = pad_hvo_timesteps(monotonic_array, TIME_STEPS)

                # ensure right dimensions of arrays
                assert len(hvo_array) == TIME_STEPS, f"HVO_array is {len(hvo_array)} steps long, when it should be 32. Midi path: {midi_path}"
                assert len(monotonic_array) == TIME_STEPS, f"MONOTONIC_array is {len(monotonic_array)} steps long, when it should be 32. Midi path: {midi_path}"

                assert len(monotonic_array[0]) == FEATURES, f"MONOTONIC_array has {len(monotonic_array[0])} features, when it should have 27. Midi path: {midi_path}"
                assert len(hvo_array[0]) == FEATURES, f"HVO_array has {len(hvo_array[0])} features, when it should have 27. Midi path: {midi_path}"
                
                hvo_pairs.append((monotonic_array, hvo_array))

            except AssertionError as ae:
                raise ae
            except Exception as e:
                print(f"Error processing {midi_file}: {e}")
                errors += 1

    # Serialize the HVO sequences to a file
    with open(output_path, 'wb') as file:
        pickle.dump(hvo_pairs, file)
    print(f"Serialized {len(hvo_pairs)} HVO sequences to {output_path}. {errors} errors occurred.")

def partitionData(sourceDir, trainingDir, testDir, validationDir):
    print("----------------------------------")
    print(f"partitioning data from {sourceDir} into {trainingDir}, {testDir}, and {validationDir}")

    testPercent = int(0.1 * len(os.listdir(sourceDir)))
    validationPercent = int(0.1 * len(os.listdir(sourceDir)))

    for i in range(testPercent):
        randomChoice = random.randint(0, len(os.listdir(sourceDir)) - 1)
        f = os.listdir(sourceDir)[randomChoice]

        oldPath = sourceDir + '/' + f
        newPath = testDir + '/' + f

        os.rename(oldPath, newPath)

    print(f"moved {testPercent} randomly chosen files from {sourceDir} to {testDir}")

    for i in range(validationPercent):
        randomChoice = random.randint(0, len(os.listdir(sourceDir)) - 1)
        f = os.listdir(sourceDir)[randomChoice]

        oldPath = sourceDir + '/' + f
        newPath = validationDir + '/' + f

        os.rename(oldPath, newPath)

    print(f"moved {validationPercent} randomly chosen files from {sourceDir} to {validationDir}")

    remainingFiles = len(os.listdir(sourceDir))
    filesMoved = 0
    for f in os.listdir(sourceDir):
        oldPath = sourceDir + '/' + f
        newPath = trainingDir + '/' + f

        filesMoved += 1

        os.rename(oldPath, newPath)

    assert remainingFiles == filesMoved, f"remainingFiles: {remainingFiles}, filesMoved: {filesMoved}"

    print(f"moved the rest of {remainingFiles} files from {sourceDir} to {trainingDir}")
    print("----------------------------------")


# Conversion functions --------------------------------------------------------
# -----------------------------------------------------------------------------

def midi_to_hvo_seq(midi_path: str) -> hvo_seq.HVO_Sequence:
    """
    Converts the midi file at midi path into an HVOSequence.
    """
    # Convert the MIDI file to a NoteSequence
    ns = midi_io.midi_file_to_note_sequence(midi_path)

    # Convert the NoteSequence to an HVOSequence
    hvo_sequence = io_helpers.note_sequence_to_hvo_sequence(ns, drum_mappings.ROLAND_REDUCED_MAPPING)

    # TODO: In the data pipeline, some hvo sequences have an offset of inf. 
    # This might be a bug in the hvo_sequence library or the note sequence library.
    # For now, we'll just skip these sequences, but we should fix this bug.
    if not is_hvo_seq_valid(hvo_sequence):
        raise Exception(f"Invalid HVO sequence. midi path: {midi_path}")

    return hvo_sequence

def hvo_seq_to_array(hvo_sequence: hvo_seq.HVO_Sequence) -> np.ndarray:
    """
    Converts an HVOSequence to a numpy array.
    """
    hits = hvo_sequence.hits
    velocities = hvo_sequence.velocities
    offsets = hvo_sequence.offsets
    
    hvo_array = np.concatenate((hits, velocities, offsets), axis=1)

    return hvo_array

def array_to_hvo_seq(hvo_array: np.ndarray, tempo: int=120) -> hvo_seq.HVO_Sequence:
    """
    Converts an hvo_array to an HVOSequence.
    """
    if not is_valid_hvo_array(hvo_array):
        raise Exception(f"Invalid HVO array. hvo_array: {hvo_array.shape}")

    hvo_sequence = hvo_seq.HVO_Sequence(drum_mappings.ROLAND_REDUCED_MAPPING)
    hvo_sequence.hvo = hvo_array
    hvo_sequence.add_tempo(time_step=0, qpm=tempo)

    hvo_sequence.add_time_signature(time_step=0, numerator=4, denominator=4, beat_division_factors=[4])

    return hvo_sequence

def hvo_seq_to_midi(hvo_sequence: hvo_seq.HVO_Sequence, output_path: str):
    """
    Converts an HVOSequence to a MIDI file.
    """
    # Convert the HVOSequence to a NoteSequence
    ns = hvo_sequence.to_note_sequence(9)
    # Write the NoteSequence to a MIDI file
    io_helpers.save_note_sequence_to_midi(ns, output_path)


# Helper functions ------------------------------------------------------------
# -----------------------------------------------------------------------------

def pad_hvo_timesteps(hvo_array, time_steps) -> np.ndarray:
    
    assert len(hvo_array) <= time_steps, f"can't pad hvo_array to {time_steps} time steps because it is already {len(hvo_array)} time steps long"

    missing_timesteps = time_steps - len(hvo_array)
    return np.pad(hvo_array, ((0, missing_timesteps),(0, 0)), 'constant', constant_values=0.0)

def is_hvo_seq_valid(hvo_sequence) -> bool:
    return is_valid_hvo_array(hvo_sequence.hits) and is_valid_hvo_array(hvo_sequence.velocities) and is_valid_hvo_array(hvo_sequence.offsets)

def is_valid_hvo_array(hvo_array) -> bool:
    return np.any(np.isinf(hvo_array)) == False and np.any(np.isnan(hvo_array)) == False

if __name__ == "__main__":
    mid = mido.MidiFile("/Users/danielfloresg/cs/full_thesis_proj/processing/partitioned/training/32_jazz_120_beat_4-4_slice_002.mid")
    print(mid.tracks[0])