from hvo_sequence import io_helpers, drum_mappings, hvo_seq
from note_seq import midi_io

import numpy as np
import mido

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


def hvo_seq_to_monotonic_seq(hvo_sequence: hvo_seq.HVO_Sequence):
    """
    Converts an HVOSequence to its monotonic counterpart
    """
    monotonic_array = hvo_sequence.flatten_voices()
    monotonic_seq = array_to_hvo_seq(monotonic_array, tempo=hvo_sequence.tempos[0].qpm)
    return monotonic_seq


def midi_to_monotonic_midi(midi_path: str, output_path: str):
    """
    Converts a MIDI file to a monotonic MIDI file.
    """
    hvo_sequence = midi_to_hvo_seq(midi_path)
    monotonic_sequence = hvo_seq_to_monotonic_seq(hvo_sequence)
    hvo_seq_to_midi(monotonic_sequence, output_path)

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