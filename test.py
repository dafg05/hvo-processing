import numpy as np
import tools
import os
from hvo_sequence import io_helpers

TEST_DIR = "test"
OUT_DIR = TEST_DIR + "/out"

def test_pad_hvo_timesteps():
    hvo_array = np.array([[1,2], [3,4]])
    padded_array = tools.pad_hvo_timesteps(hvo_array, 3)
    assert len(padded_array) == 3, f"padded_array is {len(padded_array)} time steps long, when it should be 4"
    print(padded_array)

def test_midi_to_hvo():
    midi_path = TEST_DIR + "/test0.mid"
    hvo_sequence = tools.midi_to_hvo_seq(midi_path)
    
    ns = hvo_sequence.to_note_sequence()
    io_helpers.save_note_sequence_to_midi(ns, OUT_DIR + "/hvo_full.mid")
    print("saved hvo_full.mid")    

def test_hvo_to_monotonic():
    midi_path = TEST_DIR + "/test1.mid"
    hvo_sequence = tools.midi_to_hvo_seq(midi_path)
    monotonic_array = hvo_sequence.flatten_voices()

    monotonic_sequence = hvo_sequence.copy()
    monotonic_sequence.hvo = monotonic_array

    ns = monotonic_sequence.to_note_sequence()
    io_helpers.save_note_sequence_to_midi(ns, OUT_DIR + "/hvo_mono.mid")
    print("saved hvo_mono.mid")

if __name__ == "__main__":
    if not os.path.exists(TEST_DIR):
        os.mkdir(TEST_DIR)
    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)

    test_pad_hvo_timesteps()
    test_midi_to_hvo()
    test_hvo_to_monotonic()
    