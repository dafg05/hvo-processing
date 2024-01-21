import numpy as np
import os
from hvo_processing import converters
from hvo_sequence import io_helpers

TEST_DIR = "test"
OUT_DIR = TEST_DIR + "/out"

def test_pad_hvo_timesteps():
    hvo_array = np.array([[1,2], [3,4]])
    padded_array = converters.pad_hvo_timesteps(hvo_array, 3)
    assert len(padded_array) == 3, f"padded_array is {len(padded_array)} time steps long, when it should be 4"
    print("Padded hvo_array")
    print(padded_array)

def test_midi_to_hvo_array_and_back():
    """
    Pipeline:
    midi -> hvo_sequence -> hvo_array -> hvo_sequence -> midi
    """

    midi_path = TEST_DIR + "/test0.mid"
    hvo_sequence = converters.midi_to_hvo_seq(midi_path)
    hvo_array = converters.hvo_seq_to_array(hvo_sequence)

    print(f"Made hvo_array from midi. hvo_array: {hvo_array.shape}")

    hvo_sequence = converters.array_to_hvo_seq(hvo_array, tempo=80)
    converters.hvo_seq_to_midi(hvo_sequence, OUT_DIR + "/hvo_full.mid")

    print("Constructed midi from hvo_array. Saved to hvo_full.mid")
    
def test_hvo_to_monotonic():
    midi_path = TEST_DIR + "/test1.mid"
    converters.midi_to_monotonic_midi(midi_path, OUT_DIR + "/hvo_mono_alt.mid")

    print("Converted full hvo to monotonic hvo. Saved to hvo_mono_alt.mid")

if __name__ == "__main__":
    if not os.path.exists(TEST_DIR):
        os.mkdir(TEST_DIR)
    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)

    test_pad_hvo_timesteps()
    test_midi_to_hvo_array_and_back()
    test_hvo_to_monotonic()