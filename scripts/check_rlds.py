#!/usr/bin/env python3
import sys
import tensorflow_datasets as tfds

b = tfds.builder_from_directory(sys.argv[1])
assert b.info.splits["train"].num_examples >= 1
episode = next(iter(b.as_dataset(split="train").take(1)))
step = next(episode["steps"].as_numpy_iterator())
assert step["ee_action_6d"].shape == (23,)
print("RLDS OK")
