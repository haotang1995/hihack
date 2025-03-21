# Change to balrog env to run it

import os, os.path as osp
import msgpack
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
import time
import shutil
from pathlib import Path

from nle.language_wrapper.wrappers.nle_language_wrapper import nle_language_obsv
from datasets import Dataset, DatasetDict, load_from_disk

DATASET_PATH = osp.join(osp.dirname(osp.abspath(__file__)), 'data', 'full_aa')
TEMP_DATASET_PATH = osp.join(osp.dirname(osp.abspath(__file__)), 'full_aa_datasets')
nle_language = nle_language_obsv.NLELanguageObsv()

print(f"Looking for datasets in: {DATASET_PATH}")

# Create temp directory for individual datasets
os.makedirs(TEMP_DATASET_PATH, exist_ok=True)

def load_dataset(seed):
    filename = osp.join(DATASET_PATH, seed, 'history.msgpack')
    if not osp.exists(filename):
        print(f"Dataset file not found: {filename}")
        return None
    print(f"Loading dataset from: {filename}")
    with open(filename, 'rb') as f:
        dataset = msgpack.unpack(f)
    print(f"Loaded dataset, length: {len(dataset)}")
    return dataset

def textualize_obs(data):
    obs = data['obs']

    glyphs = obs['glyphs']
    blstats = obs['blstats']
    tty_cursor = obs['tty_cursor']
    inv_strs = obs['inv_strs']
    inv_letters = obs['inv_letters']
    tty_chars = obs['tty_chars']

    obs.update({
        "text_glyphs": nle_language.text_glyphs(glyphs, blstats).decode("latin-1"),
        "text_message": nle_language.text_message(tty_chars).decode("latin-1"),
        "text_blstats": nle_language.text_blstats(blstats).decode("latin-1"),
        "text_inventory": nle_language.text_inventory(inv_strs, inv_letters).decode("latin-1"),
        "text_cursor": nle_language.text_cursor(glyphs, blstats, tty_cursor).decode("latin-1"),
    })

    long_term_observations = [
        ("text_message", "message"),
        ("text_glyphs", "language observation"),
        ("text_cursor", "cursor"),
    ]

    short_term_observations = [
        ("text_blstats", "statistics"),
        ("text_inventory", "inventory"),
    ]

    long_term_context = "\n".join([f"{name}:\n{obs[key]}\n" for key, name in long_term_observations])
    short_term_context = "\n".join([f"{name}:\n{obs[key]}\n" for key, name in short_term_observations])

    obs.update({
        "balrog_long_term_context": long_term_context,
        "balrog_short_term_context": short_term_context,
    })

    data['obs'] = obs
    if 'done' in data:
        data['done'] = int(data['done'])
    if 'summary' in data and 'seed' in data['summary'] and data['summary']['seed']:
        data['summary']['seed'] = [int(x) for x in data['summary']['seed']]
    if 'info' in data and 'is_ascended' in data['info']:
        data['info']['is_ascended'] = int(data['info']['is_ascended'])
    for key in ['action', 'reward', 'done', 'info', 'summary']:
        if key not in data:
            data[key] = None

    return data

def process_dataset(seed):
    start_time = time.time()
    dataset = load_dataset(seed)
    load_time = time.time() - start_time

    if dataset is None:
        return None

    process_start = time.time()
    processed_dataset = [textualize_obs(data) for data in dataset]
    process_time = time.time() - process_start
    print(f'Processed dataset, length: {len(processed_dataset)}')

    # Convert to HF Dataset and save to disk
    hf_dataset = Dataset.from_list(processed_dataset)
    dataset_path = osp.join(TEMP_DATASET_PATH, f'dataset_{seed}')
    hf_dataset.save_to_disk(dataset_path)
    print(f"Saved dataset to: {dataset_path}")

    return {
        'seed': seed,
        'dataset_path': dataset_path,
        'length': len(hf_dataset),
        'timing': {
            'load_time': load_time,
            'process_time': process_time,
            'total_time': time.time() - start_time
        }
    }

# Get list of seeds
seeds = [seed for seed in os.listdir(DATASET_PATH) if osp.isdir(osp.join(DATASET_PATH, seed))]
print(f"\nFound {len(seeds)} seeds to process")

# Process all datasets in parallel
num_processes = min(10, len(seeds))  # Don't use more processes than seeds
print(f"Using {num_processes} processes for parallel processing")

start_time = time.time()
with Pool(processes=num_processes) as pool:
    results = pool.map(process_dataset, seeds)
parallel_time = time.time() - start_time

# Collect timing information
timing_stats = {
    'load_times': [],
    'process_times': [],
    'total_times': []
}
dataset_lengths = []

dataset_paths = {}
for result in results:
    if result is not None:
        dataset_paths[result['seed']] = result['dataset_path']
        timing_stats['load_times'].append(result['timing']['load_time'])
        timing_stats['process_times'].append(result['timing']['process_time'])
        timing_stats['total_times'].append(result['timing']['total_time'])
        dataset_lengths.append(result['length'])

# Print timing statistics
print("\nTiming Statistics:")
print(f"Parallel processing time: {parallel_time:.2f} seconds")
print(f"Average load time per dataset: {np.mean(timing_stats['load_times']):.2f} seconds")
print(f"Average process time per dataset: {np.mean(timing_stats['process_times']):.2f} seconds")
print(f"Average total time per dataset: {np.mean(timing_stats['total_times']):.2f} seconds")
print(f"Max total time for a dataset: {np.max(timing_stats['total_times']):.2f} seconds")
print(f"Min total time for a dataset: {np.min(timing_stats['total_times']):.2f} seconds")

# Calculate average length and create dataset name
avg_length = np.mean(dataset_lengths)
dataset_name = f'haotang/full-nla-aa-step{int(avg_length)}-eps{len(dataset_paths)}'
print(f"\nAverage dataset length: {avg_length:.2f}")

if False: # do not push to hub as it's too large
    # Create DatasetDict by loading datasets one at a time
    print("\nCreating DatasetDict...")
    datasets = {}
    for seed, path in dataset_paths.items():
        print(f"Loading dataset for seed {seed}...")
        datasets[seed] = load_from_disk(path)
        # Clean up individual dataset files to save space
        # shutil.rmtree(path)

    dataset = DatasetDict(datasets)

    # Push to hub with memory-efficient settings
    print(f"\nPushing dataset to hub: {dataset_name}")
    start_time = time.time()
    dataset.push_to_hub(
        dataset_name,
        private=False,
        # max_shard_size="1GB"  # Shard the dataset into 1GB chunks
    )
    push_time = time.time() - start_time
    print(f"Push to hub time: {push_time:.2f} seconds")

print(f"Total execution time: {time.time() - start_time:.2f} seconds")
