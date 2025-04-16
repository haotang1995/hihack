# Change to balrog env to run it

import os, os.path as osp
import msgpack
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
import time
import shutil
from pathlib import Path
import copy

from nle_language_wrapper import nle_language_obsv, NLELanguageWrapper
nle_language = nle_language_obsv.NLELanguageObsv()

from datasets import Dataset, DatasetDict, load_from_disk

import argparse

def main():
    parser = argparse.ArgumentParser(description="Process NLE datasets")
    parser.add_argument('--role', type=str, default=None, nargs='+', help='Role to process')
    parser.add_argument('--seed', type=str, default=None, nargs='+', help='Seed to process')
    parser.add_argument('-c', '--num_processes', type=int, default=10, help='Number of processes to use')
    parser.add_argument('--step_limit', type=int, default=None, help='Step limit for processing')
    args = parser.parse_args()

    DATASET_PATH = osp.join(osp.dirname(osp.abspath(__file__)), 'data', 'full_aa'+('' if args.role is None else '_' + '_'.join(sorted(args.role))))
    if not args.seed:
        seeds = [seed for seed in os.listdir(DATASET_PATH) if osp.isdir(osp.join(DATASET_PATH, seed))]
    else:
        seeds = args.seed
    print(f"\nFound {len(seeds)} seeds to process")

    num_processes = min(args.num_processes, len(seeds))  # Don't use more processes than seeds
    print(f"Using {num_processes} processes for parallel processing")

    seeds = [seeds[ri] for ri in np.random.permutation(len(seeds))]  # Shuffle seeds for randomness
    for seed in seeds:
        process_dataset(osp.join(DATASET_PATH, seed), num_processes=num_processes, role=args.role, step_limit=args.step_limit,)

def process_dataset(dataset_path, num_processes=10, role=None, step_limit=None,):
    hf_dataset_path = osp.join(dataset_path, f'hf_text_dataset_{step_limit if step_limit is not None else "full"}.hf')
    if osp.exists(hf_dataset_path):
        print(f"Dataset already processed, skipping: {hf_dataset_path}")
        return None

    start_time = time.time()
    dataset = load_dataset(dataset_path)
    load_time = time.time() - start_time

    if dataset is None:
        print(f"Failed to load dataset from: {dataset_path}")
        return None
    print(f"Loaded dataset, length: {len(dataset)} in {load_time:.2f} seconds")

    if (step_limit is None and not dataset[-1]['done']) or (step_limit is not None and len(dataset) < step_limit):
        print(f"Dataset not finished, last entry: {dataset[-1]['info']}")
        # os.remove(osp.join(dataset_path, 'history.msgpack'))
        return None
    else:
        print(f"Dataset finished, last entry: {dataset[-1]['info']}")

    if role is not None:
        try:
            first_message = textualize_obs(dataset[0])['obs']['text_message'].lower()
        except Exception as e:
            print(f"Error processing first message: {e}")
            return None
        if not all(kw.lower() in first_message for kw in role):
            print(f"Role '{role}' not found in first message: {first_message}")
            return None
        else:
            print(f"Role '{role}' found in first message: {first_message}")


    process_start = time.time()
    with Pool(num_processes) as pool:
        # Use partial to pass the nle_language instance to the worker function
        textualize_obs_partial = partial(textualize_obs)
        processed_dataset = pool.map(textualize_obs_partial, dataset)
    # processed_dataset = [textualize_obs(data) for data in dataset]
    process_time = time.time() - process_start
    print(f'Processed dataset, length: {len(processed_dataset)} in {process_time:.2f} seconds')

    # Convert to HF Dataset and save to disk
    hf_dataset = Dataset.from_list(processed_dataset)
    hf_dataset.save_to_disk(hf_dataset_path)
    print(f"Saved dataset to: {hf_dataset_path}")

    print(f"Dataset processing complete. Total time: {time.time() - start_time:.2f} seconds")

    return {
        'seed': osp.basename(dataset_path),
        'dataset_path': dataset_path,
        'length': len(hf_dataset),
        'timing': {
            'load_time': load_time,
            'process_time': process_time,
            'total_time': time.time() - start_time
        }
    }

def load_dataset(dataset_path):
    filename = osp.join(dataset_path, 'history.msgpack')
    if not osp.exists(filename):
        print(f"Dataset file not found: {filename}")
        return None
    print(f"Loading dataset from: {filename}")
    with open(filename, 'rb') as f:
        dataset = msgpack.unpack(f)
    return dataset

def ascii_render(chars):
    chars = np.array(chars,)
    rows, cols = chars.shape
    result = ""
    for i in range(rows):
        for j in range(cols):
            entry = chr(chars[i, j])
            result += entry
        result += "\n"
    return result

def textualize_obs(data):
    obs = data['obs']

    glyphs = obs['glyphs']
    blstats = obs['blstats']
    tty_cursor = obs['tty_cursor']
    inv_strs = obs['inv_strs']
    inv_letters = obs['inv_letters']
    tty_chars = obs['tty_chars']

    out_obs = {
        "text_glyphs": nle_language.text_glyphs(glyphs, blstats).decode("latin-1"),
        "text_message": nle_language.text_message(tty_chars).decode("latin-1"),
        "text_blstats": nle_language.text_blstats(blstats).decode("latin-1"),
        "text_inventory": nle_language.text_inventory(inv_strs, inv_letters).decode("latin-1"),
        "text_cursor": nle_language.text_cursor(glyphs, blstats, tty_cursor).decode("latin-1"),
        "text_map": ascii_render(glyphs),
    }

    data = copy.deepcopy(data)
    data['obs'] = out_obs
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

if __name__ == "__main__":
    main()
