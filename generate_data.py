import functools
import gym
import multiprocessing
import nle.nethack as nh
import numpy as np
import os
import pathlib
import pdb
import sys
import time
import msgpack
import random

from argparse import ArgumentParser
from multiprocessing.pool import ThreadPool
from pathlib import Path
from tqdm import tqdm
import msgpack

from autoascend_env_wrapper import AutoAscendEnvWrapper
base_path = str(pathlib.Path().resolve())
HIHACK_PATH = os.path.join(base_path[:base_path.find('hihack')], 'hihack')

from hao_config import DEBUG_MODE
from nle_language_wrapper import nle_language_obsv, NLELanguageWrapper
nle_language = nle_language_obsv.NLELanguageObsv()

def get_message(obs):
    tty_chars = obs['tty_chars']
    return nle_language.text_message(tty_chars).decode("latin-1")

def get_seeds(n,
              target_role,
              start_seed=0):

    if not target_role:
        return np.array([i for i in range(start_seed, n+start_seed)])

    target_role = [kw.lower() for kw in target_role]
    relevant_seeds = []
    restricted_seeds = []
    with tqdm(total=n) as pbar:
        while not len(relevant_seeds) == n:
            candidate_seed = start_seed
            while 1:
                env = gym.make('NetHackChallenge-v0')
                env.seed(candidate_seed, candidate_seed)
                obs = env.reset()
                message = get_message(obs).lower()
                if all([kw in message for kw in target_role]):
                    break
                # print(f'\tSkipping seed {candidate_seed} with message: {message}')
                candidate_seed += 10**5
                candidate_seed = candidate_seed % sys.maxsize
                env.close()
            print(f'Found seed {candidate_seed} with message: {message}')
            if candidate_seed not in relevant_seeds and candidate_seed not in restricted_seeds:
                relevant_seeds += [candidate_seed]
                pbar.update(1)
            start_seed += 1
    return np.array(relevant_seeds).astype(int)

def gen_and_write_episode(idx,
                          start_i,
                          total_rollouts,
                          data_dir,
                          seeds,
                          zbase=1):
    with tqdm(total=total_rollouts, position=idx, desc=str(os.getpid())) as pbar:
        for game_id in range(start_i, start_i + total_rollouts):
            # unpack game seed
            if game_id >= seeds.shape[0]:
                break
            game_seed = seeds[game_id]

            history_path = os.path.join(data_dir, f'{game_seed}', 'history.msgpack')
            if os.path.exists(history_path):
                with open(history_path, 'rb') as f:
                    history = msgpack.unpack(f, raw=False)
                if history[-1]['done']:
                    print(f'Seed {game_seed} already done!')
                    pbar.update(1)
                    continue
            env = AutoAscendEnvWrapper(
                gym.make(
                    'NetHackChallenge-v0',
                    no_progress_timeout=1000,
                    savedir=os.path.join(data_dir, f'{game_seed}'),
                    save_ttyrec_every=1,
                    max_episode_steps=200000000
                    # max_episode_steps=10010,
                ),
                agent_args=dict(panic_on_errors=True, verbose=False),
                # step_limit=5010,
            )
            env.env.seed(game_seed, game_seed)
            try:
                env.main()
            except BaseException as e:
                if DEBUG_MODE:
                    raise e
                else:
                    with open(history_path, 'rb') as f:
                        history = msgpack.unpack(f, raw=False)
                    history[-1]['done'] = 2
                    history[-1]['info']['end_reason'] = str(e)
                    with open(history_path, 'wb') as f:
                        msgpack.pack(history, f)

            pbar.update(1)
    return 1

def create_dataset(args):
    # set main filepath
    data_dir = os.path.join(HIHACK_PATH, args.base_dir, args.dataset_name + ('' if args.role is None else '_' + '_'.join(sorted(args.role))))
    os.makedirs(data_dir, exist_ok=True)

    # first determine n unique seeds
    if args.role is None:
        role = []
    else:
        role = args.role

    relevant_seeds = get_seeds(args.episodes, role, args.seed)
    random.shuffle(relevant_seeds)

    # seeds_done = [int(f[f.rfind('/')+1:]) for f in os.listdir(data_dir)]
    seeds_done = []
    # for seed in relevant_seeds:
        # if os.path.exists(os.path.join(data_dir, f'{seed}', 'history.msgpack')):
            # seeds_done.append(seed)
    relevant_seeds = np.array(list(set(list(relevant_seeds)).difference(set(seeds_done))))
    print(f'{relevant_seeds.shape[0]} seeds generated!')


    ## parallelize dataset generation + saving
    num_proc = max(min(multiprocessing.cpu_count(), args.cores), 1) # use no more than the number of available cores
    num_rollouts_per_proc = (relevant_seeds.shape[0] // num_proc) + 1
    gen_helper_fn = functools.partial(
        gen_and_write_episode,
        data_dir=data_dir,
        seeds=relevant_seeds,
        zbase=int(np.log10(args.episodes) + 0.5)
    )

    # generate remaining args
    gen_args = []
    start_i = 0
    for j, proc in enumerate(range(num_proc - 1)):
        gen_args += [[j, start_i, num_rollouts_per_proc]]
        start_i += num_rollouts_per_proc
    if relevant_seeds.shape[0] - start_i > 0:
        gen_args += [[num_proc - 1, start_i, relevant_seeds.shape[0] - start_i]]

    # run pool
    if DEBUG_MODE:
        for k in range(num_proc):
            if len(gen_args) > k:
                gen_helper_fn(*gen_args[k])
    else:
        pool = multiprocessing.Pool(num_proc)
        runs = [pool.apply_async(gen_helper_fn, args=gen_args[k]) for k in range(num_proc) if len(gen_args) > k]
        results = [p.get() for p in runs]



def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--base_dir', default='data', type=str, help='dir where to store data')
    parser.add_argument('--dataset_name', default='full_aa', type=str)
    parser.add_argument('--seed', default=0, type=int, help='starting random seed')
    parser.add_argument('-c', '--cores', default=4, type=int, help='cores to employ')
    parser.add_argument('-n', '--episodes', type=int, default=100000)
    # parser.add_argument('--role', choices=('arc', 'bar', 'cav', 'hea', 'kni',
                                           # 'mon', 'pri', 'ran', 'rog', 'sam',
                                           # 'tou', 'val', 'wiz'),
                        # action='append')
    parser.add_argument('--role', default=None, type=str, nargs='+', help='keyword to search for in the first message to specify the role of the player')
    parser.add_argument('--panic-on-errors', default=True, action='store_true')

    args = parser.parse_args()

    print('ARGS:', args)
    return args

def main():
    args = parse_args()
    create_dataset(args)

if __name__ == '__main__':
    main()
