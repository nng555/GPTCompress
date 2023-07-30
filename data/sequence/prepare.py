import argparse
import pickle
import pathlib
import os
import requests
import numpy as np
import scipy.stats as ss

TOKENS = [str(c) for c in range(10)]
SPECIAL_MAP = {
    'prompt': '|',
    'answer': '>',
    'sep': ',',
    'eos': '\n',
    'doc': 'D',
    'pad': '\n'
}
VOCAB = TOKENS + list(set(SPECIAL_MAP.values()))
OTOS = {
    'copy': ['0', 'gen_random'],
    'scopy': ['1', 'gen_random'],
    'sum_mod': ['2', 'gen_random'],
    'add': ['3', 'gen_num'],
}
STOI = {c: i for i, c in enumerate(VOCAB)}
ITOS = {i: c for i, c in enumerate(VOCAB)}

def pad_to_max(arr, pad_id, max_len):
    return np.pad(arr, (0, max_len - len(arr)), 'constant', constant_values=pad_id)

def encode(s_list):
    if isinstance(s_list, str):
        return [STOI[c] for c in s_list]
    else:
        return [encode(s) for s in s_list]

def decode(i_list):
    if isinstance(i_list[0], int):
        return ''.join([ITOS[i] for i in i_list])
    else:
        return [decode(i) for i in i_list]

def ltoi(l):
    return int(ltos(l))

def ltos(l):
    return ''.join([str(v) for v in l])

def gen_random():
    raise NotImplementedError

def gen_num():
    seq_len = 10
    start = np.random.choice(np.arange(1, 10))
    seq = np.random.choice(np.arange(10), size=seq_len-1)
    seq = np.insert(seq, 0, start)
    return seq


def add(seq_fn):
    s1 = seq_fn()
    s2 = seq_fn()

    s1 = ltos(s1)
    s2 = ltos(s2)
    ans = int(s1) + int(s2)
    return s1 + ',' + s2, str(ans), None

def copy(seq_fn):
    s = ''.join([str(v) for v in seq_fn()])
    return s, s, None

def scopy(seq_fn, stride=None, offset=None):
    s = seq_fn()
    if stride is None:
        stride = np.random.randint(2, 10)
    if offset is None:
        offset = np.random.randint(10)
    s = ''.join([str(v) for v in s])
    ans = ''.join([str(v) for v in s[offset::stride]])
    return s, ans, f'{stride}{offset}'


def generate(nexamples, op, **kwargs):

    assert op in OTOS, f"Operation {op} not supported"
    assert op in globals(), f"Operation {op} not defined"
    op_fn = globals()[op]
    op_name, seq_fn = OTOS[op]
    seq_fn = globals()[seq_fn]

    task_name = op
    for k, v in kwargs.items():
        task_name += f'_{k}_{v}'
    data_dir = os.path.join(os.path.dirname(__file__), task_name)
    pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)

    """
    # markov chain generation
    x = np.arange(-5, 5)
    xU, xL = x + 0.5, x - 0.5
    prob = ss.norm.cdf(xU, scale=1.5) - ss.norm.cdf(xL, scale=1.5)
    prob = prob / prob.sum()
    """

    # generate raw data
    data = []
    for i in range(nexamples):

        # TODO: set this from command line
        #seq_len = np.random.randint(5, 30)
        #start = np.random.choice(TOKENS)
        #seq = np.random.choice(x, size=seq_len-1, p=prob)
        #seq = np.insert(seq, 0, start)
        #seq = np.cumsum(seq) % 10
        seq, ans, suf = op_fn(seq_fn, **kwargs)
        prompt = op_name + ('' if suf is None else suf)

        example = SPECIAL_MAP['eos'] + seq + SPECIAL_MAP['prompt'] + prompt + \
            SPECIAL_MAP['answer'] + ans + SPECIAL_MAP['eos']
        data.append(example)

    # write raw data
    with open(os.path.join(data_dir, 'raw.txt'), 'w') as of:
        for d in data:
            of.write(d)

    # generate splits
    train_data = data[:int(nexamples * 0.9)]
    val_data = data[int(nexamples * 0.9):]

    # encode
    train_ids = encode(train_data)
    val_ids = encode(val_data)

    # stack examples
    max_train_len = max([len(ids) for ids in train_ids])
    max_val_len = max([len(ids) for ids in val_ids])

    train_ids = np.stack([pad_to_max(ids, STOI[SPECIAL_MAP['eos']], max_train_len) for ids in train_ids])
    val_ids = np.stack([pad_to_max(ids, STOI[SPECIAL_MAP['eos']], max_val_len) for ids in val_ids])

    train_ids = train_ids.astype(np.int16)
    val_ids = val_ids.astype(np.int16)

    """
    # add initial eos token
    train_ids[0] = [STOI[SPECIAL_MAP['eos']]] + train_ids[0]
    val_ids[0] = [STOI[SPECIAL_MAP['eos']]] + val_ids[0]

    # flatten and export to bin
    train_ids = np.array([l for sublist in train_ids for l in sublist], dtype=np.int16)
    val_ids = np.array([l for sublist in val_ids for l in sublist], dtype=np.int16)
    """

    train_ids.tofile(os.path.join(data_dir, 'train.bin'))
    val_ids.tofile(os.path.join(data_dir, 'val.bin'))

    # write metadata
    meta = {
        'vocab_size': len(VOCAB),
        'itos': ITOS,
        'stoi': STOI,
        'specials': SPECIAL_MAP,
        'ntrain': len(train_ids),
        'nval': len(val_ids),
    }
    with open(os.path.join(data_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nexamples', type=int, help='# of examples')
    parser.add_argument('-o', '--op', type=str, default='odd', help='operation to generate')
    args, unknown = parser.parse_known_args()

    for arg in unknown:
        if arg.startswith(("-", "--")):
            # you can pass any arguments to add_argument
            parser.add_argument(arg.split('=')[0], type=int)

    args = parser.parse_args()
    generate(**args.__dict__)
