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
    'doc': 'D'
}
VOCAB = TOKENS + list(SPECIAL_MAP.values())
OTOS = {
    'copy': '0',
    'scopy': '1',
    'sum_mod': '2',
    'add': '3',
}
STOI = {c: i for i, c in enumerate(VOCAB)}
ITOS = {i: c for i, c in enumerate(VOCAB)}

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

def add(s1, s2):
    s1 = int(''.join(s1))
    s2 = int(''.join(s2))
    return [int(v) for v in str(s1 + s2)], '2'

def sum_mod(s, n=None, mod=None):
    if n is None:
        n = np.random.randint(2, 10)
    if mod is None:
        mod = np.random.randint(2, 11)

    ans = []
    i = 0
    while(i < len(s)):
        ans.append(np.sum(s[i:i+n]) % mod)
        i += n
    if mod == 10:
        mod = 0
    return ans, f'{n}{mod}'

def copy(s):
    return s, None

def scopy(s, stride=None, offset=None):
    if stride is None:
        stride = np.random.randint(2, 10)
    if offset is None:
        offset = np.random.randint(10)
    return s[offset::stride], f'{stride}{offset}'

def generate(nexamples, op, **kwargs):

    assert op in OTOS, f"Operation {op} not supported"
    assert op in globals(), f"Operation {op} not defined"
    op_fn = globals()[op]

    task_name = op
    for k, v in kwargs.items():
        task_name += f'_{k}_{v}'
    data_dir = os.path.join(os.path.dirname(__file__), task_name)
    pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)

    x = np.arange(-5, 5)
    xU, xL = x + 0.5, x - 0.5
    prob = ss.norm.cdf(xU, scale=1.5) - ss.norm.cdf(xL, scale=1.5)
    prob = prob / prob.sum()

    # generate raw data
    data = []
    for i in range(nexamples):

        # TODO: set this from command line
        seq_len = np.random.randint(5, 30)
        start = np.random.choice(TOKENS)
        seq = np.random.choice(x, size=seq_len-1, p=prob)
        seq = np.insert(seq, 0, start)
        seq = np.cumsum(seq) % 10

        ans, suf = op_fn(seq, **kwargs)
        ans = ''.join([str(v) for v in ans])
        seq = ''.join([str(v) for v in seq])

        op_name = OTOS[op] + ('' if suf is None else suf)

        example = ''.join(seq) + SPECIAL_MAP['prompt'] + op_name + \
            SPECIAL_MAP['answer'] + ''.join(ans) + SPECIAL_MAP['eos']
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

    # add initial eos token
    train_ids[0] = [STOI[SPECIAL_MAP['eos']]] + train_ids[0]
    val_ids[0] = [STOI[SPECIAL_MAP['eos']]] + val_ids[0]

    # flatten and export to bin
    train_ids = np.array([l for sublist in train_ids for l in sublist], dtype=np.int16)
    val_ids = np.array([l for sublist in val_ids for l in sublist], dtype=np.int16)
    train_ids.tofile(os.path.join(data_dir, 'train.bin'))
    val_ids.tofile(os.path.join(data_dir, 'val.bin'))

    # write metadata
    meta = {
        'vocab_size': len(VOCAB),
        'itos': ITOS,
        'stoi': STOI,
        'specials': SPECIAL_MAP,
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
