import argparse
import pickle
import pathlib
import os
import requests
import numpy as np

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
    'odd': '1',
    'even': '2',
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

def copy(s):
    return s

def odd(s):
    return s[::2]

def even(s):
    return s[1::2]

def generate(nexamples, op):

    assert op in OTOS, f"Operation {op} not supported"
    assert op in globals(), f"Operation {op} not defined"
    op_fn = globals()[op]

    data_dir = os.path.join(os.path.dirname(__file__), op)
    pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)

    # generate raw data
    data = []
    for i in range(nexamples):

        # TODO: set this from command line
        seq_len = np.random.randint(2, 30)

        seq = ''.join(np.random.choice(TOKENS, size=seq_len))
        example = seq + SPECIAL_MAP['prompt'] + OTOS[op] + SPECIAL_MAP['answer'] + op_fn(seq) + SPECIAL_MAP['eos']
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
    args = parser.parse_args()
    generate(args.nexamples, args.op)
