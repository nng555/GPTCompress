import numpy as np
import math
import torch
from collections import defaultdict

def pad_to_max(arr, pad_id, max_len=None):
    if type(arr[0]) in [np.ndarray, np.memmap]:
        if max_len is None:
            max_len = max([len(l) for l in arr])
        return np.stack([pad_to_max(l, pad_id, max_len) for l in arr])
    else:
        assert max_len is not None, "must provide max_len"
        return np.pad(arr, (0, max_len - len(arr)), 'constant', constant_values=pad_id)

def preprocess_lm(data, meta_specials):
    eos_id = meta_specials['eos']
    labels = data[:, 1:]
    labels = np.concatenate((labels, np.ones((len(labels), 1)) * eos_id), axis=-1)
    labels[(data == eos_id) & (labels == eos_id)] = -1
    for aidx in zip(*np.where(data == meta_specials['answer'])):
        labels[aidx[0], :aidx[1]] = -1
    return data, labels

def preprocess_length_transform(data, meta_specials, ntokens=None):
    eos_id = meta_specials['eos']
    doc_id = meta_specials['doc']

    full_idx = data
    full_labels = full_idx[:, 1:]
    full_labels = np.concatenate((full_labels, np.ones((len(full_labels), 1)) * meta_specials['eos']), axis=-1)
    full_labels[(full_idx == eos_id) | (full_labels == eos_id)] = -1
    teacher_masks = (full_labels != -1)

    enc_idxs = []
    lm_idxs = []
    lm_labels = []

    for pidx, aidx in zip(zip(*np.where(data == meta_specials['prompt'])), zip(*np.where(data == meta_specials['answer']))):
        enc_idxs.append(full_idx[pidx[0], :pidx[1]])
        lm_idxs.append(full_idx[pidx[0], pidx[1]:])
        lm_labels.append(full_labels[pidx[0], pidx[1]:])
        lm_labels[-1][0:aidx[1] - pidx[1]] = -1

    enc_idxs = pad_to_max(enc_idxs, eos_id)
    lm_idxs = pad_to_max(lm_idxs, eos_id)
    lm_labels = pad_to_max(lm_labels, -1)

    doc_idxs = np.ones((len(lm_idxs), ntokens)) * doc_id
    doc_idxs = np.concatenate((np.ones((len(lm_idxs), 1)) * eos_id, doc_idxs), -1)

    lm_idxs = np.concatenate((doc_idxs, lm_idxs), axis=-1)
    lm_labels = np.concatenate((np.ones_like(doc_idxs) * -1, lm_labels), axis=-1)

    return full_idx, teacher_masks, lm_idxs, lm_labels, enc_idxs,

def preprocess_gumbel(data, meta_specials):
    eos_id = meta_specials['eos']
    doc_id = meta_specials['doc']

    full_idx = data
    full_labels = full_idx[:, 1:]
    full_labels = np.concatenate((full_labels, np.ones((len(full_labels), 1)) * meta_specials['eos']), axis=-1)
    full_labels[(full_idx == eos_id) | (full_labels == eos_id)] = -1
    teacher_masks = (full_labels != -1)

    enc_idxs = []
    lm_idxs = np.copy(full_idx)

    for pidx, aidx in zip(zip(*np.where(data == meta_specials['prompt'])), zip(*np.where(data == meta_specials['answer']))):
        full_labels[aidx[0], :aidx[1]] = -1
        enc_idxs.append(full_idx[pidx[0], :pidx[1]])
        lm_idxs[pidx[0], 1:pidx[1]] = doc_id

    enc_idxs = pad_to_max(enc_idxs, eos_id)

    return full_idx, teacher_masks, lm_idxs, full_labels, enc_idxs,

def preprocess_enc(data, meta_specials):
    eos_id = meta_specials['eos']
    doc_id = meta_specials['doc']
    pad_id = meta_specials['pad']

    full_idx = data
    full_labels = full_idx[:, 1:]
    full_labels = np.concatenate((full_labels, np.ones((len(full_labels), 1)) * meta_specials['eos']), axis=-1)
    full_labels[(full_idx == eos_id) | (full_labels == eos_id)] = -1

    enc_idxs = []
    lm_idxs = []
    lm_labels = []
    teacher_masks = np.zeros_like(full_labels)

    for pidx, aidx in zip(zip(*np.where(data == meta_specials['prompt'])), zip(*np.where(data == meta_specials['answer']))):
        full_labels[aidx[0], :aidx[1]] = -1
        enc_idxs.append(full_idx[pidx[0], :pidx[1]])
        lm_idxs.append(full_idx[pidx[0], pidx[1]:])
        lm_labels.append(full_labels[pidx[0], pidx[1]:])

    teacher_masks = (full_labels != -1)
    enc_idxs = pad_to_max(enc_idxs, eos_id)
    lm_idxs = pad_to_max(lm_idxs, eos_id)
    lm_labels = pad_to_max(lm_labels, -1)

    doc_idxs = np.repeat(doc_idxs[None, :], len(full_idx), axis=0)
    enc_idxs = np.concatenate((doc_idxs, enc_idxs), axis=-1)
    lm_idxs = np.concatenate((np.ones_like(doc_idxs) * doc_id, lm_idxs), axis=-1)
    lm_labels = np.concatenate((np.ones_like(doc_idxs) * -1, lm_labels), axis=-1)

    return full_idx, teacher_masks, lm_idxs, lm_labels, enc_idxs

def blockify_lm(data, block_size, meta_specials):
    min_block_size = block_size

    blocks = []
    block_lens = []
    labels = []

    s_idx = 0
    eos_id = meta_specials['eos']
    pad_id = meta_specials['pad']
    while s_idx < len(data) - 1:

        # grab block while respecting example boundaries
        e_idx = min(s_idx + block_size, len(data) - 1)
        while data[e_idx] != eos_id:
            e_idx -= 1
        block = data[s_idx:e_idx + 1] # include EOS at end of block

        if len(block) < min_block_size:
            min_block_size = len(block)

        assert block[0] == eos_id and block[-1] == eos_id

        # pad block to full size
        # block looks like [EOS] x [EOS] x ... x [EOS] [EOS] ... [EOS]
        block = pad_to_max(block, pad_id, 1 + block_size)

        lens = []

        # grab answers as labels
        # ------ | ---- > [------- [EOS]]
        if 'answer' in meta_specials:
            ans_idxs = np.where(block == meta_specials['answer'])[0]
            eos_idxs = np.where(block == eos_id)[0][1:] # skip start EOS
            start_eidx = 0

            # set ignore indexes and labels
            label = np.ones_like(block) * -1
            for aidx, eidx in zip(ans_idxs, eos_idxs):
                label[aidx + 1:eidx + 1] = block[aidx + 1:eidx + 1]
                lens.append(eidx - start_eidx)
                start_eidx = eidx
            block_lens.append(lens)
        else:
            label = np.copy(block)
            label[block == pad_id] = -1

        # offset labels and inputs
        labels.append(label[1:])
        blocks.append(block[:-1])

        # update start index, should always be EOS
        s_idx = e_idx

    print(f"min block size: {min_block_size}/{block_size}")
    if 'answer' in meta_specials:
        return np.array(blocks, dtype=np.int16), np.array(labels, dtype=np.int16), block_lens
    else:
        return np.array(blocks, dtype=np.int16), np.array(labels, dtype=np.int16)

def blockify_fixed_enc(data, block_size, meta_specials, c_block_size=50, ntokens=0):
    min_block_size = block_size

    blocks = []
    enc_blocks = []
    lm_blocks = []
    teacher_masks = []
    lm_labels = []

    s_idx = 0
    eos_id = meta_specials['eos']
    doc_id = meta_specials['doc']
    pad_id = meta_specials['pad']

    while s_idx < len(data) - 1:

        # grab block while respecting example boundaries
        e_idx = min(s_idx + block_size - 1, len(data) - 1)
        while data[e_idx] != eos_id:
            e_idx -= 1
        block = data[s_idx:e_idx + 1] # include EOS at end of block
        block = pad_to_max(block, pad_id, 1 + block_size)

        teacher_mask = np.zeros_like(block)
        teacher_mask[c_block_size:] = 1
        teacher_mask[block == pad_id] = 0

        enc_block = block[:c_block_size]
        lm_block = np.copy(block)
        lm_block[:c_block_size] = doc_id

        lm_label = np.copy(lm_block)
        lm_label[:c_block_size] = -1
        lm_label[lm_label == pad_id] = -1

        if len(block) < min_block_size:
            min_block_size = len(block)

        # offset labels and inputs
        blocks.append(block[:-1])
        teacher_masks.append(teacher_mask[:-1])
        enc_blocks.append(enc_block)
        lm_blocks.append(lm_block[:-1])
        lm_labels.append(lm_label[1:])

        # update start index, should always be EOS
        s_idx = e_idx
        break

    print(f"min block size: {min_block_size}/{block_size}")
    return np.array(blocks, dtype=np.int16), np.array(teacher_masks, dtype=np.int16), np.array(lm_blocks, dtype=np.int16), \
             np.array(lm_labels, dtype=np.int16), np.array(enc_blocks, dtype=np.int16)


def blockify_enc(data, block_size, meta_specials, kernel_size=0, ntokens=0):

    assert not (kernel_size == 0 and ntokens == 0), "must set one of kernel_size or ntokens"
    min_block_size = block_size

    blocks = []
    block_lens = []
    label_masks = []
    lm_blocks = []
    lm_labels = []
    lm_lens = []
    enc_blocks = []
    enc_lens = []

    eos_id = meta_specials['eos']
    doc_id = meta_specials['doc']

    # if static compression length, pregenerate single or multiple doc idxs
    if kernel_size == 0:
        if 'doc0' not in meta_specials:
            doc_idxs = [doc_id]
        else:
            doc_idxs = [doc_id] + [f'doc{i}' for i in range(1, ntokens)]

    s_idx = 0
    while s_idx < len(data) - 1:

        # grab block while respecting example boundaries
        e_idx = min(s_idx + block_size - 1, len(data) - 1)
        while data[e_idx] != eos_id:
            e_idx -= 1
        block = data[s_idx:e_idx + 1]
        blocks.append(pad_to_max(block, eos_id, block_size))

        if len(block) < min_block_size:
            min_block_size = len(block)

        prompt_idxs = np.where(block == meta_specials['prompt'])[0]
        ans_idxs = np.where(block == meta_specials['answer'])[0]
        eos_idxs = np.where(block == eos_id)[0][1:] # skip start EOS
        start_eidx = 0

        # grab contexts for encoder and replace with doc
        # block: [EOS] ------ | ---- > -------
        # lm:    [EOS] D | ---- > -------
        # enc:   [EOS] D --------
        label_mask = np.zeros(block_size)
        extra_blocks = defaultdict(list)
        for pidx, aidx, eidx in zip(prompt_idxs, ans_idxs, eos_idxs):
            extra_blocks['block_len'].append(eidx - start_eidx)
            label_mask[aidx + 1:eidx] = True

            # if convolving, generate dynamic doc idxs
            if kernel_size != 0:
                nids = math.ceil((pidx - start_eidx) / kernel_size)
                doc_idxs = [doc_id] * nids
            else:
                nids = ntokens

            extra_blocks['lm_block'].append(
                np.concatenate((
                    np.array([eos_id] + doc_idxs),
                    block[pidx:eidx],
                ))
            )
            extra_blocks['lm_label'].append(
                np.concatenate((
                    np.ones(1 + nids + (aidx - pidx)) * -1,
                    block[aidx + 1:eidx + 1],
                ))
            )
            extra_blocks['lm_len'].append(1 + nids + (eidx - pidx))

            # no extra doc tokens if convolving
            if kernel_size != 0:
                enc_block = np.concatenate((
                    block[start_eidx:pidx],
                ))
                pad_len = nids * kernel_size
                # pad to kernel_size to align embeddings
                enc_block = pad_to_max(enc_block, eos_id, pad_len)
                extra_blocks['enc_len'].append(pad_len)
            # replace EOS with DOC if no convolution
            else:
                enc_block = np.concatenate((
                    np.array(doc_idxs),
                    block[start_eidx + 1:pidx],
                ))
                extra_blocks['enc_len'].append(pidx - (start_eidx + 1) + nids)

            extra_blocks['enc_block'].append(enc_block)
            start_eidx = eidx

        label_masks.append(label_mask)
        lm_blocks.append(pad_to_max(np.concatenate(extra_blocks['lm_block']), eos_id, block_size))
        lm_labels.append(pad_to_max(np.concatenate(extra_blocks['lm_label']), -1, block_size))
        enc_blocks.append(pad_to_max(np.concatenate(extra_blocks['enc_block']), eos_id, block_size))
        enc_lens.append(extra_blocks['enc_len'])
        lm_lens.append(extra_blocks['lm_len'])
        block_lens.append(extra_blocks['block_len'])
        s_idx = e_idx

    print(f"min block size: {min_block_size}/{block_size}")
    return np.array(blocks, dtype=np.int16), \
           np.array(label_masks, dtype=np.int16), \
           np.array(lm_blocks, dtype=np.int16), \
           np.array(lm_labels, dtype=np.int16), \
           np.array(enc_blocks, dtype=np.int16), \
           enc_lens, \
           lm_lens, \
           block_lens, # ragged list so can't convert

def blockify(data, block_size, meta_specials, model_type="lm", c_block_size=0, ntokens=0):
    if model_type == "lm":
        return blockify_lm(data, block_size, meta_specials)
    else:
        return blockify_fixed_enc(data, block_size, meta_specials, c_block_size=c_block_size, ntokens=ntokens)

def preprocess(data, meta_specials, model_type="lm", kernel_size=0, ntokens=0):
    if model_type == "lm":
        return preprocess_lm(data, meta_specials)
    else:
        return preprocess_length_transform(data, meta_specials, ntokens=ntokens)

def make_attn_mask(seq_lens, block_size, causal=False):
    total_len = np.sum(seq_lens)
    assert total_len <= block_size, "total length must be less than block size"
    blocks = [torch.ones((slen, slen)) for slen in seq_lens]
    if causal:
        blocks = [torch.tril(block) for block in blocks]
    if total_len < block_size:
        blocks.append(torch.zeros((block_size - total_len, block_size - total_len)))
    attn_mask = torch.block_diag(*blocks)
    return attn_mask

def send_to_device(device, device_type, *args):
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        return [a.pin_memory().to(device, non_blocking=True) for a in args]
    else:
        return [a.pin_memory().to(device) for a in args]

# get random batch of blocks from dataset
def get_batch(data, batch_size, device, device_type, model_type="lm"):

    # build standard batch for LM
    if model_type == "lm":
        if len(data) == 3:
            xs, ys, x_lens = data
        else:
            xs, ys = data
            x_lens = None

        ix = torch.randint(len(xs), (batch_size,))

        x = torch.from_numpy(xs[ix].astype(np.int64))
        y = torch.from_numpy(ys[ix].astype(np.int64))

        if x_lens is not None:
            x_lens = [x_lens[i] for i in ix]
            block_size = x.shape[-1]
            x_mask = torch.stack([make_attn_mask(lens, block_size, causal=True) for lens in x_lens])

            x, y, x_mask = send_to_device(device, device_type, x, y, x_mask)

            return {
                'idx': x,
                'targets': y,
                'attn_mask': x_mask
            }
        else:
            x, y = send_to_device(device, device_type, x, y)
            return {
                'idx': x,
                'targets': y,
            }

    # build batch for compression
    else:
        if len(data) == 5:
            xs, y_masks, lm_xs, lm_ys, enc_xs = data
        else:
            xs, y_masks, lm_xs, lm_ys, enc_xs, enc_lens, lm_lens, x_lens = data
        ix = torch.randint(len(lm_xs), (batch_size,))

        x = torch.from_numpy(xs[ix].astype(np.int64))
        y_mask = torch.from_numpy(y_masks[ix].astype(np.int64))
        lm_x = torch.from_numpy(lm_xs[ix].astype(np.int64))
        lm_y = torch.from_numpy(lm_ys[ix].astype(np.int64))
        enc_x = torch.from_numpy(enc_xs[ix].astype(np.int64))

        if len(data) != 5:
            enc_ls = [enc_lens[i] for i in ix]
            lm_ls = [lm_lens[i] for i in ix]
            x_ls = [x_lens[i] for i in ix]

            block_size = lm_x.shape[-1]
            enc_masks = torch.stack([make_attn_mask(lens, block_size) for lens in enc_ls])
            lm_masks = torch.stack([make_attn_mask(lens, block_size, causal=True) for lens in lm_ls])
            x_masks = torch.stack([make_attn_mask(lens, block_size, causal=True) for lens in x_ls])

            x, y_mask, lm_x, lm_y, enc_x, enc_masks, lm_masks, x_masks = \
                    send_to_device(device, device_type, x, y_mask, lm_x, lm_y, enc_x, enc_masks, lm_masks, x_masks)

            return {
                'idx': x,
                'teacher_mask': y_mask,
                'lm_idx': lm_x,
                'enc_idx': enc_x,
                'targets': lm_y,
                'enc_attn_mask': enc_masks,
                'lm_attn_mask': lm_masks,
                'full_mask': x_masks,
            }
        else:
            x, y_mask, lm_x, lm_y, enc_x = \
                    send_to_device(device, device_type, x, y_mask, lm_x, lm_y, enc_x)

            return {
                'idx': x,
                'teacher_mask': y_mask,
                'lm_idx': lm_x,
                'enc_idx': enc_x,
                'targets': lm_y,
            }
