import numpy as np
import math
import torch
from collections import defaultdict

def pad_to_max(arr, pad_id, max_len):
    return np.pad(arr, (0, max_len - len(arr)), 'constant', constant_values=pad_id)

def blockify_lm(data, block_size, meta_specials):
    min_block_size = block_size

    blocks = []
    block_lens = []
    labels = []

    s_idx = 0
    eos_id = meta_specials['eos']
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
        block = pad_to_max(block, eos_id, 1 + block_size)

        lens = []

        # grab answers as labels
        # ------ | ---- > [------- [EOS]]
        ans_idxs = np.where(block == meta_specials['answer'])[0]
        eos_idxs = np.where(block == eos_id)[0][1:] # skip start EOS
        start_eidx = 0

        # set ignore indexes and labels
        label = np.ones_like(block) * -1
        for aidx, eidx in zip(ans_idxs, eos_idxs):
            label[aidx + 1:eidx + 1] = block[aidx + 1:eidx + 1]
            lens.append(eidx - start_eidx)
            start_eidx = eidx

        # offset labels and inputs
        labels.append(label[1:])
        blocks.append(block[:-1])
        block_lens.append(lens)

        # update start index, should always be EOS
        s_idx = e_idx

    print(f"min block size: {min_block_size}/{block_size}")
    return np.array(blocks, dtype=np.int16), np.array(labels, dtype=np.int16), block_lens

def blockify_enc(data, block_size, meta_specials, kernel_size=-1):
    min_block_size = block_size

    blocks = []
    label_masks = []
    lm_blocks = []
    lm_labels = []
    enc_blocks = []
    enc_lens = []

    eos_id = meta_specials['eos']
    doc_id = meta_specials['doc']

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
            label_mask[aidx + 1:eidx] = True

            if kernel_size != -1:
                nids = math.ceil((pidx - start_eidx) / kernel_size)
            #  1 docid by default
            else:
                nids = 1

            # TODO: causal block attn mask
            extra_blocks['lm_block'].append(
                np.concatenate((
                    np.array([eos_id] + [doc_id] * nids),
                    block[pidx:eidx],
                ))
            )
            extra_blocks['lm_label'].append(
                np.concatenate((
                    np.ones(1 + nids + (aidx - pidx)) * -1,
                    block[aidx + 1:eidx + 1],
                ))
            )

            # no extra doc tokens if convolving
            if kernel_size != -1:
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
                    np.array([doc_id]),
                    block[start_eidx + 1:pidx],
                ))
                extra_blocks['enc_len'].append(pidx - start_eidx)

            extra_blocks['enc_block'].append(enc_block)
            start_eidx = eidx

        label_masks.append(label_mask)
        lm_blocks.append(pad_to_max(np.concatenate(extra_blocks['lm_block']), eos_id, block_size))
        lm_labels.append(pad_to_max(np.concatenate(extra_blocks['lm_label']), -1, block_size))
        enc_blocks.append(pad_to_max(np.concatenate(extra_blocks['enc_block']), eos_id, block_size))
        enc_lens.append(extra_blocks['enc_len'])
        s_idx = e_idx

    print(f"min block size: {min_block_size}/{block_size}")
    return np.array(blocks, dtype=np.int16), \
           np.array(label_masks, dtype=np.int16), \
           np.array(lm_blocks, dtype=np.int16), \
           np.array(lm_labels, dtype=np.int16), \
           np.array(enc_blocks, dtype=np.int16), \
           enc_lens # ragged list so can't convert

def blockify(data, block_size, meta_specials, model_type="lm", kernel_size=-1):
    if model_type == "lm":
        return blockify_lm(data, block_size, meta_specials)
    else:
        return blockify_enc(data, block_size, meta_specials, kernel_size=kernel_size)

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
        xs, ys, x_lens = data
        ix = torch.randint(len(xs), (batch_size,))

        x = torch.from_numpy(xs[ix].astype(np.int64))
        y = torch.from_numpy(ys[ix].astype(np.int64))
        x_lens = [x_lens[i] for i in ix]

        block_size = x.shape[-1]
        x_mask = torch.stack([make_attn_mask(lens, block_size, causal=True) for lens in x_lens])

        x, y, x_mask = send_to_device(device, device_type, x, y, x_mask)

        return {
            'idx': x,
            'targets': y,
            'attn_mask': x_mask
        }

    # build batch for compression
    else:
        xs, y_masks, lm_xs, lm_ys, enc_xs, enc_lens = data
        ix = torch.randint(len(lm_xs), (batch_size,))

        x = torch.from_numpy(xs[ix].astype(np.int64))
        y_mask = torch.from_numpy(y_masks[ix].astype(np.int64))
        lm_x = torch.from_numpy(lm_xs[ix].astype(np.int64))
        lm_y = torch.from_numpy(lm_ys[ix].astype(np.int64))
        enc_x = torch.from_numpy(enc_xs[ix].astype(np.int64))
        enc_ls = [enc_lens[i] for i in ix]

        block_size = lm_x.shape[-1]
        enc_masks = torch.stack([make_attn_mask(lens, block_size) for lens in enc_ls])

        x, y_mask, lm_x, lm_y, enc_x, enc_masks = send_to_device(device, device_type, x, y_mask, lm_x, lm_y, enc_x, enc_masks)

        return {
            'idx': x,
            'teacher_mask': y_mask,
            'lm_idx': lm_x,
            'enc_idx': enc_x,
            'targets': lm_y,
            'attn_mask': enc_masks,
        }

