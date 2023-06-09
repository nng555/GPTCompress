import numpy as np
import torch
from collections import defaultdict

def pad_to_max(arr, pad_id, max_len):
    return np.pad(arr, (0, max_len - len(arr)), 'constant', constant_values=pad_id)

def blockify_lm(data, block_size, meta_specials):
    min_block_size = block_size

    blocks = []
    labels = []

    s_idx = 0
    eos_id = meta_specials['eos']
    while s_idx < len(data) - 1:

        # grab block while respecting example boundaries
        e_idx = min(s_idx + block_size, len(data) - 1)
        while data[e_idx] != eos_id:
            e_idx -= 1
        block = data[s_idx:e_idx + 1]

        if len(block) < min_block_size:
            min_block_size = len(block)

        assert block[0] == eos_id and block[-1] == eos_id

        # pad block to full size
        # block looks like [EOS] x [EOS] x ... x [EOS] [EOS] ... [EOS]
        block = pad_to_max(block, eos_id, 1 + block_size)

        # grab answers as labels
        # ------ | ---- > [------- [EOS]]
        ans_idxs = np.where(block == meta_specials['answer'])[0]
        eos_idxs = np.where(block == eos_id)[0][1:] # skip start EOS

        # set ignore indexes and labels
        label = np.ones_like(block) * -1
        for aidx, eidx in zip(ans_idxs, eos_idxs):
            label[aidx + 1:eidx + 1] = block[aidx + 1:eidx + 1]

        # offset labels and inputs
        labels.append(label[1:])
        blocks.append(block[:-1])

        # update start index, should always be EOS
        s_idx = e_idx

    print(f"min block size: {min_block_size}/{block_size}")
    return np.array(blocks, dtype=np.int16), np.array(labels, dtype=np.int16)

def blockify_enc(data):

    blocks = []
    label_masks = []
    lm_blocks = []
    lm_labels = []
    enc_blocks = []
    enc_lens = []

    eos_id = meta_specials['eos']
    doc_id = meta_specials['doc']

    # grab contexts for encoder and replace with doc
    # [EOS] [------] | ---- > -------
    # [EOS] D | ---- > -------
    prompt_idxs = np.where(block == meta_specials['prompt'])[0]
    eos_idxs = np.where(block == eos_id)[0][1:] # skip first EOS

    while s_idx < len(data) - 1:

        # grab block while respecting example boundaries
        e_idx = min(s_idx + block_size, len(data) - 1)
        while data[e_idx] != eos_id:
            e_idx -= 1
        block = data[s_idx:e_idx + 1]
        full_blocks.append(pad_to_max(block, eos_id, block_size))

        if len(block) < min_block_size:
            min_block_size = len(block)

        prompt_idxs = np.where(block == meta_specials['prompt'])[0]
        ans_idxs = np.where(block == meta_specials['answer'])[0]
        eos_idxs = np.where(block == eos_id)[0][1:] # skip start EOS
        start_eidx = 0

        label_mask = np.zeros(block_size)
        extra_blocks = defaultdict(list)
        for pidx, aidx, eidx in zip(prompt_idxs, ans_idxs, eos_idxs):
            label_mask[aidx:eidx] = True
            extra_blocks['lm_block'].append(
                np.concatenate(
                    np.array([eos_id, doc_id]),
                    block[pidx:eidx],
                )
            )
            extra_blocks['lm_label'].append(
                np.concatenate(
                    np.ones(2 + (aidx - pidx)) * -1,
                    block[aidx:eidx],
                )
            )
            extra_blocks['enc_block'].append(
                np.concatenate(
                    np.array([eos_id, doc_id]),
                    block[start_eidx + 1:p_idx],
                )
            )
            extra_blocks['enc_len'].append(p_idx - start_eidx)
            start_eidx = eidx

        label_masks.append(label_mask)
        lm_blocks.append(pad_to_max(np.concatenate(extra_blocks['lm_block']), eos_id, block_size))
        lm_labels.append(pad_to_max(np.concatenate(extra_blocks['lm_label'])[1:], eos_id, block_size))
        enc_blocks.append(pad_to_max(np.concatenate(extra_blocks['enc_block']), eos_id, block_size))
        enc_lens.append(extra_blocks['enc_len'])

    return np.array(full_blocks, dtype=np.int16), \
           np.array(label_masks, dtype=np.int16), \
           np.array(lm_blocks, dtype=np.int16), \
           np.array(lm_labels, dtype=np.int16), \
           np.array(enc_blocks, dtype=np.int16), \
           enc_lens # ragged list so can't convert

def blockify(data, block_size, meta_specials, model_type="lm"):
    if model_type == "lm":
        return blockify_lm(data, block_size, meta_specials)
    else:
        return blockify_enc(data, block_size, meta_specials)

def make_attn_mask(seq_lens, block_size):
    total_len = np.sum(seq_lens)
    assert total_len < block_size, "total length must be less than block size"
    blocks = [torch.ones((slen, slen)) for slen in seq_lens]
    if total_len < block_size:
        blocks.append(torch.ones((block_size - total_len, block_size - total_len)))
    attn_mask = torch.block_diag(*blocks)
    return attn_mask

def send_to_device(device, device_type, *args):
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        return [a.pin_memory().to(device, non_blocking=True) for a in args]
    else:
        return [a.pinmemory().to(device) for a in args]

# get random batch of blocks from dataset
def get_batch(data, batch_size, device, device_type, model_type="lm"):

    # build standard batch for LM
    if model_type == "lm":
        xs, ys = data
        ix = torch.randint(len(xs), (batch_size,))

        x = torch.from_numpy(xs[ix].astype(np.int64))
        y = torch.from_numpy(ys[ix].astype(np.int64))

        x, y = send_to_device(device, device_type, x, y)

        return {
            'idx': x,
            'targets': y,
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

        block_size = lm_x.shape[-1]
        enc_masks = torch.stack([make_attn_mask(lens, block_size) for lens in enc_lens])

        lm_x, lm_y, enc_x, enc_masks = send_to_device(device, device_type, lm_x, lm_y, enc_x, enc_masks)

        return {
            'idx': x,
            'teacher_mask': y_mask,
            'lm_idx': lm_x,
            'enc_idx': enc_x,
            'targets': lm_y,
            'attn_mask': enc_masks,
        }

