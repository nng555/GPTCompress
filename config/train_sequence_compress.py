# train on toy sequence dataset

out_dir = 'out-sequence'
model_type = 'compress'
ckpt_name = 'ckpt_enc.pt'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'shakespeare-char'
wandb_run_name = 'mini-gpt'

dataset = 'sequence'
tasks = ['odd']
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 4
n_head = 8
n_embd = 64
dropout = 0.0

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 50000
lr_decay_iters = 50000 # make equal to max_iters usually
weight_decay = 1
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# on macbook also add
#device = 'cpu'  # run on cpu only
#compile = False # do not torch compile the model

soft_loss_weight = 0.0
hard_loss_weight = 1.0
kernel_size = 2
