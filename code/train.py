# python train.py

from model import Transformer, ModelConfig
from trainer import Trainer, TrainerConfig, DataLoader

from transformers import AutoTokenizer
import torch
import torch.distributed as dist
import argparse
import os

torch.set_float32_matmul_precision('high')
torch.cuda.empty_cache()

tokenizer_id = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
tokenizer.pad_token = tokenizer.eos_token

checkpoint_path = './model_testing'
continue_train = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_distributed():
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in this build.")
    if dist.is_initialized():
        return
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ddp", action="store_true", help="Enable DDP (launch with torchrun).")
    return parser.parse_args()

args = parse_args()

train_config = TrainerConfig(
    vocab_size = 50368,
    num_epochs = 10,

    use_ddp = args.ddp,
    use_moe = True,
    use_lossfreebalance = True,
    clean_cuda_cache = True,
    use_compile = True,
    use_dtype = "bfloat16",

    seed = 42,
    max_seq_len = 128,
    batch_size = 512,
    accumulation_steps = 1,
    
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    learning_rate = 4e-4,
    betas = (0.90, 0.97),
    update_rate = 5e-6,

    val_ratio = 0.005,
    steps_for_eval = 20,
    eval_interval = 20,
    
    mlm_probability = 0.30,

    checkpoints_frequency = 1000,
    path_to_checkpoints = "./model_testing",

    tokenized_dataset_path = "",
    hf_dataset_name = "allenai/c4",
    hf_dataset_config = "en",
    hf_dataset_split = "train",
    hf_text_field = "text",
    hf_add_eos = False,
    hf_cache_dir = "./.cache/hf",
    hf_tokenized_path = "./.cache/tokenized",
    hf_num_proc = 64,
    eval_log_file = "log/eval_c4.txt",
    use_wandb = True,
    wandb_project = "forschungsprojekt",
    wandb_run_name = "moebert",
    log_mlm_accuracy = True,
    mlm_accuracy_interval = 100,
    log_expert_stats = True,
    expert_stats_interval = 100,
)

config = ModelConfig(
        vocab_size = 50368,

        num_dims = 768,
        num_heads = 12,
        num_kv_heads = 12,
        num_layers = 22,
        ffn_hidden_dims = 1152,

        layernorm_eps=1e-6,

        attention_probs_dropout_prob = 0.1,
        attn_qkv_bias = False,
        attn_out_bias = False,
        attn_out_dropout_prob = 0.0,
        global_attn_every_n_layers = 3,
        sliding_window = 128,
        rotary_emb_base = 10000,
    
        context_len = 128,
        
        use_cache = False,
        use_flash = True,
        use_moe = True,

        moe_num_experts = 16,
        moe_routed_experts = 1,
        moe_eps = 1e-6,
        moe_aux_loss_coef = 0.01,
        moe_shared_experts = 1,
        use_lossfreebalance = True,
)


model = Transformer(config)
if continue_train:
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    state_dict = checkpoint['model']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k[len("_orig_mod."):]] = v 
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)

rank = 0
world_size = 1
if args.ddp:
    setup_distributed()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

if not args.ddp:
    model.to(device)

data_loader = DataLoader(train_config, tokenizer=tokenizer, rank=rank, world_size=world_size)
trainer = Trainer(train_config, model, tokenizer)
trainer.train(data_loader)

cleanup_distributed()
