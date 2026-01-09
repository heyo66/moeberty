# You can start ddp train with pytorch==2.20
# pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0

# torchrun --nproc_per_node=4 ddp_train.py


from model import Transformer, ModelConfig
from trainer import Trainer, TrainerConfig, DataLoader

from transformers import AutoTokenizer
import torch
from torch.distributed import init_process_group, destroy_process_group

import os

torch.cuda.empty_cache()

tokenizer_id = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
tokenizer.pad_token = tokenizer.eos_token


train_config = TrainerConfig(
    vocab_size = tokenizer.vocab_size,
    num_epochs = 1,

    use_ddp = True,
    use_moe = False,
    use_lossfreebalance = False,
    clean_cuda_cache = True,
    use_compile = False,
    use_dtype = "float16",

    seed = 1,
    max_seq_len = 1024,
    batch_size = 8,
    accumulation_steps = 4,
    
    weight_decay = 0.1,
    warmup_ratio = 0.01,
    learning_rate = 1e-3,
    betas = (0.90, 0.95),
    update_rate = 1e-5,

    val_ratio = 0.005,
    steps_for_eval = 20,
    eval_interval = 100,

    checkpoints_frequency = 1000,
    path_to_checkpoints = "./model_testing",

    tokenized_dataset_path = "",
    hf_dataset_name = "",
    hf_dataset_split = "train",
    hf_text_field = "",
    hf_add_eos = True,
    hf_cache_dir = "./.cache/hf",
    hf_num_proc = 4,
    eval_log_file = "log/eval.txt",
    use_wandb = True,
    wandb_project = "forschungsprojekt",
    wandb_run_name = "smollm-ddp",
)

config = ModelConfig(
        vocab_size = tokenizer.vocab_size,

        num_dims = 768,
        num_heads = 16,
        num_kv_heads = 16,
        num_layers = 30,
        ffn_hidden_dims = 1024,

        rmsnorm_eps = 1e-6,
        rope_theta = 1e5,

        attention_probs_dropout_prob = 0.0,
        attn_qkv_bias = False,
        attn_out_bias = False,
        attn_out_dropout_prob = 0.0,
    
        context_len = 2048,
        
        use_cache = False,
        use_flash = True,
        use_moe = False,

        moe_num_experts = 4,
        moe_active_experts = 4,
        moe_eps = 1e-6,
        moe_aux_loss_coef = 0.01,
        moe_shared_experts = 1,
        use_lossfreebalance = False,
    )

init_process_group("nccl")

model = Transformer(config)

data_loader = DataLoader(
    train_config,
    tokenizer=tokenizer,
    rank=int(os.environ['RANK']),
    world_size=int(os.environ['WORLD_SIZE']),
)


trainer = Trainer(train_config, model, tokenizer)
trainer.train(data_loader)

destroy_process_group()
