import time
import math
import os
from contextlib import nullcontext
from dataclasses import dataclass, asdict
from typing import Tuple

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from datatrove.utils.dataset import DatatroveFolderDataset
except ImportError:
    DatatroveFolderDataset = None
try:
    import wandb
except ImportError:
    wandb = None

@dataclass
class TrainerConfig:
    vocab_size: int                 
    num_epochs: int                 

    use_ddp: bool                   
    use_moe: bool                   # enable mixture-of-experts
    use_lossfreebalance: bool       # use Auxiliary-loss-free load balancing strategy for mixture-of-experts from DeepSeek https://arxiv.org/pdf/2408.15664
    clean_cuda_cache: bool = True   # Helps prevent OOM errors during eval on large models
    use_compile: bool = True        # use torch.compile()
    use_dtype: str = "bfloat16"

    seed: int = 1998                
    max_seq_len: int = 1024         # maximum context length for batch
    batch_size: int = 1             # numbe of batches
    accumulation_steps: int = 1
    
    # Optimizer parameters
    weight_decay: float = 0.1
    warmup_ratio: float = 0.01
    learning_rate: float = 1e-3
    betas: Tuple[float, float] = (0.90, 0.95)
    update_rate: float = 1e-5  # update_rate of biases for loss-free balancing

    val_ratio: int = 0.005
    steps_for_eval: int = 20                            # number of steps for evaluation
    eval_interval: int = 50

    checkpoints_frequency: int = 500
    path_to_checkpoints: str = "./model_testing"        # path to directory to save checkpoints

    tokenized_dataset_path: str = ""                    # path to directory with tokeized dataset
    hf_dataset_name: str = ""                           # huggingface dataset name (e.g. "HuggingFaceTB/cosmopedia")
    hf_dataset_config: str = "stanford"                         # optional hf dataset config name
    hf_dataset_split: str = "train"
    hf_text_field: str = ""                             # text field name in hf dataset
    hf_add_eos: bool = True
    hf_cache_dir: str = ""                              # optional cache dir
    hf_num_proc: int = 1                                # map workers for tokenization
    hf_tokenized_path: str = ""                         # optional path to save/load tokenized dataset
    eval_log_file: str = "logs/eval.txt"                # path to file to write eval results
    use_wandb: bool = False
    wandb_project: str = ""
    wandb_run_name: str = ""
    wandb_entity: str = ""
    wandb_log_interval: int = 1
    log_mlm_accuracy: bool = True
    mlm_accuracy_interval: int = 1
    log_expert_stats: bool = False
    expert_stats_interval: int = 100
    mlm_probability: float = 0.15



class DataLoader():
    def __init__(self, config, tokenizer=None, rank=0, world_size=1):
        self.config = config
        self.tokenizer = tokenizer
        self.current_epoch = 0
        self.seed = config.seed
        self.token_size = 2 if config.vocab_size < 65535 else 4
        self.rank = rank
        self.world_size = world_size

        self.load_dataset(self.seed)
        self.len_dataset = len(self.dataset)

        if rank == 0:
            print(f"Total tokens loaded: {self.len_dataset * config.max_seq_len:,}")

        self.train_len_dataset = math.ceil((1 - config.val_ratio) * self.len_dataset)
        self.val_len_dataset = self.len_dataset - self.train_len_dataset

        shard_size = self.len_dataset // world_size 
        self.train_start_idx = rank * shard_size
        self.train_end_idx = self.train_start_idx + shard_size
        self.train_current_idx = self.train_start_idx

        self.val_start_idx = self.train_len_dataset
        self.val_current_idx = self.val_start_idx

    def get_batch(self, current_idx: int, start_idx: int, end_idx: int):
        """
        Returns a batch of sequences for MLM training.
        For MLM, we just need the raw sequences - masking happens in the trainer.
        
        Returns:
            x: Tensor of shape (batch_size, seq_len) containing input_ids
            new_idx: Updated index for next batch
        """
        new_idx = current_idx + self.config.batch_size
        
        # For MLM, just get the full sequences - no need to offset by 1
        x_list = [self.dataset[idx]['input_ids'] 
                  for idx in range(current_idx, min(new_idx, self.len_dataset))]
        x = torch.stack(x_list)
    
        if new_idx >= end_idx:
            new_idx = start_idx
            self.new_epoch()

        return x, new_idx

    def next_batch(self, split):
        """
        Get next batch for training or validation.
        
        Args:
            split: 'train' or 'val'
            
        Returns:
            x: Tensor of input_ids (batch_size, seq_len)
            None: Placeholder for compatibility (labels created by masking)
        """
        if split == "train":
            x, self.train_current_idx = self.get_batch(
                self.train_current_idx, 
                self.train_start_idx, 
                self.train_end_idx
            )
        else:  # validation
            x, self.val_current_idx = self.get_batch(
                self.val_current_idx, 
                self.val_start_idx, 
                self.len_dataset
            )
        # Return None for y since labels are created by _mask_inputs() in the trainer
        return x, None
    
    def reset(self, rank: int = 0, world_size: int = 1):
        """Reset dataloader to initial state."""
        self.current_epoch = 0
        self.seed = self.config.seed
        self.load_dataset(self.seed)
        self.len_dataset = len(self.dataset)

        self.val_len_dataset = self.len_dataset - self.train_len_dataset

        shard_size = self.len_dataset // world_size 
        self.train_start_idx = rank * shard_size
        self.train_end_idx = self.train_start_idx + shard_size
        self.train_current_idx = self.train_start_idx

        self.val_start_idx = self.train_len_dataset
        self.val_current_idx = self.val_start_idx

    def new_epoch(self):
        """Start a new epoch with shuffled data."""
        self.current_epoch += 1
        if self.config.hf_dataset_name and hasattr(self.dataset, "shuffle"):
            self.dataset = self.dataset.shuffle(seed=self.seed + self.current_epoch)
        else:
            self.load_dataset(self.seed + self.current_epoch)

    def load_dataset(self, seed: int):
        """Load dataset from tokenized files or HuggingFace."""
        if self.config.hf_dataset_name:
            self.dataset = self.load_hf_dataset(seed)
            return

        if DatatroveFolderDataset is None:
            raise ImportError(
                "datatrove is required for tokenized_dataset_path, but it's not installed."
            )

        self.dataset = DatatroveFolderDataset(
            folder_path=self.config.tokenized_dataset_path,
            filename_pattern=os.path.join(
                self.config.tokenized_dataset_path, "**", "*.ds"
            ),
            seq_len=self.config.max_seq_len,
            token_size=self.token_size,
            recursive=True,
            shuffle=True,
            seed=seed + self.rank,
        )

    def load_hf_dataset(self, seed: int):
        """Load and tokenize HuggingFace dataset."""
        if self.tokenizer is None:
            raise ValueError("tokenizer is required when using hf_dataset_name.")

        try:
            from datasets import load_dataset, load_from_disk
        except ImportError as exc:
            raise ImportError(
                "datasets is required for hf_dataset_name. pip install datasets"
            ) from exc

        dataset_kwargs = {
            "path": self.config.hf_dataset_name,
            "split": self.config.hf_dataset_split,
        }
        if self.config.hf_dataset_config:
            dataset_kwargs["name"] = self.config.hf_dataset_config
        if self.config.hf_cache_dir:
            dataset_kwargs["cache_dir"] = self.config.hf_cache_dir

        tokenized_path = self.config.hf_tokenized_path
        if tokenized_path and os.path.isdir(tokenized_path):
            dataset = load_from_disk(tokenized_path)
        else:
            dataset = load_dataset(**dataset_kwargs)

        if "input_ids" not in dataset.column_names:
            text_field = self.config.hf_text_field
            if not text_field:
                if "text" in dataset.column_names:
                    text_field = "text"
                else:
                    for name, feature in dataset.features.items():
                        if getattr(feature, "dtype", None) == "string":
                            text_field = name
                            break
            if not text_field or text_field not in dataset.column_names:
                raise ValueError(
                    "hf_text_field is required and must exist in the dataset."
                )

            seq_len = self.config.max_seq_len
            eos_id = self.tokenizer.eos_token_id

            def tokenize_and_chunk(batch):
                """
                Tokenize text and pack into fixed-length sequences.
                For MLM, we want complete sequences, not offset pairs.
                """
                tokenized = self.tokenizer(
                    batch[text_field],
                    add_special_tokens=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )
                input_ids = []
                buffer = []
                for ids in tokenized["input_ids"]:
                    if self.config.hf_add_eos and eos_id is not None:
                        ids = ids + [eos_id]
                    buffer.extend(ids)
                    # Pack sequences to exactly seq_len tokens
                    while len(buffer) >= seq_len:
                        input_ids.append(buffer[:seq_len])
                        buffer = buffer[seq_len:]
                return {"input_ids": input_ids}

            map_kwargs = {
                "batched": True,
                "remove_columns": dataset.column_names,
                "desc": "Tokenizing and chunking for MLM",
            }
            if self.config.hf_num_proc > 1:
                map_kwargs["num_proc"] = self.config.hf_num_proc

            dataset = dataset.map(tokenize_and_chunk, **map_kwargs)
            if tokenized_path:
                os.makedirs(tokenized_path, exist_ok=True)
                dataset.save_to_disk(tokenized_path)

        dataset = dataset.shuffle(seed=seed)
        if self.world_size > 1:
            dataset = dataset.shard(
                num_shards=self.world_size,
                index=self.rank,
                contiguous=True
            )
        dataset.set_format(type="torch", columns=["input_ids"])
        return dataset

    def num_train_steps(self):
        """Calculate number of training steps per epoch."""
        return math.ceil(
            (self.train_end_idx - self.train_start_idx) / self.config.batch_size
        )


class Trainer():
    def __init__(self, config, model, tokenizer):
        self.config = config
        self.model = model
        self.raw_m = model
        self.num_epochs = config.num_epochs

        self.use_moe = config.use_moe
        self.use_lossfreebalance = config.use_lossfreebalance if self.use_moe else False
        self.clean_cuda_cache = config.clean_cuda_cache
        self.dtype = getattr(torch, self.config.use_dtype)

        self.steps_for_eval = config.steps_for_eval
        self.weight_decay = config.weight_decay
        self.update_rate = config.update_rate if self.use_moe else 0
        self.mlm_probability = config.mlm_probability

        self.device = torch.device(f"cuda:0") if torch.cuda.is_available() else 'cpu'
        if self.device.type == 'cuda':
            torch.cuda.manual_seed(config.seed)
            n_gpus = torch.cuda.device_count()
        
        self.mask_token_id = getattr(tokenizer, "mask_token_id", None) if tokenizer is not None else None
        if self.mask_token_id is None:
            raise ValueError("Tokenizer must define mask_token_id for MLM training.")
        self.special_token_ids = set()
        for token_id in (
            getattr(tokenizer, "pad_token_id", None),
            getattr(tokenizer, "cls_token_id", None),
            getattr(tokenizer, "sep_token_id", None),
            getattr(tokenizer, "bos_token_id", None),
            getattr(tokenizer, "eos_token_id", None),
        ):
            if token_id is not None:
                self.special_token_ids.add(token_id)

        use_compile = self.config.use_compile and self.device.type == "cuda" and torch.__version__.startswith("2")
        if use_compile:
            self.model = torch.compile(self.model)
            
        # DDP
        if n_gpus > 1 and config.use_ddp:
            if not torch.distributed.is_available() or not torch.distributed.is_initialized():
                raise RuntimeError(
                    "DDP requested but torch.distributed is not initialized. "
                    "Launch with torchrun and call init_process_group() before Trainer()."
                )
            self.ddp = True
            self.ddp_rank = int(os.environ['RANK'])
            self.ddp_local_rank = int(os.environ['LOCAL_RANK'])
            self.ddp_world_size = int(os.environ['WORLD_SIZE'])
            self.device = torch.device(f"cuda:{self.ddp_local_rank}")
            torch.cuda.set_device(self.device)
            self.master_process = self.ddp_rank == 0

            self.model.to(self.device)
            
            self.model = DDP(
                self.model,
                device_ids=[self.ddp_local_rank],
                find_unused_parameters=self.use_moe,
                static_graph=True,
            )
        else:
            self.ddp = False
            self.ddp_rank = 0
            self.ddp_world_size = 1
            self.master_process = True

            if self.device != "cpu":
                self.model.to(self.device)

        self.base_model = self.model.module if isinstance(self.model, DDP) else self.model
        self.raw_m = self.base_model

        if self.master_process:
            base_model = self.base_model
            print("Device:", self.device)
            print(f"Model's trainable params: {sum([p.data.numel() for p in self.model.parameters() if p.requires_grad]) / 1e6:.2f}M")
            print(f"Tokens per step: {self.config.batch_size * self.config.max_seq_len * self.ddp_world_size * self.config.accumulation_steps}")
            print(f"use {'torch.compile()'}: {use_compile}")
            print(f"Use MoE: {'Yes ' if self.use_moe else 'No'}")
            if self.use_moe:
                print(f"Number of experts: {base_model.blocks[0].ffn.num_experts}")
                print(f"Number of used experts during inference: {base_model.blocks[0].ffn.moe_routed_experts}")
                print(f"Method of aux_loss: {'loss-free-balance' if config.use_lossfreebalance else 'default'}")
                print(f"Number of parameters will be used during inference: {((sum([p.data.numel() for p in self.model.parameters() if p.requires_grad]) - sum(p.numel() for p in base_model.blocks[0].ffn.parameters()) * len(base_model.blocks) * (1-(base_model.blocks[0].ffn.moe_routed_experts + base_model.blocks[0].ffn.moe_shared_experts) / (base_model.blocks[0].ffn.num_experts + base_model.blocks[0].ffn.moe_shared_experts)))) / 1e6:.2f}M")

        self.use_wandb = bool(config.use_wandb)
        if self.use_wandb and wandb is None:
            if self.master_process:
                print("wandb is not installed; disabling W&B logging.")
            self.use_wandb = False
        if self.use_wandb and self.master_process:
            wandb_project = config.wandb_project or os.environ.get("WANDB_PROJECT", "uncategorized")
            wandb_entity = config.wandb_entity or os.environ.get("WANDB_ENTITY", None)
            wandb_run_name = config.wandb_run_name or os.environ.get("WANDB_RUN_NAME", None)
            wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=wandb_run_name,
                config=asdict(config),
            )
    
    def _mask_inputs(self, input_ids: torch.Tensor):
        """
        Create masked inputs and labels for MLM training.
    
        Strategy (following BERT):
        - 15% of tokens are selected for masking
        - Of those selected:
          - 80% replaced with [MASK]
          - 10% replaced with random token
          - 10% kept unchanged
        
        Args:
            input_ids: Tensor of shape (batch_size, seq_len)
        
        Returns:
            masked_inputs: Input with some tokens masked
            labels: Original tokens at masked positions, -100 elsewhere
        """
        # Validate inputs are within vocab range
        if input_ids.max() >= self.config.vocab_size:
            raise ValueError(
                f"Input contains token ID {input_ids.max()} which is >= vocab_size {self.config.vocab_size}"
            )
        if input_ids.min() < 0:
            raise ValueError(
                f"Input contains negative token ID {input_ids.min()}"
            )
    
        # Validate mask token is in range
        if self.mask_token_id >= self.config.vocab_size:
            raise ValueError(
                f"mask_token_id {self.mask_token_id} is >= vocab_size {self.config.vocab_size}"
            )
    
        labels = input_ids.clone()
    
        # Create probability matrix for masking
        prob_matrix = torch.full(labels.shape, self.mlm_probability, device=labels.device)
    
        # Don't mask special tokens (padding, CLS, SEP, BOS, EOS)
        if self.special_token_ids:
            special_mask = torch.zeros_like(labels, dtype=torch.bool)
            for token_id in self.special_token_ids:
                special_mask |= (labels == token_id)
            prob_matrix.masked_fill_(special_mask, 0.0)
    
        # Select tokens to mask
        masked_indices = torch.bernoulli(prob_matrix).bool()
    
        # Set labels to -100 for non-masked tokens (ignore in loss)
        labels = labels.masked_fill(~masked_indices, -100)
    
        # Create masked inputs
        masked_inputs = input_ids.clone()
    
        # Generate random values for the 80/10/10 split
        rand = torch.rand(labels.shape, device=labels.device)
    
        # 80% of masked tokens -> replace with [MASK]
        mask_token_mask = masked_indices & (rand < 0.8)
        masked_inputs[mask_token_mask] = self.mask_token_id
    
        # 10% of masked tokens -> replace with random token
        # CRITICAL FIX: Ensure random tokens are within valid range [0, vocab_size)
        random_token_mask = masked_indices & (rand >= 0.8) & (rand < 0.9)
        if random_token_mask.any():
            # Generate random tokens in valid range
            random_tokens = torch.randint(
                low=0,
                high=self.config.vocab_size,  # high is exclusive, so this gives [0, vocab_size)
                size=labels.shape,
                dtype=masked_inputs.dtype,
                device=labels.device
            )
            masked_inputs[random_token_mask] = random_tokens[random_token_mask]
    
        # Remaining 10% of masked tokens -> keep unchanged
        # (no action needed, already in masked_inputs)
    
        return masked_inputs, labels
    
    def _calculate_mlm_accuracy(self, logits: torch.Tensor, labels: torch.Tensor):
        """
        Calculate MLM accuracy for masked tokens.
        
        Args:
            logits: Model predictions (batch_size, seq_len, vocab_size)
            labels: Ground truth labels (batch_size, seq_len) with -100 for non-masked
            
        Returns:
            accuracy: Percentage of correctly predicted masked tokens
        """
        # Get predictions (argmax over vocab dimension - last dimension)
        predictions = torch.argmax(logits, dim=-1)  # (batch_size, seq_len)
        
        # Create mask for tokens that should be predicted (labels != -100)
        mask = labels != -100
        
        # Calculate correct predictions
        correct = (predictions == labels) & mask
        
        # Calculate accuracy only over masked tokens
        if mask.sum() == 0:
            return 0.0
        
        accuracy = correct.sum().float() / mask.sum().float()
        return accuracy.item()
    
    def step(self, data_loader, accumulation_steps: int,
              num_tokens: int, split: str = "train", compute_mlm_accuracy: bool = True):
        """
        Performs single forward/backward pass with gradient accumulation.
        Returns: (total_loss, cross_entropy_loss, mlm_accuracy, number_of_processed_tokens)
        """
        x, _ = data_loader.next_batch(split=split)
        x = x.to(self.device)
        x, y = self._mask_inputs(x)
        num_tokens += torch.numel(x)

        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            logits, loss, ce_loss = self.model(x, y)

        # Calculate MLM accuracy (before loss division and backward)
        # Detach logits to avoid keeping computation graph for accuracy calculation
        mlm_acc = None
        if compute_mlm_accuracy:
            with torch.no_grad():
                # Ensure logits are in the right shape (batch_size, seq_len, vocab_size)
                if logits.dim() == 2:
                    # If logits are (batch_size * seq_len, vocab_size), reshape them
                    batch_size = x.size(0)
                    seq_len = x.size(1)
                    logits_reshaped = logits.view(batch_size, seq_len, -1)
                    mlm_acc = self._calculate_mlm_accuracy(logits_reshaped, y)
                else:
                    mlm_acc = self._calculate_mlm_accuracy(logits, y)

        loss /= accumulation_steps

        loss.backward()
        return loss, ce_loss, mlm_acc, num_tokens
    

    def train(self, data_loader):
        num_steps_per_epoch = math.ceil(data_loader.num_train_steps() / self.config.accumulation_steps)

        # Configuration of optimizer and schedulers
        # Using AdamW with cosine decay and warmup - similar to Llama's training setup
        optimizer = torch.optim.AdamW(
            self.model.parameters(),  
            lr=self.config.learning_rate,
            betas=self.config.betas,
            weight_decay=self.weight_decay,
            fused=(self.device.type=="cuda")
        )
        
        warmup_steps = math.floor(self.config.warmup_ratio * num_steps_per_epoch * self.num_epochs)
        warmup_factor = lambda step: 0.05 + 0.95 * (step / max(warmup_steps, 1))
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=warmup_factor
        )

        cos_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=(num_steps_per_epoch * self.num_epochs) - warmup_steps, 
            eta_min=0.1 * self.config.learning_rate
        )
        
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cos_scheduler],
            milestones=[warmup_steps])

        last_step = num_steps_per_epoch - 1
        self.model.train()

        for epoch in range(self.num_epochs):
            for step in range(num_steps_per_epoch):
                t0 = time.perf_counter()
                accumulated_loss = 0.0
                ce_loss_accum = 0.0
                ce_loss_steps = 0
                mlm_acc_accum = 0.0
                mlm_acc_steps = 0
                num_tokens = 0

                should_log_mlm_acc = (
                    self.config.log_mlm_accuracy
                    and (step % self.config.mlm_accuracy_interval == 0 or step == last_step)
                )
                should_log_expert_stats = (
                    self.use_moe
                    and self.config.log_expert_stats
                    and (step % self.config.expert_stats_interval == 0 or step == last_step)
                )

                ddp_nosync_ctx = self.model.no_sync() if self.ddp else nullcontext()
                with ddp_nosync_ctx:
                    for _ in range(self.config.accumulation_steps - 1):
                        loss, ce_loss, mlm_acc, num_tokens = self.step(
                            data_loader,
                            self.config.accumulation_steps,
                            num_tokens,
                            split="train",
                            compute_mlm_accuracy=should_log_mlm_acc,
                        )
                        accumulated_loss += loss
                        if isinstance(ce_loss, torch.Tensor):
                            ce_loss_accum += ce_loss.detach()
                            ce_loss_steps += 1
                        if mlm_acc is not None:
                            mlm_acc_accum += mlm_acc
                            mlm_acc_steps += 1

                loss, ce_loss, mlm_acc, num_tokens = self.step(
                    data_loader,
                    self.config.accumulation_steps,
                    num_tokens,
                    split="train",
                    compute_mlm_accuracy=should_log_mlm_acc,
                )
                accumulated_loss += loss.detach()
                if isinstance(ce_loss, torch.Tensor):
                    ce_loss_accum += ce_loss.detach()
                    ce_loss_steps += 1
                if mlm_acc is not None:
                    mlm_acc_accum += mlm_acc
                    mlm_acc_steps += 1

                # Calculate average MLM accuracy
                avg_mlm_acc = mlm_acc_accum / mlm_acc_steps if mlm_acc_steps > 0 else None

                # Calculate expert biases using Auxiliary Loss-Free Balance method for MoE (https://arxiv.org/pdf/2408.15664)
                if self.use_moe and self.use_lossfreebalance: 
                    base_model = self.base_model
                    for block in range(len(base_model.blocks)):
                        topk_indices = getattr(base_model.blocks[block].ffn, "last_topk_indices", None)
                        if topk_indices is None:
                            continue
                        expert_counts = torch.bincount(topk_indices.flatten(), minlength=base_model.blocks[block].ffn.num_experts)  
                        avg_count = expert_counts.float().mean()
                        for i, count in enumerate(expert_counts):
                            error = avg_count - count.float()
                            base_model.blocks[block].ffn.expert_biases.data[i] += self.update_rate * torch.sign(error)

                norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                t1 = time.perf_counter()

                tokens_per_sec = num_tokens / (t1 - t0) * self.ddp_world_size

                # Logging 
                if self.master_process:
                    acc_str = f"{avg_mlm_acc:.4f}" if avg_mlm_acc is not None else "n/a"
                    print(f"Epoch: {epoch} | Step: {step} | loss: {accumulated_loss:.4f} | acc: {acc_str} | norm: {norm:.4f} | lr: {scheduler.get_last_lr()[0]} | tok/s: {tokens_per_sec}")
                if self.master_process and self.use_wandb and (step % self.config.wandb_log_interval == 0):
                    global_step = epoch * num_steps_per_epoch + step
                    log_data = {
                        "train/loss": float(accumulated_loss),
                        "train/grad_norm": float(norm),
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/tokens_per_sec": float(tokens_per_sec),
                        "train/epoch": epoch,
                        "train/step": step,
                    }
                    if avg_mlm_acc is not None:
                        log_data["train/mlm_accuracy"] = float(avg_mlm_acc)
                    if ce_loss_steps > 0:
                        log_data["train/ce_loss"] = float(ce_loss_accum / ce_loss_steps)
                    if should_log_expert_stats:
                        base_model = self.base_model
                        expert_counts = None
                        for layer_idx, block in enumerate(base_model.blocks):
                            ffn = block.ffn
                            topk_indices = getattr(ffn, "last_topk_indices", None)
                            if topk_indices is None:
                                continue
                            block_counts = torch.bincount(
                                topk_indices[:, 0],
                                minlength=ffn.num_experts,
                            ).float()
                            layer_fractions = block_counts / block_counts.sum().clamp(min=1.0)
                            for i in range(layer_fractions.numel()):
                                log_data[f"experts/layer_{layer_idx}/top1_frac_{i}"] = float(layer_fractions[i])
                            log_data[f"experts/layer_{layer_idx}/top1_entropy"] = float(
                                -(layer_fractions * (layer_fractions + 1e-8).log()).sum()
                            )
                            log_data[f"experts/layer_{layer_idx}/top1_max_frac"] = float(layer_fractions.max())
                            expert_counts = block_counts if expert_counts is None else expert_counts + block_counts
                        if expert_counts is not None:
                            fractions = expert_counts / expert_counts.sum().clamp(min=1.0)
                            for i in range(fractions.numel()):
                                log_data[f"experts/top1_frac_{i}"] = float(fractions[i])
                            log_data["experts/top1_entropy"] = float(
                                -(fractions * (fractions + 1e-8).log()).sum()
                            )
                            log_data["experts/top1_max_frac"] = float(fractions.max())
                    wandb.log(log_data, step=global_step)
                
                # Evaluation 
                if self.master_process and ((step>0 and step % self.config.eval_interval == 0) or step == last_step):
                    self.model.eval() 
                    val_loss, val_acc = self.eval(data_loader)

                    eval_dir = os.path.dirname(self.config.eval_log_file)
                    if eval_dir:
                        os.makedirs(eval_dir, exist_ok=True)

                    with open(self.config.eval_log_file, "a") as f:
                        f.write(f"Step: {step * (epoch+1)}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}, norm: {norm:.4f}, lr: {scheduler.get_last_lr()[0]}, time: {t1 - t0:.2f}s, tok/s: {tokens_per_sec:.1f} \n")

                    if self.use_wandb:
                        global_step = epoch * num_steps_per_epoch + step
                        wandb.log(
                            {
                                "val/loss": float(val_loss),
                                "val/mlm_accuracy": float(val_acc),
                                "val/epoch": epoch,
                                "val/step": step,
                            },
                            step=global_step,
                        )

                    self.model.train()
                    if self.clean_cuda_cache:
                        torch.cuda.empty_cache()

                # Save Chekpoints
                if self.master_process and ((step % self.config.checkpoints_frequency == 0 and step > 0) or step == last_step):
                    self.save_checkpoints(optimizer, self.config.path_to_checkpoints, name=str((epoch+1) * step))
        if self.master_process and self.use_wandb:
            wandb.finish()
    
    def eval(self, data_loader):
        """
        Evaluates model on validation split using running average of first [steps_for_eval] batches.
        Returns: (average_loss, average_mlm_accuracy)
        """
        with torch.no_grad():
            val_loss_accum = 0.0
            val_acc_accum = 0.0
            for _ in range(self.steps_for_eval):
                x, _ = data_loader.next_batch(split="val")
                x = x.to(self.device)
                x, y = self._mask_inputs(x)
                with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                    logits, loss, ce_loss = self.model(x, y)
                
                # Calculate MLM accuracy - handle both 2D and 3D logits
                if logits.dim() == 2:
                    # If logits are (batch_size * seq_len, vocab_size), reshape them
                    batch_size = x.size(0)
                    seq_len = x.size(1)
                    logits_reshaped = logits.view(batch_size, seq_len, -1)
                    mlm_acc = self._calculate_mlm_accuracy(logits_reshaped, y)
                else:
                    mlm_acc = self._calculate_mlm_accuracy(logits, y)
                
                loss /= self.steps_for_eval
                val_loss_accum += loss.detach()
                val_acc_accum += mlm_acc
                
            avg_acc = val_acc_accum / self.steps_for_eval
            return val_loss_accum, avg_acc

    def save_checkpoints(self, optimizer, path: str, name: str):
        os.makedirs(path, exist_ok=True)
        checkpoint_path = os.path.join(path, f"model.checkpoint.{name}.pt")
        # self.model.save_pretrained(".checkpoint_path", config=config)
        checkpoint = {
                    'model': self.model.state_dict(),
                    'optimizer':optimizer.state_dict(),
                }
        torch.save(checkpoint, checkpoint_path)
        print("Checkpoints saved")
