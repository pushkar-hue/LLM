import os
import math
import time
import modal


training_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-devel-ubuntu22.04",
        add_python="3.10"
    )
    .pip_install(
        "torch>=2.4.0",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "ninja",
        "packaging",
        "wheel",
    )
    .pip_install(
        "flash-attn",
        extra_options="--no-build-isolation",
    )
    .pip_install(
        "transformers",
        "datasets",
        "huggingface_hub",
        "accelerate",
        "wandb",
    )
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_file("model.py", remote_path="/root/model.py")
)
app = modal.App("gemma3-pretraining-ddp")

from huggingface_hub import HfApi
import threading

api = HfApi()

def token_streamer(dataset, tokenizer, block_size, batch_size, device):
    buffer = []
    for sample in dataset:
        tokens = tokenizer.encode(sample["text"], add_special_tokens=True)
        buffer.extend(tokens)
        
        while len(buffer) >= (block_size + 1) * batch_size:
            X_batch = []
            Y_batch = []
            for _ in range(batch_size):
                chunk = buffer[:block_size + 1]
                X_batch.append(chunk[:-1])
                Y_batch.append(chunk[1:])
                buffer = buffer[block_size:] 
            
            import torch
            yield (torch.tensor(X_batch, dtype=torch.long, device=device), 
                   torch.tensor(Y_batch, dtype=torch.long, device=device))

def estimate_loss(model, train_iter, val_iter, tokenizer, cfg, device, eval_iters=10):
    import torch
    out = {}
    model.eval()
    
    for split, iterator in [('train', train_iter), ('val', val_iter)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            try:
                batch = next(iterator)
                
                # Check if it's a dictionary (from DataLoader) or tuple (from fallback token_streamer)
                if isinstance(batch, dict):
                    X = batch['X'].to(device, non_blocking=True)
                    Y = batch['Y'].to(device, non_blocking=True)
                else:
                    X, Y = batch[0].to(device), batch[1].to(device)
                    
            except StopIteration:
                from datasets import load_dataset
                dataset_stream = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True)
                
                if split == "train":
                    new_ds = dataset_stream.skip(10000) 
                else:
                    new_ds = dataset_stream.take(10000) 
                    
                # Re-initialize the fallback streamer
                iterator = token_streamer(new_ds, tokenizer, cfg.block_size, cfg.batch_size, device)
                batch = next(iterator)
                X, Y = batch[0], batch[1] # token_streamer yields tuples already on device
                
            with torch.inference_mode(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                _, loss = model(X, targets=Y)
                if loss.dim() > 0: 
                    loss = loss.mean()
            losses[k] = loss.item()
        out[split] = losses.mean()
        
    model.train()
    return out


def get_parameter_norm(model):
    total = 0.0

    for p in model.parameters():
        if p.requires_grad:
            total += p.data.norm(2).item() ** 2

    return total ** 0.5


def get_lr(step, cfg):

    if step < cfg.warmup_steps:
        return cfg.learning_rate * step / cfg.warmup_steps

    if step > cfg.lr_decay_steps:
        return cfg.min_lr

    decay_ratio = (
        step - cfg.warmup_steps
    ) / (
        cfg.lr_decay_steps - cfg.warmup_steps
    )

    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))

    return cfg.min_lr + coeff * (
        cfg.learning_rate - cfg.min_lr
    )
def tokenize_and_chunk(examples, tokenizer, block_size):
    tokens = tokenizer(examples["text"], add_special_tokens=True)["input_ids"]
    
    # Flatten into a single long sequence of tokens
    all_tokens = [t for seq in tokens for t in seq]
    
    # Calculate max lengths divisible by our target chunk size
    seq_len = block_size + 1
    total_length = (len(all_tokens) // seq_len) * seq_len
    
    # Chunk the flat list
    X_batch, Y_batch = [], []
    for i in range(0, total_length, seq_len):
        chunk = all_tokens[i : i + seq_len]
        X_batch.append(chunk[:-1])
        Y_batch.append(chunk[1:])
        
    return {"X": X_batch, "Y": Y_batch}

def get_dataloaders(rank, world_size, tokenizer, cfg, start_step=0):
    from datasets import load_dataset
    from datasets.distributed import split_dataset_by_node
    from torch.utils.data import DataLoader
    
    dataset_stream = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True)
    dataset_stream = split_dataset_by_node(dataset_stream, rank=rank, world_size=world_size)
    
    tokenized_dataset = dataset_stream.map(
        tokenize_and_chunk, 
        batched=True, 
        fn_kwargs={"tokenizer": tokenizer, "block_size": cfg.block_size},
        remove_columns=list(dataset_stream.features.keys())
    ).with_format("torch")

    # We cannot use skip() on a tokenized stream without causing a massive CPU bottleneck.
    # The model will re-process the beginning of the stream, but with the loaded optimizer state.
    val_dataset = tokenized_dataset.take(2000)
    train_dataset = tokenized_dataset.skip(2000) # Only skip the validation set

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, num_workers=4, prefetch_factor=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, num_workers=2, pin_memory=True)
    
    return train_loader, val_loader


def background_upload(local_path, repo_path, repo_id):
    try:
        print("Starting background upload to Hub...")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=repo_id,
            repo_type="model",
        )
        print("Upload complete!")
    except Exception as e:
        print(f"Background upload failed: {e}")

def ddp_worker(rank, world_size):
    import torch
    import wandb
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.distributed import init_process_group, destroy_process_group
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from model import Gemma3LanguageModel, Config
    
    # Initialize the process group
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"
    master_process = (rank == 0)
    
    cfg = Config()
    
    # Setup gradient accumulation settings scaled by world size
    if cfg.gradient_accumulation_steps % world_size == 0:
        cfg.gradient_accumulation_steps //= world_size

    # Initialize W&B only on the master rank process
    if master_process:
        wandb.init(
            project="gemma-500m-pretraining", 
            name="smollm-ddp-run",
            config={
                "learning_rate": cfg.learning_rate,
                "architecture": "Gemma 3 Custom DDP",
                "dataset": "HuggingFaceTB/smollm-corpus",
                "batch_size": cfg.batch_size,
                "block_size": cfg.block_size,
                "world_size": world_size
            }
        )
    print(f"[Rank {rank}] Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    print(f"[Rank {rank}] Tokenizer loaded", flush=True)
    vocab_size = len(tokenizer)

    # Instantiate model and wrap inside DDP container
    print(f"[Rank {rank}] Building model...", flush=True)
    model = Gemma3LanguageModel(vocab_size=vocab_size).to(device)
    model.gradient_checkpointing_enable()
    
    
    print(f"[Rank {rank}] DDP wrapping...", flush=True)
    model = DDP(model, device_ids=[rank])

    import torch._dynamo
    torch._dynamo.config.optimize_ddp = False
    model = torch.compile(model)

    
    raw_model = model.module
    decay = []
    no_decay = []
    start_step = cfg.start_step
    for name, param in raw_model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim >= 2:
            decay.append(param)
        else:
            no_decay.append(param)

    optimizer = torch.optim.AdamW(
                    [
                        {"params": decay, "weight_decay": cfg.weight_decay},
                        {"params": no_decay, "weight_decay": 0.0},
                    ],
                    lr=cfg.learning_rate,
                    betas=(cfg.beta1, cfg.beta2),
                    eps=cfg.eps
                )

    
    print(f"[Rank {rank}] Loading and configuring dataset pipeline...")  
    train_loader, val_loader = get_dataloaders(rank, world_size, tokenizer, cfg)
    
    train_iterator = iter(train_loader)
    val_iterator = iter(val_loader)

    if start_step > 0:
        try:
            from huggingface_hub import hf_hub_download
            
            print(f"[Rank {rank}] Attempting to download Step {start_step} checkpoint...")
            ckpt_path = hf_hub_download(
                repo_id="notninja/gemma-500m-smollm", 
                filename="checkpoint_latest.pt", 
                repo_type="model",
                revision="a6931f727d63b48d80b0314b137b4b03a66a9f22"  
            )
            
            print(f"[Rank {rank}] Loading checkpoint into memory...")
            checkpoint = torch.load(ckpt_path, map_location=device)
            
            # Restore the model weights and optimizer state
            raw_model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_step = checkpoint["step"]
            
            print(f"[Rank {rank}] Successfully time-traveled back to step {start_step}!")
            
        except Exception as e:
            print(f"[Rank {rank}] Checkpoint load failed. ({e})")
    
    
    model.train()
    print(f"[Rank {rank}] Starting training...", flush=True)
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)
    for step in range(start_step, cfg.max_steps):

        lr = get_lr(step, cfg)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        
        # Evaluation Block (Handled by Master Process)
        if (step % cfg.n_eval == 0 or step == cfg.max_steps - 1) and master_process:
            losses = estimate_loss(model, train_iterator, val_iterator, tokenizer, cfg, device, eval_iters=10)
            print(f"Step {step:06d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}", flush=True)
            wandb.log({"train/loss": losses['train'], "val/loss": losses['val'], "step": step, "lr": lr})

        
        # Micro-stepping accumulation loop
        # Micro-stepping accumulation loop
        for micro_step in range(cfg.gradient_accumulation_steps):
            model.require_backward_grad_sync = (micro_step == cfg.gradient_accumulation_steps - 1)
            
            try:
                batch = next(train_iterator)
            except StopIteration:
                # Re-initialize if we somehow exhaust the infinite stream
                train_iterator = iter(train_loader)
                batch = next(train_iterator)

            # Move background-processed batches to the GPU
            X = batch['X'].to(device, non_blocking=True)
            Y = batch['Y'].to(device, non_blocking=True)

            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                _, loss = model(X, targets=Y)
            
            loss = loss / cfg.gradient_accumulation_steps # scale the loss to account for gradient accumulation

            loss.backward()


        # Optimizer execution step
        grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        cfg.grad_clip
                    )
        optimizer.step()
        dt = time.time() - t0
        t0 = time.time()

        tokens_processed = (
            cfg.batch_size
            * cfg.block_size
            * cfg.gradient_accumulation_steps
            * world_size
        )

        samples_processed = (
            cfg.batch_size
            * cfg.gradient_accumulation_steps
            * world_size
        )

        tokens_per_sec = tokens_processed / dt
        samples_per_sec = samples_processed / dt

        gpu_mem = torch.cuda.max_memory_allocated() / 1024**3

        param_norm = get_parameter_norm(raw_model)

        remaining_steps = cfg.max_steps - step

        eta_hours = remaining_steps * dt / 3600
        if master_process:
            wandb.log({
            "train/loss": loss.item() * cfg.gradient_accumulation_steps,
            "lr": lr,
            "grad_norm": grad_norm.item(),
            "param_norm": param_norm,
            "tokens/sec": tokens_per_sec,
            "samples/sec": samples_per_sec,
            "gpu_memory_gb": gpu_mem,
            "step": step,
            "train/eta_hours": eta_hours
        })
        torch.cuda.reset_peak_memory_stats()
        optimizer.zero_grad(set_to_none=True)

        if step > 0 and step % 2000 == 0 and step != start_step and master_process:
            print(f"Saving checkpoint {step}")
            torch.save(
                {
                    "step": step,
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": cfg.__dict__,
                },
                "checkpoint.pt",
            )
            
            # Fire and forget the upload; don't block Rank 0!
            threading.Thread(
                target=background_upload, 
                args=("checkpoint.pt", "checkpoint_latest.pt", "notninja/gemma-500m-smollm")
            ).start()

            # raw_model.push_to_hub(
            #     "notninja/gemma-500m-smollm",
            #     commit_message=f"Training step {step}",
            # )    
    if master_process:
        wandb.finish()
        
    destroy_process_group()

@app.function(
    image=training_image,
    gpu="A10G:4",                                   
    timeout=86400,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret") 
    ] 
)
def train():
    import torch
    import torch.multiprocessing as mp
    
    world_size = torch.cuda.device_count()
    print(f"Discovered {world_size} available GPUs. Spawning DDP Process Workers...", flush=True)
    
    # Target execution block dynamically via mp.spawn
    mp.spawn(
        ddp_worker,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )

@app.local_entrypoint()
def main():
    train.spawn()