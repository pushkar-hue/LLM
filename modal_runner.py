import os
import math
import time
import modal

# 1. Container Image Definition
training_image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .pip_install(
        "torch>=2.4.0",
        "transformers",
        "datasets",
        "huggingface_hub",
        "accelerate",
        "wandb"
    )
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}) 
    .add_local_file("model.py", remote_path="/root/model.py")
)

app = modal.App("gemma3-pretraining-ddp")

# --- DATA STREAMER UTIL ---
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

# --- ESTIMATE LOSS UTIL ---
def estimate_loss(model, train_iter, val_iter, tokenizer, cfg, device, eval_iters=10):
    import torch
    out = {}
    model.eval()
    
    for split, iterator in [('train', train_iter), ('val', val_iter)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            try:
                X, Y = next(iterator)
            except StopIteration:
                from datasets import load_dataset
                dataset_stream = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True)
                
                if split == "train":
                    new_ds = dataset_stream.skip(10000) 
                else:
                    new_ds = dataset_stream.take(10000) 
                    
                iterator = token_streamer(new_ds, tokenizer, cfg.block_size, cfg.batch_size, device)
                X, Y = next(iterator)
                
            with torch.inference_mode(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                _, loss = model(X, targets=Y)
                if loss.dim() > 0: 
                    loss = loss.mean()
            losses[k] = loss.item()
        out[split] = losses.mean()
        
    model.train()
    return out


def get_lr(step):

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

# --- DDP WORKER EXECUTION LOOP ---
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
    gradient_accumulation_steps = 4
    if gradient_accumulation_steps % world_size == 0:
        gradient_accumulation_steps //= world_size

    # Initialize W&B only on the master rank process
    if master_process:
        wandb.init(
            project="gemma-500m-pretraining", 
            name="smollm-ddp-run",
            config={
                "learning_rate": cfg.learnin_rate,
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
    model = torch.compile(model)
    print(f"[Rank {rank}] DDP wrapping...", flush=True)
    model = DDP(model, device_ids=[rank])
    raw_model = model.module

    optimizer = torch.optim.AdamW(
                    [
                        {"params": decay, "weight_decay": cfg.weight_decay},
                        {"params": no_decay, "weight_decay": 0.0},
                    ],
                    lr=cfg.learning_rate,
                    betas=(cfg.beta1, cfg.beta2),
                )
    print(f"[Rank {rank}] Loading dataset...")  
    # Dataset stream setup
    dataset_stream = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True)
    
    # Quick sharding strategy for streaming datasets across ranks to prevent identical processing
    dataset_stream = dataset_stream.shard(num_shards=world_size, index=rank)
    
    val_dataset = dataset_stream.take(2000)
    train_dataset = dataset_stream.skip(2000)

    train_iterator = token_streamer(train_dataset, tokenizer, cfg.block_size, cfg.batch_size, device)
    val_iterator = token_streamer(val_dataset, tokenizer, cfg.block_size, cfg.batch_size, device)
    
    model.train()
    print(f"[Rank {rank}] Starting training...", flush=True)
    
    for step in range(cfg.max_steps):

        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        # Evaluation Block (Handled by Master Process)
        if (step % cfg.n_eval == 0 or step == cfg.max_steps - 1) and master_process:
            losses = estimate_loss(model, train_iterator, val_iterator, tokenizer, cfg, device, eval_iters=10)
            print(f"Step {step:06d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}", flush=True)
            wandb.log({"train/loss": losses['train'], "val/loss": losses['val'], "step": step})

        optimizer.zero_grad(set_to_none=True)
        
        # Micro-stepping accumulation loop
        for micro_step in range(gradient_accumulation_steps):
            # Toggle backward sync boundary check
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            
            try:
                X, Y = next(train_iterator)
            except StopIteration:
                dataset_stream = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True)
                dataset_stream = dataset_stream.shard(num_shards=world_size, index=rank)
                train_dataset = dataset_stream.skip(2000)
                train_iterator = token_streamer(train_dataset, tokenizer, cfg.block_size, cfg.batch_size, device)
                X, Y = next(train_iterator)

            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                _, loss = model(X, targets=Y)
                if loss.dim() > 0:  
                    loss = loss.mean()
                loss = loss / gradient_accumulation_steps

            loss.backward()

        # Optimizer execution step
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if master_process:
            wandb.log({"batch/loss": loss.item() * gradient_accumulation_steps})

        # Save Checkpoint State safely
        if step > 0 and step % 500 == 0 and master_process:
            print(f"Saving and pushing checkpoint at step {step} to Hugging Face...", flush=True)
            raw_model.push_to_hub("notninja/gemma-500m-smollm", commit_message=f"Training step {step}")
            
    if master_process:
        wandb.finish()
        
    destroy_process_group()

# --- 2. MODAL SERVERLESS SYSTEM ENTRYPOINT ---
@app.function(
    image=training_image,
    gpu="A10G:2",                                   
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

if __name__ == "__main__":
    with app.run():
        train.remote()