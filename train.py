import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from model import Gemma3LanguageModel, Config


cfg = Config()

device = cfg.device

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

vocab_size = len(tokenizer) # Usually 32000 for Mistral

model = Gemma3LanguageModel(vocab_size=vocab_size).to(device=device, dtype=torch.bfloat16)
model = torch.compile(model)  # Compile the model for performance optimization

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learnin_rate, weight_decay=0.1)


print("Loading streaming datasets...")
train_dataset = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True)
val_dataset = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="valid", streaming=True)

def token_streamer(dataset, tokenizer, block_size, batch_size):
    """
    Consumes the streaming dataset, tokenizes text, and packs tokens into 
    continuous blocks to create X (inputs) and Y (targets) dynamically.
    """
    buffer = []
    for sample in dataset:
        # Tokenize the incoming text
        tokens = tokenizer.encode(sample["text"], add_special_tokens=True)
        buffer.extend(tokens)
        
        # Once the buffer has enough tokens to fulfill a batch
        while len(buffer) >= (block_size + 1) * batch_size:
            X_batch = []
            Y_batch = []
            for _ in range(batch_size):
                # Slice out a chunk of block_size + 1
                chunk = buffer[:block_size + 1]
                X_batch.append(chunk[:-1]) # Input tokens
                Y_batch.append(chunk[1:])  # Shifted target tokens
                
                # Shift buffer forward by block_size
                buffer = buffer[block_size:] 
            
            yield torch.tensor(X_batch, dtype=torch.long, device=device), \
                  torch.tensor(Y_batch, dtype=torch.long, device=device)

@torch.no_grad()
def estimate_loss(model, train_iter, val_iter, eval_iters=20):
    """Evaluates the model over a fixed window of batches from the streaming iterators."""
    out = {}
    model.eval()
    
    for split, iterator in [('train', train_iter), ('val', val_iter)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            try:
                X, Y = next(iterator)
            except StopIteration:
                # Re-initialize stream if it runs dry during evaluation
                if split == 'train':
                    new_ds = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True)
                else:
                    new_ds = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="valid", streaming=True)
                iterator = token_streamer(new_ds, tokenizer, cfg.block_size, cfg.batch_size)
                X, Y = next(iterator)
                
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                _, loss = model(X, targets=Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
        
    model.train()
    return out


model.train()

print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')


print('Starting training loop...')

for step in range(max_steps):

    if iter % cfg.n_eval == 0 or iter == cfg.max_iters - 1:
        losses = estimate_loss(model, train_iterator, val_iterator, eval_iters=10)
        print(f"Step {step:06d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")


    # Fetch next batch from the continuous stream
    try:
        X, Y = next(data_iterator)
    except StopIteration:
        # If the stream exhausts, re-initialize it
        data_iterator = token_streamer(dataset, tokenizer, block_size, batch_size)
        X, Y = next(data_iterator)


    optimizer.zero_grad(set_to_none=True)
    
    # Execute forward pass under the bfloat16 autocast context (mixed precision)
    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        logits, loss = model(X, targets=Y)
    
    # Backward pass (scaler not needed for bf16)
    loss.backward()
    
    # Gradient clipping to maintain training stability
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # Optimizer step
    optimizer.step()
    
    if step > 0 and step % 500 == 0:
        print(f"Saving and pushing checkpoint at step {step}...")
        # Ensure you run `huggingface-cli login` in your terminal first
        # Replace 'your-username' with your actual HF handle
        model.push_to_hub(
            "notninja/gemma-500m-smollm", 
            commit_message=f"Training step {step}"
        )