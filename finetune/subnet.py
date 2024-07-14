"""
Instruction-tuning on the Alpaca dataset using a subnet finetuning procedure (updating scores).

Note: If you run into a CUDA error "Expected is_sm80 to be true, but got false", uncomment the line
`torch.backends.cuda.enable_flash_sdp(False)` in the script below (see https://github.com/Lightning-AI/lit-llama/issues/101).
"""
import sys
from pathlib import Path
import os
import time
from functools import partial

import lightning as L
from lightning.fabric.strategies import FSDPStrategy
import numpy as np
import torch
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate import generate
from lit_llama.model_subnet import Block, LLaMA, LLaMAConfig, SupermaskLinear
from lit_llama.tokenizer import Tokenizer
from lit_llama.utils import save_model_checkpoint
# from scripts.prepare_alpaca import generate_prompt
# from scripts.prepare_dolly import generate_prompt

import logging
import json

instruction_tuning = True
eval_interval = 1000
save_interval = 1000
eval_iters = 100
log_interval = 100
devices = 8

# Hyperparameters
learning_rate = 3e-5
batch_size = 32 / devices
micro_batch_size = 4
gradient_accumulation_iters = batch_size // micro_batch_size
assert gradient_accumulation_iters > 0
epoch_size = 50000  # train dataset size
num_epochs = 5
max_iters = num_epochs * (epoch_size // micro_batch_size) // devices
weight_decay = 0.0
block_size = 512
warmup_iters = 100

def main(
    data_dir: str = "data/alpaca",
    pretrained_path: str = "checkpoints/lit-llama/7B/lit-llama.pth",
    tokenizer_path: str = "checkpoints/lit-llama/tokenizer.model",
    sparsity_attn: float = 0.5,
    sparsity_mlp: float = 0.5,
    subnet_mode: str = "both",
    ratio: float = 1.0,
):
    task_name = data_dir.split("/")[-1]
    print(f"Using sparsity_attn = {sparsity_attn}, sparsity_mlp = {sparsity_mlp}.")
    out_dir = f"out/subnet/{task_name}/sub-{subnet_mode}_sparsity_attn-{sparsity_attn}_sparsity_mlp-{sparsity_mlp}_ratio-{ratio}"
    os.makedirs(out_dir, exist_ok=True)

    global logger
    logger = logging.getLogger(f'task-{task_name}_sub-{subnet_mode}_sparsity_attn-{sparsity_attn}_sparsity_mlp-{sparsity_mlp}_ratio-{ratio}')
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(os.path.join(out_dir, 'log.txt'))
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})
    strategy = FSDPStrategy(auto_wrap_policy=auto_wrap_policy, activation_checkpointing=Block, limit_all_gathers=True)
    fabric = L.Fabric(accelerator="cuda", devices=devices, precision="bf16-mixed", strategy=strategy)
    # fabric = L.Fabric(accelerator="cuda", devices=devices, precision="bf16-mixed")
    
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    train_data, val_data = load_datasets(data_dir=data_dir)

    config = LLaMAConfig.from_name("7B")
    config.block_size = block_size
    config.subnet_mode = subnet_mode
    config.sparsity_attn = sparsity_attn
    config.sparsity_mlp = sparsity_mlp
    config.ratio = ratio

    # print(config)
    # print("Max Seq_Len:", config.max_position_embeddings)

    checkpoint = torch.load(pretrained_path)
    
    with fabric.device:
        torch.set_default_tensor_type(torch.HalfTensor)
        model = LLaMA(config).bfloat16()
        torch.set_default_tensor_type(torch.FloatTensor)
        model.load_state_dict(checkpoint, strict=False)

    # with fabric.device:
    #     model = LLaMA(config)
    #     model.load_state_dict(checkpoint, strict=False)
    
    # with fabric.init_module():
    #     model = LLaMA(config)
    #     model.load_state_dict(checkpoint, strict=False)

    model = fabric.setup_module(model)

    # params_to_check = {}
    for name, param in model.named_parameters():
        # if "scores" not in name:
        #     param.requires_grad = False
        #     params_to_check[name] = param.clone().detach()
        if "scores" in name:
            assert name.split(".")[-1] == "scores"
            # print("Parameters with scores:", name)
            # print(f"Parameters requires grad: {name}")
            logger.info(f"Parameters requires grad: {name}")
            param.requires_grad = True
        else:
            # print("Parameters without scores:", name)
            param.requires_grad = False
            # params_to_check[name] = param.clone().detach()
    
    # if not params_to_check:
    #     raise ValueError("params_to_check is empty. No parameters were found that meet the criteria.")
    # else:
    #     # print(params_to_check.keys())
    #     logger.info(params_to_check.keys())

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, foreach=False)
    optimizer = fabric.setup_optimizers(optimizer)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    # model, optimizer = fabric.setup(model, optimizer)

    for name, param in model.named_parameters():
        if "scores" in name:
            assert param.requires_grad == True
        else:
            assert param.requires_grad == False
        # print(f"Parameter: {name}, requires_grad: {param.requires_grad}")

    # train(fabric, model, optimizer, train_data, val_data, out_dir, task_name)
    train(fabric, model, optimizer, train_data, val_data, tokenizer_path, out_dir, task_name)
    # train(fabric, model, optimizer, train_data, val_data, tokenizer_path, out_dir, params_to_check)

    # subnet_masks = {}
    # for name, module in model.named_modules():
    #     if isinstance(module, SupermaskLinear):
    #         subnet_masks[name] = module.get_subnet_mask().detach().cpu()
    
    # Save the final checkpoint at the end of training
    save_model_checkpoint(fabric, model, os.path.join(out_dir, "lit-llama-subnet-finetuned.pth"))
    # state = {
    #     'state_dict': model.state_dict(),
    #     'optimizer': optimizer.state_dict(),
    #     'mask': subnet_masks,
    # }
    # torch.save(state, os.path.join(out_dir, "lit-llama-subnet-finetuned-state_dict-mask.pth"))

def train(
    fabric: L.Fabric,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: np.ndarray,
    val_data: np.ndarray,
    tokenizer_path: str,
    out_dir: str,
    # params_to_check: dict,
    task_name: str,
) -> None:
    """The training loop.

    Loosely based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT.
    """
    step_count = 0
    model.train()

    for iter_num in range(max_iters):

        is_accumulating = (iter_num + 1) % gradient_accumulation_iters != 0

        if step_count <= warmup_iters:
            # linear warmup
            lr = learning_rate * step_count / warmup_iters
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        t0 = time.time()
        
        input_ids, targets = get_batch(fabric, train_data)
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids)
            loss = loss_fn(logits, targets)
            fabric.backward(loss / gradient_accumulation_iters)

        dt = time.time() - t0
        if iter_num % log_interval == 0 or iter_num == max_iters:
            fabric.print(f"iter {iter_num}: train loss {loss.item():.4f}, time: {dt*1000:.2f}ms")
            logger.info(f"iter {iter_num}: train loss {loss.item():.4f}, time: {dt*1000:.2f}ms")
        
        if not is_accumulating:
            optimizer.step()
            optimizer.zero_grad()
            step_count += 1

            # trained_params = dict(model.named_parameters())
            # for name, old_param in params_to_check.items():
            #     assert name in trained_params.keys()
            #     new_param = trained_params[name]
            #     try:
            #         assert torch.equal(old_param.detach(), new_param.detach()), f"Parameter {name} has been updated!"
            #     except AssertionError as e:
            #         print(e)
            #         print(f"old_param: {old_param.detach()}")
            #         print(f"new_param: {new_param.detach()}")
            #         raise RuntimeError(f"Some parameters that should not change {name} have been updated!")
            # logger.info("All parameters that shouldn't be trained have been checked. Go to next step...")

            if step_count % eval_interval == 0 or step_count == max_iters:
                # val_loss = validate(fabric, model, val_data, task_name)
                val_loss = validate(fabric, model, val_data, tokenizer_path, task_name)
                fabric.print(f"step {iter_num}: val loss {val_loss:.4f}")
                logger.info(f"step {iter_num}: val loss {val_loss:.4f}")
                fabric.barrier()

            if step_count % save_interval == 0 or step_count == max_iters:
                # print(f"Saving weights to {out_dir}")
                logger.info(f"Saving weights to {out_dir}")
                
                # subnet_masks = {}
                # for name, module in model.named_modules():
                #     if isinstance(module, SupermaskLinear):
                #         subnet_masks[name] = module.get_subnet_mask().detach().cpu()
                
                save_model_checkpoint(fabric, model, os.path.join(out_dir, f"iter-{iter_num:06d}-ckpt.pth"))
                # state = {
                #     'state_dict': model.state_dict(),
                #     'optimizer': optimizer.state_dict(),
                #     'mask': subnet_masks,
                # }
                # torch.save(state, os.path.join(out_dir, f"iter-{iter_num:06d}-state_dict-mask.pth"))


def generate_response(model, instruction, tokenizer_path, task_name):
    if task_name == "alpaca":
        from scripts.prepare_alpaca import generate_prompt
    elif task_name == "dolly":
        from scripts.prepare_dolly import generate_prompt
    
    tokenizer = Tokenizer(tokenizer_path)
    sample = {"instruction": instruction, "input": ""}
    prompt = instruction
    if instruction_tuning:
        prompt = generate_prompt(sample)
    encoded = tokenizer.encode(prompt, bos=True, eos=False, device=model.device)

    output = generate(
        model,
        idx=encoded,
        max_seq_length=block_size,
        max_new_tokens=100,
    )
    output = tokenizer.decode(output)
    return output # output.split("### Response:")[1].strip()


@torch.no_grad()
# def validate(fabric: L.Fabric, model: torch.nn.Module, val_data: np.ndarray, task_name: str) -> torch.Tensor:
def validate(fabric: L.Fabric, model: torch.nn.Module, val_data: np.ndarray, tokenizer_path: str, task_name: str) -> torch.Tensor:
    fabric.print("Validating ...")
    logger.info("Validating ...")
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        input_ids, targets = get_batch(fabric, val_data)
        logits = model(input_ids)
        loss = loss_fn(logits, targets)
        losses[k] = loss.item()
    out = losses.mean()

    # produce an example:
    instruction = "Recommend a movie for me to watch during the weekend and explain the reason."
    
    output = generate_response(model, instruction, tokenizer_path, task_name)
    fabric.print(instruction)
    logger.info(f"Instruction: {instruction}")
    fabric.print(output)
    logger.info(f"Output: {output}")

    model.train()
    return out.item()


def loss_fn(logits, targets):
    # shift the targets such that output n predicts token n+1
    logits = logits[..., :-1, :].contiguous()
    targets = targets[..., 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
    return loss


def get_batch(fabric: L.Fabric, data: list):
    ix = torch.randint(len(data), (micro_batch_size,))

    input_ids = [data[i]["input_ids"].type(torch.int64) for i in ix]
    labels = [data[i]["labels"].type(torch.int64) for i in ix]

    max_len = max(len(s) for s in input_ids)

    def pad_right(x, pad_id):
        # pad right based on the longest sequence
        n = max_len - len(x)
        return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

    x = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
    y = torch.stack([pad_right(x, pad_id=-1) for x in labels])
    x, y = fabric.to_device((x.pin_memory(), y.pin_memory()))
    return x, y


def load_datasets(data_dir):
    train_data = torch.load(os.path.join(data_dir, "train.pt"))
    val_data = torch.load(os.path.join(data_dir, "test.pt"))
    return train_data, val_data


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")

    from jsonargparse.cli import CLI

    CLI(main)