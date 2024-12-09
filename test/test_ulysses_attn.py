import torch
import torch.distributed as dist
from yunchang import UlyssesAttention

from flash_attn import flash_attn_func
from yunchang.kernels import FlashAttentionImpl, select_flash_attn_impl

def log(msg, a, rank0_only=False):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if rank0_only:
        if rank == 0:
            print(
                f"{msg}: "
                f"max {a.abs().max().item()}, "
                f"mean {a.abs().mean().item()}",
                flush=True,
            )
        return

    for i in range(world_size):
        if i == rank:
            if rank == 0:
                print(f"{msg}:")
            print(
                f"[{rank}] "
                f"max {a.abs().max().item()}, "
                f"mean {a.abs().mean().item()}",
                flush=True,
            )
        dist.barrier()


if __name__ == "__main__":
    dist.init_process_group("nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{rank}")

    batch_size = 2
    seqlen = 3816
    nheads = 8
    d = 128
    dropout_p = 0
    causal = True
    deterministic = False

    assert seqlen % world_size == 0
    assert d % 8 == 0

    q = torch.randn(
        batch_size, seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True
    )
    k = torch.randn(
        batch_size, seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True
    )
    v = torch.randn(
        batch_size, seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True
    )
    dout = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype)

    dist.broadcast(q, src=0)
    dist.broadcast(k, src=0)
    dist.broadcast(v, src=0)
    dist.broadcast(dout, src=0)

    local_q = q.chunk(world_size, dim=1)[rank].detach().clone()
    local_k = k.chunk(world_size, dim=1)[rank].detach().clone()
    local_v = v.chunk(world_size, dim=1)[rank].detach().clone()

    local_dout = dout.chunk(world_size, dim=1)[rank].detach().clone()

    sp_pg = None #dist.new_group(ranks=[i for i in range(world_size)])

    dist_attn = UlyssesAttention(sp_pg, attn_type=FlashAttentionImpl.TORCH)

    if rank == 0:
        print("#" * 30)
        print("# ds-ulysses forward:")
        print("#" * 30)

    local_out = dist_attn(
        local_q,
        local_k,
        local_v,
        dropout_p=dropout_p,
        causal=causal,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
        deterministic=deterministic,
        return_attn_probs=True,
    )

    dist.barrier()

    if rank == 0:
        print("#" * 30)
        print("# local forward:")
        print("#" * 30)

    # reference, a local flash attn
    local_attn_fn = select_flash_attn_impl(FlashAttentionImpl.TORCH, stage="fwd-bwd")
    out_ref = local_attn_fn(
        q,
        k,
        v,
        dropout_p=dropout_p,
        causal=causal,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
        deterministic=deterministic,
        return_attn_probs=True,
    )

    dist.barrier()

    if rank == 0:
        print("#" * 30)
        print("# local forward:")
        print("#" * 30)

    # check correctness

    local_out_ref = out_ref.chunk(world_size, dim=1)[rank]

    log("out", local_out, rank0_only=True)
    log("out diff", local_out_ref - local_out)
