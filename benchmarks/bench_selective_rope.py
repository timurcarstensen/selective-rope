import argparse

import matplotlib.pyplot as plt
import torch
from fla.modules import RotaryEmbedding

from selective_rope.models.gla.configuration import GLAConfig
from selective_rope.models.gla.modeling import GLAForCausalLM, GLAModel
from selective_rope.modules.rotary import SelectiveRoPE


def cuda_bench(fn, warmup=10, iters=100):
    """Return average ms per call."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def print_speedup_table(results, x_values, baseline, label_fmt="T={:>5d}"):
    print(f"\n{'Speedup vs ' + baseline:>40s}")
    print("-" * 78)
    for i, x in enumerate(x_values):
        base = results[baseline][i]
        parts = [f"{label_fmt.format(x)}:"]
        for name in results:
            if name == baseline:
                continue
            ms = results[name][i]
            parts.append(f"{name} {base / ms:.2f}x")
        print(f"  {'  |  '.join(parts)}")


def setup_plot_style():
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams["font.size"] = 12


def plot_variants(
    ax,
    x_values,
    results,
    styles,
    xlabel,
    ylabel,
    title,
    xscale_log=True,
    yscale_log=False,
):
    for name, values in results.items():
        marker, color = styles[name]
        ax.plot(
            x_values, values, marker, label=name, color=color, linewidth=2, markersize=8
        )
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16)
    if xscale_log:
        ax.set_xscale("log", base=2)
    if yscale_log:
        ax.set_yscale("log")
    ax.set_xticks(x_values)
    ax.set_xticklabels([str(s) for s in x_values])
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="best", fontsize=11)


def measure_runtime(module, q, k, hidden, iters, device, split_fwd_bwd=False):
    needs_hidden = hidden is not None

    def fwd_bwd():
        q.grad = k.grad = None
        if needs_hidden:
            hidden.grad = None
            out = module(q=q, k=k, hidden_states=hidden)
        else:
            out = module(q, k)
        (out[0].sum() + out[1].sum()).backward()

    def fwd_only():
        if needs_hidden:
            module(q=q, k=k, hidden_states=hidden)
        else:
            module(q, k)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        total_ms = cuda_bench(fwd_bwd, warmup=10, iters=iters)
        if not split_fwd_bwd:
            return total_ms / 1000.0
        fwd_ms = cuda_bench(fwd_only, warmup=10, iters=iters)
    bwd_ms = total_ms - fwd_ms
    return total_ms / 1000.0, fwd_ms / 1000.0, bwd_ms / 1000.0


def measure_generation(module, B, H, D, prefill_len, gen_tokens, iters, device):
    model_dim = H * D

    def make_inputs(T):
        q = torch.randn(B, T, H, D, device=device, dtype=torch.bfloat16)
        k = torch.randn(B, T, H, D, device=device, dtype=torch.bfloat16)
        h = torch.randn(B, T, model_dim, device=device, dtype=torch.bfloat16)
        return q, k, h

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for _ in range(5):
            q_pf, k_pf, h_pf = make_inputs(prefill_len)
            _, _, cache = module(q_pf, k_pf, h_pf, output_final_state=True)
            for _ in range(gen_tokens):
                q1, k1, h1 = make_inputs(1)
                _, _, cache = module(q1, k1, h1, output_final_state=True, cache=cache)

        q_pf, k_pf, h_pf = make_inputs(prefill_len)
        _, _, cache = module(q_pf, k_pf, h_pf, output_final_state=True)
        q1, k1, h1 = make_inputs(1)
        per_token_ms = cuda_bench(
            lambda: module(q1, k1, h1, output_final_state=True, cache=cache),
            warmup=0,
            iters=iters * gen_tokens,
        )

    return per_token_ms


def bench_training(args):
    device = torch.device("cuda")
    print(f"Device: {device}")
    print(f"Config: B={args.B}, H={args.H}, D={args.D}, iters={args.iters}")

    seq_lengths = [2048, 4096, 8192, 16384, 32768]

    base_kwargs = dict(
        head_k_dim=args.D,
        model_dim=args.H * args.D,
        num_heads=args.H,
        skip_conv=True,
    )

    rope = RotaryEmbedding(dim=args.D, interleaved=False).to(device)

    srope = SelectiveRoPE(**base_kwargs).to(device)
    srope_triton = SelectiveRoPE(**base_kwargs, use_triton=True).to(device)
    srope_triton.load_state_dict(srope.state_dict())

    srope_gate = SelectiveRoPE(**base_kwargs, use_gate=True).to(device)
    srope_gate_triton = SelectiveRoPE(**base_kwargs, use_gate=True, use_triton=True).to(
        device
    )
    srope_gate_triton.load_state_dict(srope_gate.state_dict())

    modules = [
        ("RoPE", rope, None),
        ("SRoPE", torch.compile(srope, dynamic=True), True),
        ("SRoPE (Triton)", torch.compile(srope_triton, dynamic=True), True),
        # ("SRoPE + Gate", torch.compile(srope_gate, dynamic=True), True),
        # ("SRoPE + Gate (Triton)", torch.compile(srope_gate_triton, dynamic=True), True),
    ]
    results = {name: [] for name, _, _ in modules}
    results_fwd = {name: [] for name, _, _ in modules}
    results_bwd = {name: [] for name, _, _ in modules}

    # Compilation warmup: run each module once at max seq length
    print("Compiling...")
    T_max = max(seq_lengths)
    q_w = torch.randn(
        args.B,
        T_max,
        args.H,
        args.D,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    k_w = torch.randn(
        args.B,
        T_max,
        args.H,
        args.D,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    h_w = torch.randn(
        args.B,
        T_max,
        args.H * args.D,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for _, module, needs_hidden in modules:
            if needs_hidden:
                out = module(q=q_w, k=k_w, hidden_states=h_w)
            else:
                out = module(q_w, k_w)
            (out[0].sum() + out[1].sum()).backward()
    torch.cuda.synchronize()
    del q_w, k_w, h_w

    for T in seq_lengths:
        print(f"\nSequence length: {T}")
        q = torch.randn(
            args.B,
            T,
            args.H,
            args.D,
            device=device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        k = torch.randn(
            args.B,
            T,
            args.H,
            args.D,
            device=device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        hidden = torch.randn(
            args.B,
            T,
            args.H * args.D,
            device=device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )

        for name, module, needs_hidden in modules:
            total_s, fwd_s, bwd_s = measure_runtime(
                module,
                q,
                k,
                hidden if needs_hidden else None,
                args.iters,
                device,
                split_fwd_bwd=True,
            )
            total_ms, fwd_ms, bwd_ms = total_s * 1000, fwd_s * 1000, bwd_s * 1000
            tput = args.B * T * 1000 / total_ms
            results[name].append(total_ms)
            results_fwd[name].append(fwd_ms)
            results_bwd[name].append(bwd_ms)
            print(
                f"  {name:25s} "
                f"fwd {fwd_ms:7.3f}  bwd {bwd_ms:7.3f}  "
                f"total {total_ms:7.3f} ms  |  {tput:>10,.0f} tok/s"
            )

    print_speedup_table(results, seq_lengths, "SRoPE")
    print_speedup_table(results_fwd, seq_lengths, "SRoPE", label_fmt="T={:>5d} fwd")
    print_speedup_table(results_bwd, seq_lengths, "SRoPE", label_fmt="T={:>5d} bwd")

    styles = {
        "RoPE": ("o-", "#1f77b4"),
        "SRoPE": ("s-", "#ff7f0e"),
        "SRoPE (Triton)": ("s--", "#ff7f0e"),
        "SRoPE + Gate": ("D-", "#2ca02c"),
        "SRoPE + Gate (Triton)": ("D--", "#2ca02c"),
    }

    tput_results = {
        name: [args.B * T * 1000 / ms for T, ms in zip(seq_lengths, ms_list)]
        for name, ms_list in results.items()
    }

    setup_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    plot_variants(
        ax1,
        seq_lengths,
        results,
        styles,
        "Sequence Length",
        "Runtime per Iteration (ms)",
        "Runtime",
        yscale_log=True,
    )
    plot_variants(
        ax2,
        seq_lengths,
        tput_results,
        styles,
        "Sequence Length",
        "Throughput (tok/s)",
        "Throughput",
    )
    fig.suptitle(
        f"SelectiveRoPE Training (B={args.B}, H={args.H}, D={args.D})",
        fontsize=16,
    )
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {args.output}")


def bench_prefill(args):
    device = torch.device("cuda")
    print(f"Device: {device}")
    print(f"Config: B={args.B}, H={args.H}, D={args.D}, iters={args.iters}")

    seq_lengths = [2048, 4096, 8192, 16384, 32768]

    base_kwargs = dict(
        head_k_dim=args.D,
        model_dim=args.H * args.D,
        num_heads=args.H,
        skip_conv=True,
    )

    rope = RotaryEmbedding(dim=args.D, interleaved=False).to(device)

    srope = SelectiveRoPE(**base_kwargs).to(device)
    srope_triton = SelectiveRoPE(**base_kwargs, use_triton=True).to(device)
    srope_triton.load_state_dict(srope.state_dict())

    modules = [
        ("RoPE", rope, None),
        ("SRoPE", torch.compile(srope, dynamic=True), True),
        ("SRoPE (Triton)", torch.compile(srope_triton, dynamic=True), True),
    ]
    results = {name: [] for name, _, _ in modules}

    # Compilation warmup: run each module once at max seq length to trigger compilation
    print("Compiling...")
    T_max = max(seq_lengths)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        q_w = torch.randn(
            args.B, T_max, args.H, args.D, device=device, dtype=torch.bfloat16
        )
        k_w = torch.randn(
            args.B, T_max, args.H, args.D, device=device, dtype=torch.bfloat16
        )
        h_w = torch.randn(
            args.B, T_max, args.H * args.D, device=device, dtype=torch.bfloat16
        )
        for _, module, needs_hidden in modules:
            if needs_hidden:
                module(q=q_w, k=k_w, hidden_states=h_w)
            else:
                module(q_w, k_w)
        torch.cuda.synchronize()
    del q_w, k_w, h_w

    for T in seq_lengths:
        print(f"\nSequence length: {T}")
        q = torch.randn(args.B, T, args.H, args.D, device=device, dtype=torch.bfloat16)
        k = torch.randn(args.B, T, args.H, args.D, device=device, dtype=torch.bfloat16)
        hidden = torch.randn(
            args.B, T, args.H * args.D, device=device, dtype=torch.bfloat16
        )

        for name, module, needs_hidden in modules:
            h = hidden if needs_hidden else None

            def fwd(m=module, qq=q, kk=k, hh=h):
                if hh is not None:
                    m(q=qq, k=kk, hidden_states=hh)
                else:
                    m(qq, kk)

            with (
                torch.no_grad(),
                torch.autocast(device_type="cuda", dtype=torch.bfloat16),
            ):
                ms = cuda_bench(fwd, warmup=10, iters=args.iters)

            tput = args.B * T * 1000 / ms
            results[name].append(ms)
            print(f"  {name:25s}  {ms:7.3f} ms  |  {tput:>10,.0f} tok/s")

    print_speedup_table(results, seq_lengths, "SRoPE")

    styles = {
        "RoPE": ("o-", "#1f77b4"),
        "SRoPE": ("s-", "#ff7f0e"),
        "SRoPE (Triton)": ("s--", "#ff7f0e"),
    }

    tput_results = {
        name: [args.B * T * 1000 / ms for T, ms in zip(seq_lengths, ms_list)]
        for name, ms_list in results.items()
    }

    setup_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    plot_variants(
        ax1,
        seq_lengths,
        results,
        styles,
        "Sequence Length",
        "Latency (ms)",
        "Prefill Latency",
        yscale_log=True,
    )
    plot_variants(
        ax2,
        seq_lengths,
        tput_results,
        styles,
        "Sequence Length",
        "Throughput (tok/s)",
        "Prefill Throughput",
    )
    fig.suptitle(
        f"SelectiveRoPE Prefill (B={args.B}, H={args.H}, D={args.D})",
        fontsize=16,
    )
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {args.output}")


def bench_generation(args):
    device = torch.device("cuda")
    print(f"Device: {device}")
    print(f"Config: H={args.H}, D={args.D}, iters={args.iters}")

    batch_sizes = [1, 2, 4, 8, 16, 32]
    prefill_len = 512
    gen_tokens = 2048

    base_kwargs = dict(
        head_k_dim=args.D,
        model_dim=args.H * args.D,
        num_heads=args.H,
    )

    rope = RotaryEmbedding(dim=args.D, interleaved=False).to(device)

    srope_pytorch = SelectiveRoPE(**base_kwargs).to(device)
    srope_triton = SelectiveRoPE(**base_kwargs, use_triton=True).to(device)
    srope_triton.load_state_dict(srope_pytorch.state_dict())

    srope_gate = SelectiveRoPE(**base_kwargs, use_gate=True).to(device)
    srope_gate_triton = SelectiveRoPE(**base_kwargs, use_gate=True, use_triton=True).to(
        device
    )
    srope_gate_fused = SelectiveRoPE(
        **base_kwargs, use_gate=True, use_triton=True, fuse_gate=True
    ).to(device)
    srope_gate_triton.load_state_dict(srope_gate.state_dict())
    srope_gate_fused.load_state_dict(srope_gate.state_dict())

    srope_variants = {
        "SRoPE": srope_pytorch,
        "SRoPE (Triton)": srope_triton,
        "SRoPE + Gate": srope_gate,
        "SRoPE + Gate (Triton)": srope_gate_triton,
        "SRoPE + Gate (fused)": srope_gate_fused,
    }

    print(
        f"\nGeneration benchmark: {gen_tokens} tokens after {prefill_len}-token prefill"
    )
    print("=" * 78)

    results = {"RoPE": [], **{name: [] for name in srope_variants}}

    for B in batch_sizes:
        print(f"\nBatch size: {B}")

        # RoPE baseline: stateless, just measure single-token rotation
        q1 = torch.randn(B, 1, args.H, args.D, device=device, dtype=torch.bfloat16)
        k1 = torch.randn(B, 1, args.H, args.D, device=device, dtype=torch.bfloat16)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            tok_ms = cuda_bench(
                lambda: rope(q1, k1),
                warmup=10,
                iters=args.iters * gen_tokens,
            )
        results["RoPE"].append(tok_ms)
        tput = B / tok_ms * 1000
        print(f"  {'RoPE':25s}  {tok_ms:.4f} ms/tok  |  {tput:>8.0f} tok/s")

        for name, module in srope_variants.items():
            tok_ms = measure_generation(
                module, B, args.H, args.D, prefill_len, gen_tokens, args.iters, device
            )
            results[name].append(tok_ms)
            tput = B / tok_ms * 1000
            print(f"  {name:25s}  {tok_ms:.4f} ms/tok  |  {tput:>8.0f} tok/s")

    print_speedup_table(results, batch_sizes, "SRoPE", "B={:>2d}")

    styles = {
        "RoPE": ("o-", "#1f77b4"),
        "SRoPE": ("s-", "#ff7f0e"),
        "SRoPE (Triton)": ("s--", "#ff7f0e"),
        "SRoPE + Gate": ("D-", "#2ca02c"),
        "SRoPE + Gate (Triton)": ("D--", "#2ca02c"),
        "SRoPE + Gate (fused)": ("D:", "#2ca02c"),
    }

    tput_results = {
        name: [B / ms * 1000 for B, ms in zip(batch_sizes, ms_list)]
        for name, ms_list in results.items()
    }

    setup_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    plot_variants(
        ax1,
        batch_sizes,
        results,
        styles,
        "Batch Size",
        "Per-token Latency (ms)",
        "Latency",
    )
    plot_variants(
        ax2,
        batch_sizes,
        tput_results,
        styles,
        "Batch Size",
        "Throughput (tok/s)",
        "Throughput",
    )
    fig.suptitle(
        f"SelectiveRoPE Generation (H={args.H}, D={args.D})",
        fontsize=16,
    )
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {args.output}")


_SROPE_BASE = {
    "skip_conv": False,
    "use_gate": False,
    "phi_parametrization": "low-rank",
    "phi_proj_rank": 32,
    "temp_type": "rope",
    "temp_theta": 500000,
    "temp_max": 1.0,
}

GLA_1B3_PE_VARIANTS = {
    "NoPE": {"type": "nope"},
    "RoPE": {"type": "rope"},
    "SRoPE": {"type": "selective_rope", **_SROPE_BASE},
    "SRoPE (Triton)": {"type": "selective_rope", **_SROPE_BASE, "use_triton": True},
    # "SRoPE (skip_conv)": {"type": "selective_rope", **_SROPE_BASE, "skip_conv": True},
    # "SRoPE (phase_gate)": {
    #     "type": "selective_rope",
    #     **_SROPE_BASE,
    #     "use_gate": True,
    # },
}

_GLA_1B3_STYLES = {
    "NoPE": ("^-", "#2ca02c"),
    "RoPE": ("o-", "#1f77b4"),
    "SRoPE": ("s-", "#ff7f0e"),
    "SRoPE (Triton)": ("s--", "#ff7f0e"),
    "SRoPE (skip_conv)": ("D-", "#d62728"),
    "SRoPE (phase_gate)": ("P-", "#9467bd"),
}


_GLA_1B3_BASE_CFG = dict(
    hidden_size=2048,
    num_heads=16,
    num_hidden_layers=24,
    attn_mode="chunk",
    vocab_size=32000,
)


def _build_gla_models(variants, device, cls=GLAModel, compile=True, **extra_cfg):
    models = {}
    for name, pe_cfg in variants.items():
        config = GLAConfig(
            **_GLA_1B3_BASE_CFG,
            position_embedding=pe_cfg,
            fuse_cross_entropy=False,
            **extra_cfg,
        )
        model = cls(config).to(device).to(torch.bfloat16)
        if compile:
            models[name] = torch.compile(model, dynamic=True)
        else:
            models[name] = model
        print(f"  {name:25s}  {model.num_parameters():>12,} params")
    return models


def bench_gla_prefill(args):
    device = torch.device("cuda")
    B = args.B
    seq_lengths = [32768]

    print(f"Device: {device}  |  B={B}  |  iters={args.iters}")
    print("Building and compiling models...")
    models = _build_gla_models(
        GLA_1B3_PE_VARIANTS, device, cls=GLAModel, use_cache=False
    )
    for model in models.values():
        model.eval()

    # Compilation warmup
    print("Warming up...")
    T_max = max(seq_lengths)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        ids_w = torch.randint(0, 32000, (B, T_max), device=device)
        for model in models.values():
            model(input_ids=ids_w)
        torch.cuda.synchronize()
    del ids_w

    results = {name: [] for name in GLA_1B3_PE_VARIANTS}

    for T in seq_lengths:
        input_ids = torch.randint(0, 32000, (B, T), device=device)
        print(f"\nT={T}")
        for name, model in models.items():
            with (
                torch.no_grad(),
                torch.autocast(device_type="cuda", dtype=torch.bfloat16),
            ):
                ms = cuda_bench(
                    lambda m=model: m(input_ids=input_ids),
                    warmup=10,
                    iters=args.iters,
                )
            tput = B * T * 1000 / ms
            results[name].append(tput)
            print(f"  {name:25s}  {ms:7.1f} ms  {tput:>10,.0f} tok/s")

    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_variants(
        ax,
        seq_lengths,
        results,
        _GLA_1B3_STYLES,
        "Sequence Length",
        "Throughput (tok/s)",
        f"GLA 1.3B Prefill Throughput (B={B})",
    )
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {args.output}")


def bench_gla_training(args):
    device = torch.device("cuda")
    # Fixed token budget: B * T = 16384
    bt_pairs = [(1, 16384), (2, 8192), (4, 4096), (8, 2048)]

    print(f"Device: {device}  |  iters={args.iters}  |  token budget B*T=16384")
    print("Building and compiling models...")
    models = _build_gla_models(
        GLA_1B3_PE_VARIANTS, device, cls=GLAForCausalLM, use_cache=False
    )
    for model in models.values():
        model.train()

    # Compilation warmup at max seq len (B=1, T=16384)
    print("Warming up...")
    B_w, T_w = bt_pairs[0]
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        ids_w = torch.randint(0, 32000, (B_w, T_w), device=device)
        for model in models.values():
            out = model(input_ids=ids_w, labels=ids_w)
            out.loss.backward()
            model.zero_grad()
        torch.cuda.synchronize()
    del ids_w

    results = {name: [] for name in GLA_1B3_PE_VARIANTS}

    for B, T in bt_pairs:
        input_ids = torch.randint(0, 32000, (B, T), device=device)
        print(f"\nB={B}, T={T}")
        for name, model in models.items():

            def train_step(m=model, ids=input_ids):
                out = m(input_ids=ids, labels=ids)
                out.loss.backward()
                m.zero_grad()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                torch.cuda.reset_peak_memory_stats()
                ms = cuda_bench(train_step, warmup=5, iters=args.iters)
                peak_gb = torch.cuda.max_memory_allocated() / 1e9

            tput = B * T * 1000 / ms
            results[name].append(tput)
            print(f"  {name:25s}  {ms:7.1f} ms  {tput:>10,.0f} tok/s  peak {peak_gb:.2f} GB")

    x_labels = [f"B={B}\nT={T//1024}k" for B, T in bt_pairs]
    x = range(len(bt_pairs))

    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    names = list(results.keys())
    n_bars = len(names)
    width = 0.8 / n_bars
    offsets = [(i - (n_bars - 1) / 2) * width for i in range(n_bars)]

    for name, offset in zip(names, offsets):
        marker, color = _GLA_1B3_STYLES[name]
        bar_x = [xi + offset for xi in x]
        ax.bar(bar_x, results[name], width=width, label=name, color=color)

    ax.set_xlabel("Batch Size / Sequence Length", fontsize=14)
    ax.set_ylabel("Throughput (tok/s)", fontsize=14)
    ax.set_title("GLA 1.3B Training Throughput (B*T=16384)", fontsize=16)
    ax.set_xticks(list(x))
    ax.set_xticklabels(x_labels)
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")
    ax.legend(loc="best", fontsize=11)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {args.output}")


def bench_gla_generation(args):
    device = torch.device("cuda")
    batch_sizes = [1, 2, 4, 8, 16, 32]
    prefill_len = 512
    gen_tokens = 128

    print(f"Device: {device}  |  iters={args.iters}")
    print(f"Prefill={prefill_len} tokens, then generate {gen_tokens} tokens")
    print("Building models (no torch.compile for generation)...")
    models = _build_gla_models(
        GLA_1B3_PE_VARIANTS, device, cls=GLAModel, compile=False, use_cache=True
    )
    for model in models.values():
        model.eval()

    results = {name: [] for name in GLA_1B3_PE_VARIANTS}

    for B in batch_sizes:
        print(f"\nBatch size: {B}")
        for name, model in models.items():
            prefill_ids = torch.randint(0, 32000, (B, prefill_len), device=device)
            step_ids = torch.randint(0, 32000, (B, 1), device=device)

            with (
                torch.no_grad(),
                torch.autocast(device_type="cuda", dtype=torch.bfloat16),
            ):
                # Warmup: a few full prefill+generate passes
                for _ in range(10):
                    out = model(input_ids=prefill_ids, use_cache=True)
                    cache = out.past_key_values
                    for _ in range(gen_tokens):
                        out = model(
                            input_ids=step_ids,
                            past_key_values=cache,
                            use_cache=True,
                        )
                        cache = out.past_key_values
                torch.cuda.synchronize()

                # Bench: measure per-token generation after prefill
                out = model(input_ids=prefill_ids, use_cache=True)
                cache = out.past_key_values

                per_tok_ms = cuda_bench(
                    lambda: model(
                        input_ids=step_ids,
                        past_key_values=cache,
                        use_cache=True,
                    ),
                    warmup=0,
                    iters=args.iters * gen_tokens,
                )

            tput = B / per_tok_ms * 1000
            results[name].append(per_tok_ms)
            print(f"  {name:25s}  {per_tok_ms:.4f} ms/tok  |  {tput:>8.0f} tok/s")

    setup_plot_style()
    tput_results = {
        name: [B / ms * 1000 for B, ms in zip(batch_sizes, ms_list)]
        for name, ms_list in results.items()
    }
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    plot_variants(
        ax1,
        batch_sizes,
        results,
        _GLA_1B3_STYLES,
        "Batch Size",
        "Per-token Latency (ms)",
        "GLA 1.3B Generation Latency",
    )
    plot_variants(
        ax2,
        batch_sizes,
        tput_results,
        _GLA_1B3_STYLES,
        "Batch Size",
        "Throughput (tok/s)",
        "GLA 1.3B Generation Throughput",
    )
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {args.output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=[
            "training",
            "prefill",
            "generation",
            "gla-prefill",
            "gla-training",
            "gla-generation",
        ],
        help="Benchmark mode",
    )
    parser.add_argument("--B", type=int, default=4, help="Batch size")
    parser.add_argument("--H", type=int, default=8, help="Number of heads")
    parser.add_argument("--D", type=int, default=128, help="Head dimension")
    parser.add_argument(
        "--iters", type=int, default=100, help="Iterations per benchmark"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output plot filename",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = f"selective_rope_bench_{args.mode}.png"

    if args.mode == "training":
        bench_training(args)
    elif args.mode == "prefill":
        bench_prefill(args)
    elif args.mode == "generation":
        bench_generation(args)
    elif args.mode == "gla-prefill":
        bench_gla_prefill(args)
    elif args.mode == "gla-training":
        bench_gla_training(args)
    elif args.mode == "gla-generation":
        bench_gla_generation(args)


if __name__ == "__main__":
    main()
