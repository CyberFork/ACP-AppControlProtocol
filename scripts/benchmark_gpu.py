"""GPU 基准测试：ViT forward 延迟 + QLoRA 训练显存占用。

用法：
    python scripts/benchmark_gpu.py
    python scripts/benchmark_gpu.py --vit-only
    python scripts/benchmark_gpu.py --qlora-only
    python scripts/benchmark_gpu.py --output report.json
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Optional

import torch


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _gb(n_bytes: int) -> float:
    return n_bytes / 1024 ** 3


@contextmanager
def _cuda_timer():
    """精确 CUDA 计时上下文管理器。"""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    yield
    end.record()
    torch.cuda.synchronize()
    yield_result = start.elapsed_time(end)  # ms
    # 将结果挂到 end 上供外部读取
    end._elapsed_ms = yield_result


def cuda_time_ms(fn, warmup: int = 3, runs: int = 10) -> tuple[float, float]:
    """返回 (mean_ms, std_ms)。"""
    # warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(runs):
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        fn()
        t1.record()
        torch.cuda.synchronize()
        times.append(t0.elapsed_time(t1))

    import statistics
    return statistics.mean(times), statistics.stdev(times) if len(times) > 1 else 0.0


def vram_used_mb() -> float:
    return torch.cuda.memory_allocated() / 1024 ** 2


def vram_reserved_mb() -> float:
    return torch.cuda.memory_reserved() / 1024 ** 2


# ── 基准测试数据结构 ──────────────────────────────────────────────────────────

@dataclass
class VitBenchResult:
    batch_size: int
    num_patches: int
    mean_latency_ms: float
    std_latency_ms: float
    throughput_imgs_per_sec: float
    vram_delta_mb: float
    dtype: str

@dataclass
class QloraBenchResult:
    stage: int
    batch_size: int
    seq_len: int
    grad_accum: int
    vram_forward_mb: float
    vram_backward_mb: float
    vram_peak_mb: float
    trainable_params_m: float
    total_params_m: float
    dtype: str
    note: str = ""

@dataclass
class BenchmarkReport:
    gpu_name: str
    gpu_vram_gb: float
    cuda_version: str
    torch_version: str
    timestamp: str
    vit_results: list = field(default_factory=list)
    qlora_results: list = field(default_factory=list)


# ── Mock 模块（无需真实权重） ─────────────────────────────────────────────────

class MockViT(torch.nn.Module):
    """Qwen2-VL ViT 的 mock 实现，参数量/计算量近似真实模型。

    真实 Qwen2-VL ViT (1.1B):
      - 24 层 Transformer，hidden=1152，heads=16，FFN=4352
      - 输入 patches 经 temporal patch embedding → 1152 维
    """

    def __init__(self, hidden: int = 1152, num_layers: int = 24, ffn: int = 4352):
        super().__init__()
        self.patch_embed = torch.nn.Linear(14 * 14 * 3 * 4, hidden)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=16,
            dim_feedforward=ffn,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = torch.nn.LayerNorm(hidden)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(patches)           # [N, 1152]
        x = x.unsqueeze(0)                      # [1, N, 1152]
        x = self.encoder(x)
        x = self.norm(x)
        return x.squeeze(0)                     # [N, 1152]


class MockLM(torch.nn.Module):
    """Qwen2.5-3B-Instruct mock，参数量近似 3B。"""

    def __init__(self, vocab: int = 151936, hidden: int = 2048,
                 num_layers: int = 32, ffn: int = 11008):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(vocab, hidden)
        layer = torch.nn.TransformerDecoderLayer(
            d_model=hidden, nhead=16, dim_feedforward=ffn,
            batch_first=True, norm_first=True,
        )
        self.layers = torch.nn.ModuleList([layer] * num_layers)  # 共享层减少内存
        self.norm = torch.nn.LayerNorm(hidden)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, x)[0] if isinstance(layer(x, x), tuple) else layer(x, x)
        return self.norm(x)


# ── ViT Forward 延迟测试 ──────────────────────────────────────────────────────

def bench_vit(
    batch_sizes: list[int] = [1, 2, 4],
    patch_counts: list[int] = [256, 512, 1024],
    dtype: torch.dtype = torch.float16,
) -> list[VitBenchResult]:
    """测试 ViT forward 延迟（mock 模型）。"""

    print("\n" + "=" * 60)
    print("ViT Forward 延迟测试（Mock Qwen2-VL ViT, 24层）")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vit = MockViT().to(device=device, dtype=dtype)
    vit.eval()

    results = []
    patch_dim = 14 * 14 * 3 * 4  # 2352

    for bs in batch_sizes:
        for n_patches in patch_counts:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            vram_before = vram_used_mb()

            patches = torch.randn(n_patches, patch_dim, device=device, dtype=dtype)

            def _fwd():
                with torch.no_grad():
                    _ = vit(patches)

            try:
                mean_ms, std_ms = cuda_time_ms(_fwd, warmup=2, runs=8)
                vram_delta = vram_used_mb() - vram_before
                throughput = bs / (mean_ms / 1000)  # imgs/s（每 batch 1 张图）

                r = VitBenchResult(
                    batch_size=bs,
                    num_patches=n_patches,
                    mean_latency_ms=round(mean_ms, 2),
                    std_latency_ms=round(std_ms, 2),
                    throughput_imgs_per_sec=round(throughput, 1),
                    vram_delta_mb=round(vram_delta, 1),
                    dtype=str(dtype).split(".")[-1],
                )
                results.append(r)
                print(
                    f"  bs={bs:2d}  patches={n_patches:4d}  "
                    f"latency={mean_ms:7.2f}±{std_ms:5.2f}ms  "
                    f"VRAM={vram_delta:6.1f}MB"
                )
            except RuntimeError as e:
                print(f"  bs={bs:2d}  patches={n_patches:4d}  OOM: {e}")

    return results


# ── QLoRA 训练显存占用测试 ────────────────────────────────────────────────────

def bench_qlora(
    stages: list[int] = [1, 2, 3],
    batch_size: int = 2,
    seq_len: int = 512,
    grad_accum: int = 8,
) -> list[QloraBenchResult]:
    """测试各训练阶段的显存占用（mock 胶水层）。"""

    print("\n" + "=" * 60)
    print("QLoRA 训练显存占用测试（各 Stage）")
    print("=" * 60)

    import sys
    sys.path.insert(0, ".")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16
    results = []

    # 导入胶水层
    try:
        from acp.mot.perceiver import PerceiverResampler
        from acp.mot.gated_cross_attention import GatedCrossAttentionBlock
        from acp.mot.action_head import ActionHead
        from acp.mot.action_projector import ActionProjector
        from acp.mot.config import MoTConfig
        cfg = MoTConfig()
        HAS_ACP = True
    except ImportError:
        HAS_ACP = False
        print("  [WARN] acp 模块导入失败，使用纯 mock 层")

    stage_configs = {
        1: {"name": "V-L 对齐", "trainable": ["perceiver"]},
        2: {"name": "UI 融合",  "trainable": ["perceiver", "cross_attn"]},
        3: {"name": "端到端",   "trainable": ["perceiver", "cross_attn", "action_head", "action_projector"]},
    }

    for stage in stages:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        stage_info = stage_configs[stage]

        print(f"\n  Stage {stage}: {stage_info['name']}")
        print(f"  可训练层: {stage_info['trainable']}")

        if HAS_ACP:
            modules = {}
            modules["perceiver"] = PerceiverResampler(
                d_visual=cfg.d_visual, d_lang=cfg.d_lang,
                num_latents=cfg.num_latents, depth=cfg.perceiver_depth,
                rank=cfg.perceiver_rank, num_heads=cfg.perceiver_heads,
            ).to(device=device, dtype=dtype)

            if "cross_attn" in stage_info["trainable"]:
                modules["cross_attn"] = torch.nn.ModuleList([
                    GatedCrossAttentionBlock(d_lang=cfg.d_lang, heads=cfg.num_heads)
                    for _ in cfg.injection_layers
                ]).to(device=device, dtype=dtype)

            if "action_projector" in stage_info["trainable"]:
                modules["action_projector"] = ActionProjector(
                    d_lang=cfg.d_lang, d_action=cfg.d_action,
                    rank=cfg.action_proj_rank,
                ).to(device=device, dtype=dtype)

            if "action_head" in stage_info["trainable"]:
                modules["action_head"] = ActionHead(
                    d_lang=cfg.d_action, d_action=256,
                    num_action_types=cfg.num_action_types,
                    max_elements=cfg.max_elements,
                    label_vocab_size=cfg.label_vocab_size,
                ).to(device=device, dtype=dtype)
        else:
            # Fallback mock 层
            modules = {
                "perceiver": torch.nn.Sequential(
                    torch.nn.Linear(1152, 2048), torch.nn.ReLU(), torch.nn.Linear(2048, 2048)
                ).to(device=device, dtype=dtype)
            }

        # 统计可训练参数
        all_params = sum(p.numel() for m in modules.values() for p in m.parameters())
        trainable_params = sum(
            p.numel() for m in modules.values()
            for p in m.parameters() if p.requires_grad
        )

        vram_base = vram_used_mb()

        # 模拟一次前向（用随机 tensor 代替）
        visual_tokens = torch.randn(batch_size, 64, 1152, device=device, dtype=dtype)
        hidden = torch.randn(batch_size, seq_len, 2048, device=device, dtype=dtype, requires_grad=True)

        # 通过 perceiver
        if HAS_ACP and "perceiver" in modules:
            try:
                visual_out = modules["perceiver"](visual_tokens)
            except Exception:
                visual_out = visual_tokens
        else:
            visual_out = modules["perceiver"](visual_tokens.view(batch_size * 64, 1152)).view(batch_size, 64, -1)

        vram_forward = vram_used_mb()

        # 模拟 backward
        loss = visual_out.mean()
        loss.backward()

        vram_backward = vram_used_mb()
        vram_peak = torch.cuda.max_memory_allocated() / 1024 ** 2

        # 显存估算（加上 LLM base 和 ViT 的占用）
        # RTX 3090 24GB：LLM int4 ≈ 1.5-2GB，ViT fp16 ≈ 2.2GB
        vit_estimated_mb = 2200
        llm_int4_estimated_mb = 1800
        total_estimated_mb = vram_peak + vit_estimated_mb + llm_int4_estimated_mb

        note = (
            f"含 ViT(~{vit_estimated_mb/1024:.1f}GB) + LLM-4bit(~{llm_int4_estimated_mb/1024:.1f}GB)"
            f" 总估算 ~{total_estimated_mb/1024:.1f}GB"
        )

        r = QloraBenchResult(
            stage=stage,
            batch_size=batch_size,
            seq_len=seq_len,
            grad_accum=grad_accum,
            vram_forward_mb=round(vram_forward - vram_base, 1),
            vram_backward_mb=round(vram_backward - vram_base, 1),
            vram_peak_mb=round(vram_peak, 1),
            trainable_params_m=round(trainable_params / 1e6, 2),
            total_params_m=round(all_params / 1e6, 2),
            dtype=str(dtype).split(".")[-1],
            note=note,
        )
        results.append(r)

        print(f"  显存 forward={r.vram_forward_mb:.0f}MB  "
              f"backward={r.vram_backward_mb:.0f}MB  peak={r.vram_peak_mb:.0f}MB")
        print(f"  可训练参数: {r.trainable_params_m:.2f}M / {r.total_params_m:.2f}M")
        print(f"  {note}")

        # 清理
        del visual_tokens, hidden, visual_out, loss
        for m in modules.values():
            del m
        torch.cuda.empty_cache()

    return results


# ── 报告输出 ──────────────────────────────────────────────────────────────────

def print_summary(report: BenchmarkReport) -> None:
    print("\n" + "=" * 60)
    print("基准测试报告摘要")
    print("=" * 60)
    print(f"GPU:     {report.gpu_name}")
    print(f"VRAM:    {report.gpu_vram_gb:.1f} GB")
    print(f"CUDA:    {report.cuda_version}")
    print(f"PyTorch: {report.torch_version}")
    print()

    if report.vit_results:
        print("ViT Forward 延迟（float16）:")
        print(f"  {'patches':>8}  {'latency':>12}  {'VRAM':>8}")
        for r in report.vit_results:
            print(f"  {r['num_patches']:>8}  "
                  f"{r['mean_latency_ms']:>8.2f}±{r['std_latency_ms']:>4.2f}ms  "
                  f"{r['vram_delta_mb']:>6.1f}MB")

    if report.qlora_results:
        print("\nQLoRA 训练显存估算（含 ViT+LLM base）:")
        print(f"  {'Stage':>6}  {'峰值':>10}  {'总估算':>12}  {'参数':>12}")
        for r in report.qlora_results:
            note_gb = r.get("note", "")
            print(f"  Stage {r['stage']}  "
                  f"{r['vram_peak_mb']:>8.0f}MB  "
                  f"  {r['note'].split('总估算 ')[-1] if '总估算' in r['note'] else 'N/A':>8}  "
                  f"  {r['trainable_params_m']:.2f}M/{r['total_params_m']:.2f}M")

    # 3090 适配建议
    print()
    vram_gb = report.gpu_vram_gb
    print("RTX 3090 (24GB) 训练建议:")
    print("  Stage 1: batch=4, grad_accum=8  (有效批=32, 显存~12GB)")
    print("  Stage 2: batch=2, grad_accum=16 (有效批=32, 显存~18GB)")
    print("  Stage 3: batch=1, grad_accum=32 (有效批=32, 显存~22GB)")
    if vram_gb < 20:
        print(f"  [WARN] 当前 GPU VRAM {vram_gb:.1f}GB < 24GB，建议减小 batch_size")


def main():
    parser = argparse.ArgumentParser(description="GPU 基准测试")
    parser.add_argument("--vit-only",   action="store_true", help="只跑 ViT 测试")
    parser.add_argument("--qlora-only", action="store_true", help="只跑 QLoRA 测试")
    parser.add_argument("--output",     default=None,        help="输出 JSON 报告路径")
    parser.add_argument("--batch",      type=int, default=2, help="QLoRA batch size")
    parser.add_argument("--seq-len",    type=int, default=512, help="序列长度")
    args = parser.parse_args()

    import datetime

    # GPU 信息
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        gpu_name = gpu.name
        gpu_vram = gpu.total_memory / 1024 ** 3
    else:
        print("[WARN] CUDA 不可用，将使用 CPU 运行（结果不具参考意义）")
        gpu_name = "CPU"
        gpu_vram = 0.0

    report = BenchmarkReport(
        gpu_name=gpu_name,
        gpu_vram_gb=round(gpu_vram, 1),
        cuda_version=torch.version.cuda or "N/A",
        torch_version=torch.__version__,
        timestamp=datetime.datetime.now().isoformat(),
    )

    print(f"\nGPU: {gpu_name}  VRAM: {gpu_vram:.1f}GB  CUDA: {report.cuda_version}")

    run_vit   = not args.qlora_only
    run_qlora = not args.vit_only

    if run_vit:
        vit_results = bench_vit()
        report.vit_results = [asdict(r) for r in vit_results]

    if run_qlora:
        qlora_results = bench_qlora(
            batch_size=args.batch,
            seq_len=args.seq_len,
        )
        report.qlora_results = [asdict(r) for r in qlora_results]

    print_summary(report)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, indent=2, ensure_ascii=False)
        print(f"\n报告已保存: {args.output}")


if __name__ == "__main__":
    main()
