import os
import time

import jax
import numpy as np
import pandas as pd

from cartm import AttentiveTopicModel
from model_no_N_wt import AttentiveTopicModelNoNWT
from cartm.preprocessing import BatchedCorpusLoader
from common import prepare_data


def block_tree(x):
    jax.tree_util.tree_map(
        lambda y: y.block_until_ready() if hasattr(y, "block_until_ready") else y,
        x,
    )
    return x


def benchmark_any_fn(
    fn,
    *args,
    warmup_runs: int = 1,
    bench_runs: int = 5,
    name: str | None = None,
    device: str = 'cpu',
    move_to_device: bool = False,
    **kwargs,
):
    if name is None:
        name = getattr(fn, "__name__", "fn")

    if move_to_device:
        if device is None:
            raise ValueError("If move_to_device=True, you must provide device.")
        device = jax.devices(backend=device)[0]
        args = jax.device_put(args, device)
        kwargs = jax.device_put(kwargs, device)

    # Compile + first execution
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    block_tree(out)
    t1 = time.perf_counter()
    compile_plus_run = t1 - t0

    # Warmup runs
    for _ in range(warmup_runs):
        out = fn(*args, **kwargs)
        block_tree(out)
    print("=== Warmup ok ===")

    # Steady-state runs
    times = []
    for _ in range(bench_runs):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        block_tree(out)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return {
        "name": name,
        "compile_plus_run_sec": compile_plus_run,
        "steady_mean_sec": sum(times) / len(times),
        "steady_min_sec": min(times),
        "steady_max_sec": max(times),
        "steady_std": np.std(times),
        "runs": bench_runs,
    }


def add_throughput(res, num_items: int, item_name: str = "items"):
    res = dict(res)
    res[f"{item_name}_per_sec_mean"] = num_items / res["steady_mean_sec"]
    res[f"{item_name}_per_sec_max"] = num_items / res["steady_min_sec"]
    return res


def print_benchmark_result(res):
    print(f"[{res['name']}]")
    print(f"  compile + first run: {res['compile_plus_run_sec']:.6f} s")
    print(f"  steady mean       : {res['steady_mean_sec']:.6f} s")
    print(f"  steady min        : {res['steady_min_sec']:.6f} s")
    print(f"  steady max        : {res['steady_max_sec']:.6f} s")
    print(f"  runs              : {res['runs']}")

    extra_keys = [
        k for k in res.keys()
        if k not in {
            "name",
            "compile_plus_run_sec",
            "steady_mean_sec",
            "steady_min_sec",
            "steady_max_sec",
            "runs",
        }
    ]
    for k in sorted(extra_keys):
        print(f"  {k:<18}: {res[k]:.2f}")


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    data = prepare_data("20ng")

    def run_model(model, data, ctx_bounds):
        model.fit(
            data=data,
            ctx_bounds=ctx_bounds,
            max_iter=50,
            verbose=0,
            seed=np.random.randint(0, 1000),
        )

    def run_model_batched(model, batches):
        model.fit(
            data=batches,
            max_iter=50,
            verbose=0,
            seed=np.random.randint(0, 1000),
        )

    benchmark_results = []
    for n_topics in [100, 10]:
        for ctx_len in [1000, 100, 10]:
            for batch_size in [10_000]:
                for model_type in [AttentiveTopicModelNoNWT, AttentiveTopicModel]:
                    for device in ['gpu']:
                        # prepare batches
                        loader = BatchedCorpusLoader(
                            data=data.train_tokens,
                            doc_bounds=data.train_bounds,
                            batch_size=batch_size,
                        )

                        # prepare model
                        model = model_type(
                            vocab_size=len(data.vocab),
                            ctx_len=ctx_len,
                            n_topics=n_topics,
                        )

                        # move data to device
                        jax_device = jax.devices(device)[0]
                        tokenized_data = jax.device_put(data.train_tokens, device=jax_device)
                        document_bounds = jax.device_put(data.train_bounds, device=jax_device)
                        loader._batches = [
                            jax.device_put(batch, device=jax_device) for batch in loader._batches
                        ]

                        config = {
                            "model": model_type.__name__,
                            "ctx_len": ctx_len,
                            "n_topics": n_topics,
                            "n_batches": len(loader),
                            "batch_size": batch_size,
                        }

                        # batched benchmark
                        res = benchmark_any_fn(
                            run_model_batched,
                            model,
                            loader,
                            name=f"{model_type.__name__}, {ctx_len=}, {n_topics=}",
                        )
                        print_benchmark_result(res)
                        res.update(config)
                        benchmark_results.append(res)
    pd.DataFrame(benchmark_results).to_csv('./results/benchmark/summary.csv')
