#!/usr/bin/env python3
"""
Basic throughput and latency benchmark for SwarmRunner / vLLM.

Sends a configurable number of concurrent requests and measures:
  - Time to first token (TTFT)
  - Total generation time
  - Tokens per second
  - Request success rate

Usage:
    python scripts/benchmark.py [--url URL] [--requests N] [--concurrency C] [--model MODEL]

Environment variables:
    SWARM_RUNNER_PORT - Router port (default: 8100)
"""

import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass, field

import httpx


@dataclass
class RequestResult:
    success: bool
    status_code: int = 0
    ttft_ms: float = 0.0
    total_ms: float = 0.0
    tokens_generated: int = 0
    error: str = ""


@dataclass
class BenchmarkStats:
    results: list[RequestResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def successes(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def failures(self) -> int:
        return self.total - self.successes

    @property
    def avg_ttft_ms(self) -> float:
        vals = [r.ttft_ms for r in self.results if r.success and r.ttft_ms > 0]
        return sum(vals) / len(vals) if vals else 0

    @property
    def avg_total_ms(self) -> float:
        vals = [r.total_ms for r in self.results if r.success]
        return sum(vals) / len(vals) if vals else 0

    @property
    def p50_total_ms(self) -> float:
        vals = sorted(r.total_ms for r in self.results if r.success)
        if not vals:
            return 0
        return vals[len(vals) // 2]

    @property
    def p99_total_ms(self) -> float:
        vals = sorted(r.total_ms for r in self.results if r.success)
        if not vals:
            return 0
        idx = min(int(len(vals) * 0.99), len(vals) - 1)
        return vals[idx]

    @property
    def total_tokens(self) -> int:
        return sum(r.tokens_generated for r in self.results if r.success)

    @property
    def tokens_per_second(self) -> float:
        total_time = sum(r.total_ms for r in self.results if r.success) / 1000
        return self.total_tokens / total_time if total_time > 0 else 0


PROMPT = "Explain the concept of LoRA fine-tuning in three sentences."


async def send_request(
    client: httpx.AsyncClient, url: str, model: str, stream: bool
) -> RequestResult:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 128,
        "temperature": 0.0,
        "stream": stream,
    }

    start = time.perf_counter()
    ttft = 0.0
    tokens = 0

    try:
        if stream:
            async with client.stream(
                "POST", f"{url}/v1/chat/completions", json=payload, timeout=120.0
            ) as resp:
                if resp.status_code != 200:
                    body = b""
                    async for chunk in resp.aiter_bytes():
                        body += chunk
                    return RequestResult(
                        success=False,
                        status_code=resp.status_code,
                        error=body.decode(errors="replace")[:200],
                    )

                first_token = True
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk_data = json.loads(data)
                        delta = chunk_data["choices"][0].get("delta", {})
                        if delta.get("content"):
                            if first_token:
                                ttft = (time.perf_counter() - start) * 1000
                                first_token = False
                            tokens += 1
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

            total = (time.perf_counter() - start) * 1000
            return RequestResult(
                success=True,
                status_code=200,
                ttft_ms=ttft,
                total_ms=total,
                tokens_generated=tokens,
            )
        else:
            resp = await client.post(
                f"{url}/v1/chat/completions", json=payload, timeout=120.0
            )
            total = (time.perf_counter() - start) * 1000

            if resp.status_code != 200:
                return RequestResult(
                    success=False,
                    status_code=resp.status_code,
                    error=resp.text[:200],
                )

            data = resp.json()
            tokens = data.get("usage", {}).get("completion_tokens", 0)

            return RequestResult(
                success=True,
                status_code=200,
                ttft_ms=total,  # non-streaming: TTFT ~ total
                total_ms=total,
                tokens_generated=tokens,
            )

    except Exception as e:
        total = (time.perf_counter() - start) * 1000
        return RequestResult(success=False, total_ms=total, error=str(e)[:200])


async def run_benchmark(
    url: str, model: str, num_requests: int, concurrency: int, stream: bool
):
    stats = BenchmarkStats()
    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient() as client:

        async def bounded_request():
            async with semaphore:
                result = await send_request(client, url, model, stream)
                stats.results.append(result)

        print(f"Running {num_requests} requests (concurrency={concurrency}, stream={stream})")
        print(f"Target: {url}, Model: {model}")
        print("-" * 60)

        wall_start = time.perf_counter()
        tasks = [asyncio.create_task(bounded_request()) for _ in range(num_requests)]
        await asyncio.gather(*tasks)
        wall_time = time.perf_counter() - wall_start

    print(f"\nResults ({stats.successes}/{stats.total} succeeded):")
    print(f"  Wall time:       {wall_time:.1f}s")
    print(f"  Avg TTFT:        {stats.avg_ttft_ms:.1f}ms")
    print(f"  Avg latency:     {stats.avg_total_ms:.1f}ms")
    print(f"  P50 latency:     {stats.p50_total_ms:.1f}ms")
    print(f"  P99 latency:     {stats.p99_total_ms:.1f}ms")
    print(f"  Total tokens:    {stats.total_tokens}")
    print(f"  Throughput:      {stats.tokens_per_second:.1f} tok/s")
    print(f"  RPS:             {stats.successes / wall_time:.1f} req/s")

    if stats.failures > 0:
        print(f"\nFailures ({stats.failures}):")
        for r in stats.results:
            if not r.success:
                print(f"  [{r.status_code}] {r.error}")


def main():
    default_port = os.environ.get("SWARM_RUNNER_PORT", "8100")
    default_model = os.environ.get("SWARM_BASE_MODEL", "Qwen/Qwen3.5-0.8B-GPTQ-Int4")

    parser = argparse.ArgumentParser(description="SwarmRunner benchmark")
    parser.add_argument("--url", default=f"http://localhost:{default_port}", help="Base URL")
    parser.add_argument("--model", default=default_model, help="Model name")
    parser.add_argument("--requests", type=int, default=20, help="Total requests")
    parser.add_argument("--concurrency", type=int, default=4, help="Concurrent requests")
    parser.add_argument("--stream", action="store_true", help="Use streaming")
    args = parser.parse_args()

    asyncio.run(
        run_benchmark(args.url, args.model, args.requests, args.concurrency, args.stream)
    )


if __name__ == "__main__":
    main()
