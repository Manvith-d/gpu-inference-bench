import argparse, time, random, threading, csv, os
from utils.metrics import Meter
from scheduler.continuous_batcher import ContinuousBatcher
from backends.dummy_backend import DummyBackend

def get_backend(args):
    if args.backend == 'dummy':
        return DummyBackend(quantized=args.quantized, kv_cache=args.kv_cache)
    elif args.backend == 'hf':
        from backends.hf_backend import HFBackend
        return HFBackend(model_name=args.model)
    else:
        raise ValueError('Unknown backend')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', default='dummy', choices=['dummy','hf'])
    ap.add_argument('--model', default='distilgpt2')
    ap.add_argument('--users', type=int, default=200)
    ap.add_argument('--concurrency', type=int, default=32)
    ap.add_argument('--arrival', default='poisson', choices=['poisson','uniform'])
    ap.add_argument('--rate', type=float, default=50.0, help='req/sec for poisson')
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--timeout-ms', type=int, default=20)
    ap.add_argument('--max-new-tokens', type=int, default=32)
    ap.add_argument('--quantized', action='store_true')
    ap.add_argument('--kv-cache', action='store_true')
    ap.add_argument('--out', default='results/summary.csv')
    args = ap.parse_args()

    backend = get_backend(args)
    backend.warmup()
    batcher = ContinuousBatcher(backend, batch_size=args.batch_size, timeout_ms=args.timeout_ms)
    meter = Meter()

    sem = threading.Semaphore(args.concurrency)
    threads = []

    def arrival_delay():
        if args.arrival=='poisson':
            import random, math
            lam = args.rate
            return random.expovariate(lam)
        else:
            return 1.0/args.rate

    def user_task(uid:int):
        nonlocal meter
        sem.acquire()
        try:
            prompt = f'user-{uid}: what is the meaning of efficiency? '
            t_s = time.perf_counter()
            fut = batcher.submit(prompt, args.max_new_tokens)
            fut['evt'].wait()
            t_d = time.perf_counter()
            meter.add(t_s, t_d, args.max_new_tokens)
        finally:
            sem.release()

    start = time.perf_counter()
    for u in range(args.users):
        t = threading.Thread(target=user_task, args=(u,))
        t.start()
        threads.append(t)
        time.sleep(arrival_delay())

    for t in threads:
        t.join()
    end = time.perf_counter()
    batcher.shutdown()

    summ = meter.summary()
    print('Summary:', summ)
    os.makedirs('results', exist_ok=True)
    write_header = not os.path.exists(args.out)
    with open(args.out,'a',newline='') as f:
        w = csv.DictWriter(f, fieldnames=['backend','batch','quantized','kv_cache','users','concurrency','p50_ms','p90_ms','p99_ms','req_per_sec','tok_per_sec'])
        if write_header:
            w.writeheader()
        w.writerow({
            'backend': backend.name(),
            'batch': args.batch_size,
            'quantized': args.quantized,
            'kv_cache': args.kv_cache,
            'users': args.users,
            'concurrency': args.concurrency,
            'p50_ms': f"{summ['p50_ms']:.2f}",
            'p90_ms': f"{summ['p90_ms']:.2f}",
            'p99_ms': f"{summ['p99_ms']:.2f}",
            'req_per_sec': f"{summ['req_per_sec']:.2f}",
            'tok_per_sec': f"{summ['tok_per_sec']:.2f}",
        })

if __name__ == '__main__':
    main()
