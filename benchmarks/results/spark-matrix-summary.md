<!-- decode t/s (naive) divides total tokens by the longest decode span. With cold prompts prefill is serialised across slots, so an early stream is starved while a later one prefills and the naive figure understates throughput. Use concurrency.py for the corrected steady-state number and the contamination flag. -->

| mode | conc | prompt tokens | prefill t/s | decode t/s (naive) | decode t/s (per stream) | gen tokens | GPU Mem | wall s | wall t/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| plain | 1 | 1451 | 53.0 | 10.51 | 10.51 | 128 | 103.62 GiB | 39 | 3.24 |
| plain | 1 | 3941 | 51.6 | 9.05 | 9.05 | 128 | 103.62 GiB | 90 | 1.41 |
| plain | 1 | 8222 | 47.9 | 7.38 | 7.38 | 128 | 103.62 GiB | 189 | 0.68 |
| plain | 1 | 16376 | 42.3 | 5.42 | 5.42 | 128 | 103.62 GiB | 411 | 0.31 |
| plain | 1 | 33068 | 34.0 | 3.56 | 3.56 | 128 | 103.62 GiB | 1007 | 0.13 |
| plain | 1 | 65662 | 24.3 | 2.11 | 2.11 | 128 | 103.62 GiB | 2760 | 0.05 |
| mtp_no_quench | 1 | 1451 | 52.3 | 8.28 | 8.28 | 128 | 103.59 GiB | 43 | 2.97 |
| mtp_no_quench | 1 | 7924 | 48.3 | 5.96 | 5.96 | 128 | 103.62 GiB | 185 | 0.69 |
| mtp_no_quench | 1 | 32995 | 34.1 | 3.43 | 3.43 | 128 | 103.62 GiB | 1005 | 0.13 |
| plain_batched | 1 | 1451 | 51.8 | 10.39 | 10.39 | 128 | 108.88 GiB | 40 | 3.18 |
| plain_batched | 1 | 7924 | 47.5 | 7.37 | 7.37 | 128 | 108.92 GiB | 184 | 0.70 |
| plain_batched | 2 | 2×1962 | 53.0 | 9.42 | 4.73 | 256 | 108.79 GiB | 101 | 2.54 |
| plain_batched | 2 | 2×8042 | 39.5 | 1.02 | 2.71 | 256 | 108.82 GiB | 433 | 0.59 |
| plain_batched | 4 | 4×1960 | 39.6 | 2.69 | 2.03 | 512 | 108.82 GiB | 243 | 2.10 |
| plain_batched | 4 | 4×8015 | 39.3 | 1.07 | 1.58 | 512 | 108.79 GiB | 867 | 0.59 |
| plain_batched | 8 | 8×1962 | 36.6 | 2.63 | 1.21 | 1024 | 108.82 GiB | 509 | 2.01 |
| plain_batched | 8 | 8×8048 | 41.6 | 1.96 | 0.73 | 1024 | 108.82 GiB | 1635 | 0.63 |
| plain_batched_short | 1 | 208 | 43.1 | 11.12 | 11.12 | 128 | 104.32 GiB | 16 | 7.88 |
| plain_batched_short | 2 | 2×268 | 41.2 | 8.56 | 4.90 | 256 | 104.32 GiB | 36 | 7.10 |
| plain_batched_short | 4 | 4×273 | 36.9 | 7.67 | 2.51 | 512 | 104.32 GiB | 74 | 6.90 |
| plain_batched_short | 8 | 8×267 | 34.9 | 7.76 | 1.35 | 1024 | 104.32 GiB | 149 | 6.89 |
