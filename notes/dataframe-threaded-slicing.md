# Whole-window `readtable(::DataFrame)` is slow with threads

This is an investigation record, not design law: `DESIGN.md` is still the source
of truth. It exists so the next person who sees this does not have to repeat
the measurements. **No code was changed because of it.**

Summary: when `readtable` reads a DataFrame over its whole window, it slices
with DataFrames' `df[rows, :]`. With more than one thread, and at least 100,000
rows, that call spawns one task per column. On the test machine those tasks'
large allocations appear to hit glibc's per-thread allocator arenas: pages are
handed back to the kernel and then faulted in again, so each copy gets slower
and the whole slice gets slower. It is not slower on one thread. Threads add
cost but give no speedup here. The per-thread arenas are the leading suspect,
not a proven cause.

## Environment

| | |
|---|---|
| Julia | 1.12.7 (`--project=benchmark`) |
| DataFrames | 1.8.2 |
| CPU | Intel i7-8550U, 4 cores / 8 threads, laptop |
| Memory | 15.7 GiB; swap full (3998/3999 MiB), but memory pressure stall info (PSI) was 0 during the runs |
| Kernel / libc | Linux 6.18.44 (Manjaro), glibc 2.44; transparent huge pages set to `always` |
| Threads | **`JULIA_NUM_THREADS=4` is exported in the shell**, so every `julia` without `-t` ran with 4 threads |

The table is 1,000,000 rows × 3 columns: `time::Int` (`i ÷ 4`), `sym::String`
(100 distinct values), `qty::Float64`. That is 7.6 MiB per column.

## Symptoms

The benchmark as first reported (for PR #48) was mislabeled "single thread"; it
actually ran with 4 threads:

| path | whole window | 1% window |
|---|---|---|
| `readtable(df)` | 5.9 ms | 22 μs |
| `readtable(namedtuple)` (generic path) | 4.2 ms | 953 μs |
| `readtable(frame)` | 16 μs | 22 μs |

- Only the whole-window DataFrame run is slow. The 1% window (2,500 rows) is
  below DataFrames' threading threshold and is 43× faster than the generic path.
- It is a **median** effect, not a floor. The p10 of `load(readtable(df))` is
  2.5 ms, below the generic path's p10 of 3.6–4.0 ms. The distribution is
  bimodal: between 15% and 31% of 4-thread samples finish under 3.5 ms.
- It **disappears on one thread.** There, `load(readtable(df))` has a p50 of
  2.63–2.71 ms against 4.25–4.39 ms for the generic path, so the DataFrame path
  is the faster one.
- Every variant has p75/p90 tails of 15–195 ms. About 20–34% of samples
  included a GC (the test allocates about 23 MiB per call). GC explains the
  tails, not the median.

## Where the threads come from

`Base.getindex(df::DataFrame, rows::AbstractVector, :)` (and the
multi-column-index method) calls `_threaded_getindex`
(`DataFrames/src/dataframe/dataframe.jl:565` in 1.8.2):

```julia
if length(selected_rows) >= 100_000 && Threads.nthreads() > 1
    new_columns = Vector{AbstractVector}(undef, length(selected_columns))
    @sync for i in eachindex(new_columns)
        @spawn new_columns[i] = df_columns[selected_columns[i]][selected_rows]
    end
```

The same pattern is used by the `DataFrame` constructor when `copycols = true`
(`dataframe.jl:223`), and therefore by `copy(df)`. The single-column paths are
serial, and so is `vcat`. `getindex` has **no opt-out**.

History:

- **DataFrames #2647** (merged 2021-03-19) introduced threading in `getindex`,
  `copy` and the constructor. Its author was unsure whether threading pays off
  on short columns. Review suggested a `>= 1_000_000` threshold.
- **DataFrames #3274** (merged 2023-02-05) lowered the threshold to 100,000. The
  evidence was a single `@time` on 2 and 100 `rand` Float64 columns at 10⁵ + 1
  rows, with 4 threads.
- **DataFrames #3030** (merged 2022-06-12, from issue #2988) added a `threads`
  keyword to `select`/`transform`/`combine`/`subset`, but not to `getindex`,
  `copy` or the constructor. In #2988, a Dagger maintainer asked for
  parallelism levels that propagate through context rather than per call.

## Affected CausalFrames code

Every `c[rows, :]` over a chunk of at least 100,000 rows and more than one
column takes the threaded path:

| site | operator |
|---|---|
| `clipchunk!` (`src/operators.jl`) | `readcsv`, `readparquet`, `readjls` |
| `filterchunk` (`src/operators.jl`) | `filterrows` (Bool mask → `_findall` → threaded) |
| `takerows!` (`src/operators.jl`) | `head` |
| `settimechunk!` (`src/operators.jl`) | `settime`, `Acausal.settime` |
| `clipstart!` (`src/fill.jl`) | `forwardfill` with `tolerance` |
| `clipframechunk` (`src/table.jl`) | `readtable` over a DataFrame (every run) or a partial frame chunk |
| `readtable(::AbstractDataFrame)` sort (`src/table.jl`) | once per pipeline construction |
| `DataFrame(frame)` for a one-chunk frame (`src/frame.jl`) | `copy` → threaded constructor |

Only the `readtable` case was measured. Chunks at exactly 100,000 rows showed no
penalty (see T6), so the impact on the other sites depends on chunk size.
Parquet row groups can be much larger than `readcsv`'s roughly 4 MiB chunks.

## Tests run

All tests used BenchmarkTools with `evals = 1` and 150–200 samples, and report
quantiles. Scratch scripts are reproduced below.

### T1 — the kernel alone reproduces it, without DataFrames

The test compares a bare `@sync`/`@spawn` loop, which is the body of
`_threaded_getindex`, with a serial loop over the same `AbstractVector[]`
columns. Values are p50 in ms.

| config | `df[rows, :]` | serial loop | spawn loop | `load readtable(df)` | `load readtable(nt)` |
|---|---|---|---|---|---|
| `-t 1` | 6.22 † | 2.63 | 2.55 | 2.71 | 4.39 |
| `-t 4` | 6.58 † | 2.67 | **6.13** | **5.92** | 5.20 |
| `-t 4 --gcthreads=1` | 5.87 † | 2.59 | **5.65** | **5.69** | 4.16 |
| `-t 4` + glibc tunables ‡ | 2.93 † | 2.73 | 2.66 | 2.67 | 4.26 |

† `df[rows, :]` was the first benchmark in its process, which inflates it (see
T7). The spawn loop was not first.
‡ `GLIBC_TUNABLES=glibc.malloc.mmap_threshold=33554432:glibc.malloc.trim_threshold=536870912`.

**Reading:** spawning alone costs about 3.5 ms at 4 threads. DataFrames'
bookkeeping is not the cause.

### T2 — tasks start promptly; the copies themselves slow down

Each task timestamped its own start and its copy duration. Values are p50 over
150 calls, per column `[time, sym, qty]`, in ms.

| config | spawn: start latency | spawn: copy duration | spawn wall | serial: copy duration |
|---|---|---|---|---|
| `-t 1` | [0, 0.63, 1.95] (queued behind each other) | [0.62, 1.31, 0.64] | 2.70 | [0.60, 1.26, 0.63] |
| `-t 4` | [0, 0.06, 0.06] | **[5.20, 2.48, 1.82]** | 5.94 | [1.89, 2.07, 1.71] § |
| `-t 4 --gcthreads=1` | [0, 0.03, 0.03] | **[5.06, 2.50, 1.74]** | 5.60 | [0.62, 1.25, 0.64] |
| `-t 4` + tunables | [0, 0.03, 0.06] | [1.87, 2.49, 1.80] | 2.64 | [0.63, 1.44, 0.64] |

All the tasks ran off the calling thread.
§ See S2: this serial row was slow with default GC threads, but not with
`--gcthreads=1`.

**Reading:** scheduling latency is at most 0.06 ms. The slowdown is inside the
copy: the `Int` column takes 5.2 ms on a worker thread against 0.6 ms serially.
Even with the tunables, worker copies are about 3× slower each, and parallelism
only just recovers serial wall time. **Threading gives no speedup for this
shape on this machine.**

### T3 — page faults track the slowdown

Faults per call were counted from `/proc/self/stat` `minflt`.

| config | serial loop | spawn loop | `df[rows, :]` | `load readtable(df)` | `load readtable(nt)` |
|---|---|---|---|---|---|
| `-t 1` | 714–739 | 786 | 820–989 | 611 | 639 |
| `-t 4` | 1187–1370 | 2302 | 2485–2560 | 1429 | 1129 |
| `-t 4` + tunables | 37–68 | 697 | 327–540 | 594 | 67 |
| `-t 4`, `MALLOC_ARENA_MAX=1` | 963 | **894** | 706 | — | — |

With one arena (`MALLOC_ARENA_MAX=1`), `df[rows, :]` benchmarked last has a p50
of **2.66 ms**, the same as serial.

### T4 — TLB shootdowns: ruled out

System-wide deltas of the `TLB:` line in `/proc/interrupts` over 200 calls, with
an idle interval of the same wall time as a baseline. Shootdowns per call:

| config | serial | spawn | `df[rows, :]` | idle baselines |
|---|---|---|---|---|
| `-t 1` | 0.9 | 0.8 | 0.8 | 0.7–10.8 |
| `-t 4` | 1.2 | 2.8 | 2.6 | 0.3–0.7 |
| `-t 4` + tunables | 0.3 | 2.8 | 2.3 | 0.4–6.1 |

Spawning adds roughly 1.5 shootdowns per call, and that does **not** change when
the tunables remove the slowdown. At microseconds per shootdown, this cannot
account for milliseconds.

### T5 — GC threads

`--gcthreads=1` did not fix the spawn loop (p50 5.65 ms). It did fix the anomaly
of a slow *serial* instrumented loop (6.96 ms → 3.22 ms wall) that appeared with
the default GC threads (4 at `-t 4` on 1.12). The BenchmarkTools serial run was
unaffected either way. This is unexplained; see S2.

### T6 — no cliff at the threshold

`df[1:k, :]` p50, in ns per row, at k = 99,999 and k = 100,000:

| `-t 1` | `-t 4` | `-t 4 --gcthreads=1` | `-t 4` + tunables |
|---|---|---|---|
| 2.39 / 2.33 | 2.39 / 2.20 | 2.35 / 2.13 | 2.48 / 2.43 |

At 0.76 MiB per column, the threaded path is no slower. The cost appears with
larger allocations (7.6 MiB per column). A row threshold does not capture that.

### T7 — measurement pitfall: the first benchmark in a process

At `-t 1`, `df[rows, :]` has a p50 of **5.33 ms** when benchmarked first in the
process and **2.68 ms** when benchmarked last. At `-t 4` it is 6.50 against
6.28 ms, so the threaded slowdown is real and not an ordering artifact. Every
early `df[rows, :]` figure above was taken first. Put a warm-up benchmark first.

## Suspected causes, ranked

**S1 — per-thread malloc arenas return pages that are then faulted in again
(leading; not proven at the kernel level).** Evidence:

- only spawning triggers it (T1);
- the copies slow down, not the scheduling (T2);
- page faults roughly double (T3);
- both raising glibc's mmap and trim thresholds and forcing a single arena
  remove it (T1, T3);
- one thread is unaffected.

The presumed mechanism, not verified from the Julia runtime source:

1. An array this large is allocated through the C allocator on the thread that
   runs the task.
2. glibc serves each thread from its own arena.
3. The arrays are freed later, when Julia's GC sweeps, possibly on another
   thread.
4. glibc's dynamic mmap threshold, and arena heap trimming (`MADV_DONTNEED` /
   `munmap`), give the pages back to the kernel.
5. The next call's allocation on a worker faults in and zero-fills about 7.6 MiB
   of fresh pages again, with several threads faulting concurrently.

With one arena, or with the thresholds raised, memory is reused instead.
Transparent huge pages (`always`) change what each fault costs and were not
varied.

**S2 — interplay with GC threads (secondary, open).** A serial loop slowed down
with 4 GC threads and not with 1 (T5). This could be the same S1 mechanism
triggered by frees on GC threads, but it was not isolated.

**S3 — memory-bound copies (contributing, machine-specific).** Even with S1
removed, worker copies are about 3× slower each and give no wall-time gain (T2).
Copying 1M elements of three columns is bounded by memory bandwidth on a 4-core
laptop, so the threaded path can only ever add overhead here. A many-core server
with more memory channels may behave differently.

**Possible amplifiers, not varied:** full swap (although PSI was 0), laptop CPU
frequency scaling, and transparent huge pages set to `always`.

**Ruled out:**

- task start latency (T2);
- TLB shootdowns (T4);
- DataFrames' bookkeeping around the copy (T1);
- a threshold cliff (T6);
- GC pauses as the cause of the median (they explain the tails in every
  variant).

## Tests still to run

1. **Allocator syscalls.** Run `strace -f -c -e trace=mmap,munmap,madvise,brk,mremap`
   over 100 serial against 100 spawn calls at `-t 4`, each minus a setup-only
   run. `scratch/dfsyscalls.jl` below takes `none|serial|spawn`. strace was not
   installed. This directly tests whether pages are returned on every call.
2. **Kernel profile.** `perf stat -e page-faults` and `perf record -g` on the
   spawn loop: look for time in `clear_page_*` / `do_anonymous_page` /
   `zap_page_range`.
3. **Allocator swap.** `LD_PRELOAD` jemalloc or mimalloc at `-t 4`. If S1 is
   right, the slowdown should go away without any tunables.
4. **Transparent huge pages.** `always` against `madvise` against `never`.
5. **GC sweep threads.** `--gcthreads=4,0` against `--gcthreads=4,1` (Julia
   1.12's concurrent sweeper), to separate S2.
6. **Shape sweep.** Columns ∈ {3, 10, 100} × rows ∈ {10⁵, 10⁶, 10⁷} × threads
   ∈ {1, 4, 8}, comparing p50 of `df[rows, :]` with a serial per-column slice,
   to find where threading wins. This is the evidence an upstream threshold
   change would need.
7. **Other machines.** Repeat on a server with free swap, and on Julia 1.10 (the
   CI floor, which has a different GC).
8. **CausalFrames impact.** Rerun `benchmark/benchmarks.jl` at `-t 1` and `-t 4`,
   and add a large-row-group parquet read, to see whether `filterrows`, `head`
   and the readers are affected in practice.

## Potential for upstream improvement

**DataFrames.jl** is the most direct place:

- **Opt-out.** `getindex`, `copy` and the constructor have no `threads` control.
  Extending #3030's keyword (or a scoped setting, as #2988 asks for; ScopedValues
  exist from Julia 1.11) would let a library like CausalFrames, which already
  controls its own chunking, turn it off.
- **A better threshold.** A row count (#3274's 100,000, justified by a single
  `@time`) ignores element size, total bytes and column count, and parallelism is
  capped at `ncol` anyway. A bytes-and-`ncol` rule, or spawning `ncol - 1` tasks
  and running one column on the calling thread (which today idles in `@sync`),
  would cost less.
- **Evidence standard.** A PR should report quantiles over several shapes and
  thread counts, rather than one `@time` or a minimum. On this machine the
  opportunity for 1M × 3 is 6.2 ms → 2.6 ms p50 (about 2.3×). Wide frames on big
  machines may still benefit from threads, so the fix is adaptive, not removal.
  Test 6 is the data that proposal needs.

**Julia runtime (possible, needs prior-art search first):** if tests 1–3 confirm
S1, the interaction between cross-thread frees of large arrays in GC and glibc's
per-thread arenas is a runtime-level cost that every threaded allocator-heavy
workload pays, not just DataFrames'. Options would include reusing large buffers
per thread, or documenting allocator settings. Search JuliaLang/julia issues
before filing; none were checked here.

**glibc:** the tunables and `MALLOC_ARENA_MAX` behave as documented. This is a
configuration workaround, not a bug.

## Options inside CausalFrames (not adopted)

- **A serial row slice.** Replace `c[rows, :]` at the sites above with a helper:
  `DataFrame(AbstractVector[col[rows] for col in eachcol(c)], propertynames(c);
  copycols = false)`. The constructor does not thread when `copycols = false`.
  Measured for this shape: p50 2.59–2.73 ms at 4 threads, against 6.2 ms. Costs:
  - it gives up DataFrames' threading where that helps;
  - it drops DataFrames' note-metadata copy (CausalFrames does not use metadata).

  This is CausalFrames adapting to one machine's measurements. Run tests 6–7
  before adopting it.
- **Generic path overhead, independent of all this.** At one thread
  `readtable(namedtuple)` costs about 1.6 ms more than the DataFrame path. That
  is the per-run `issorted` (about 0.6 ms) plus `copytimes` copying through
  `copyto!` from a view: 1.75 ms against 0.69 ms for `time[rows]` in one
  measurement. `copytimes` could use `times[rows]` when the eltype already
  matches.
- **Documenting `MALLOC_ARENA_MAX=1` for users:** not appropriate as a package
  requirement.

## Reproducer

```julia
# julia --project=benchmark -t 4 repro.jl   (compare with -t 1)
using BenchmarkTools, DataFrames
n = 1_000_000
df = DataFrame(time = collect(0:n-1) .÷ 4,
    sym = [string("s", i % 100) for i in 0:n-1], qty = [1.0 + i % 7 for i in 0:n-1])
cols = AbstractVector[c for c in eachcol(df)]
serial(cols, r) = [c[r] for c in cols]
function spawned(cols, r)
    out = Vector{AbstractVector}(undef, length(cols))
    @sync for i in eachindex(cols); Threads.@spawn out[i] = cols[i][r]; end
    out
end
@benchmark serial($cols, 1:$n)              # warm-up: the first benchmark is inflated
for f in (() -> serial(cols, 1:n), () -> spawned(cols, 1:n), () -> df[1:n, :])
    b = @benchmark $f() samples = 150 evals = 1
    println(median(b).time / 1e6, " ms")
end
```

The full scripts (`dfslice.jl`, `dfthreads.jl`, `dfshootdown.jl`,
`dfsyscalls.jl`) lived in the session scratchpad. Their measurement code is the
same as the reproducer, plus:

- `minflt` from `/proc/self/stat`;
- the `TLB:` line of `/proc/interrupts`;
- per-task `time_ns()` stamps.
