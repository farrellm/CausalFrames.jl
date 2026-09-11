# CausalFrames.jl — module map

DESIGN.md's "Module layout" table is the canonical index; this file records the
design rationale and performance constraints behind each module.

- `src/context.jl` / `src/chunks.jl` — `Context{T}`, the `[start, stop)`
  evaluation window, and the internal chunk protocol (`ChunkSource`,
  `chunkmap`): a single-pass lazy iterator of non-empty DataFrame chunks,
  consumers taking ownership of what they're yielded; empty chunks are
  filtered out here so all downstream code may assume a chunk has rows
- `src/frame.jl` — `CausalFrame{T}`: opaque, backed by a vector of
  time-disjoint DataFrame chunks; invariants checked in the public inner
  constructor, while `load`/`stream` build through a `Trusted`-token
  constructor (O(n) scans skipped; O(1) cross-chunk/bounds guards kept);
  column-access Tables.jl interface with a cheap `Tables.schema`;
  `DataFrame(frame)` is the copy point
- `src/pipeline.jl` — `CausalPipeline{F}` (lazy `Context -> iterator of
  DataFrame chunks`; the run function's type is a parameter, not an abstract
  `Function` field), `load` (drains into one frame without copying — the
  only operation that materializes the whole window), `stream` (one frame
  per chunk)
- `src/operators.jl` — sources return a `CausalPipeline`; transforms are
  curried (`filterrows(pred)` returns `CausalPipeline -> CausalPipeline`)
  so both chain with `|>`; row functions run over concretely typed column
  table rows behind a per-chunk function barrier, never `DataFrameRow`s;
  `clipchunk!` (rename, resolve `:time`, sortedness, clip, convert) and the
  `ChunkSink` background writer are shared with the parquet operators.
  The three column operators share one selector vocabulary and one per-run,
  schema-keyed resolution memo; `reordercolumns` is the only one that reads the
  selectors as an *order*, which is why `foreachselector` exists beside
  `foreachliteral` — matching can stop at the first hit and ignore pattern
  leaves, ordering can do neither.
  `readcsv` never infers types — every column is `String` unless `types`
  opts it into a concrete one (`notes/readcsv-stringtype.md` records why
  CSV.jl's own `stringtype` default stays out). `lag` shifts times via the
  shared `shiftchunk!` and widens the context in `lagcontext`, the mirror
  of `Acausal.lead`; `settime` is the general, per-row form of that shift,
  sharing `settimechunk!` with `Acausal.settime` the same way, but it cannot
  widen the context (the shift is data-dependent) and so needs three
  independent order checks instead — see DESIGN.md's "Retiming". `head` is
  the one transform not built on `chunkmap`, which cannot terminate early:
  it drives the upstream iterator from a mutable `HeadProducer` behind a
  `ChunkSource`, the `CSVProducer` shape. If a second early-exit operator
  ever arrives, that is the point to extract a `chunks.jl` primitive
- `src/merge.jl` — `Base.merge(ps::CausalPipeline...)`, the n-ary
  time-interleaving source (extends Base rather than shadowing it; no
  zero-arg form, which would capture `merge()`): one `MergeCursor` per
  pipeline buffering a chunk, `pickwinner` choosing by `(time, argument
  index)`, and piece-at-a-time claiming (`searchsorted*` runs, no data moved
  until `batchsize` rows are pending, then one allocation per output column
  and one copy per row; a lone whole-chunk piece is adopted, or passed
  through untouched when it already has the union schema)
- `src/parquet.jl` — `readparquet`/`writeparquet`: the API, the docstrings
  and backend selection (`resolvebackend`, `parquetproducer`, `parquetsink`,
  `backendloaded`), none of which name a backend. DuckDB and Parquet2 are
  **weak** deps behind package extensions and **either one serves both
  directions**: reading prefers DuckDB (filtered streaming query) and falls
  back to Parquet2 (row groups skipped by their own statistics); writing
  prefers Parquet2 (row groups written as the stream flows by) and falls
  back to DuckDB (stream staged in a temp table, one `COPY` at the end).
  Skipping is always an optimization — chunks are clipped again on arrival,
  so results never depend on it. `backend = :duckdb`/`:parquet2` forces the
  choice, which is how the tests cover all four combinations in one process;
  working on parquet means loading `DuckDB`/`Parquet2` in the session first
- `src/jls.jl` — `writejls`/`readjls`, the untyped persistence pair: a header
  record then one `serialize`d DataFrame per chunk, each its own `serialize`
  call so records are independent. The sink is `ChunkSink` with a serializing
  write loop; the source is a `CSVProducer`-shaped `JLSProducer` over
  `clipchunk!`. It exists for columns CSV and parquet cannot encode (fitted
  models), so the file is Julia/package-version-fragile by design — not an
  interchange format
- `src/summarizers.jl` — `Summarizer` (immutable config, output column name in
  a type parameter) and `SummarizerState` (running state, typed from the input
  schema), plus their unexported interface: `emptyvalue`, `fresh`, `fresh!`,
  `update!`, `value`, `widenstate`, `dependencies`, `combine!`, `downdate!`,
  `isinvertible`. Within that:
  - `fresh!` zeroes a state in place and returns it (default `fresh(st)`, so
    it is opt-in for a custom summarizer). It exists because the transforms
    zero a state tuple per cycle, per interval, and per window query — see
    DESIGN.md's "Reusing state". Callers must use the return value
  - the structured subtypes are `MonoidSummarizer` (states combine
    associatively over stream-ordered ranges via `combine!`, fresh state as
    the identity) and `GroupSummarizer <: MonoidSummarizer` (also invertible,
    via `downdate!`). This split is what `rolling.jl` dispatches its window
    algorithm on, so which one a new summarizer claims is a performance
    decision, not a taxonomy one
  - `Min`/`Max`/`First`/`Last` share one state type (`TrackState`,
    parameterized by the combiner) and are monoids only, as is `Product`; the
    accumulators and every dependent summarizer are groups
  - the dependent summarizers (`Moment`, `Mean`, `Variance`, `Std`,
    `Covariance`, `Correlation`, `LinearRegression`) carry no state of their
    own — their state structs are empty. They declare `dependencies` and read
    those values back through the two-argument `value(st, vals)`
  - a summarizer whose value is symmetric in two columns (`DotProduct`,
    `Covariance`, every pairwise term in `LinearRegression`) folds under the
    `isless`-sorted argument order via `canonicaldot`/`canonicaldotname`, but
    still emits the column name the caller asked for. For an accumulator that
    means the reversed form becomes a dependent over the canonical one, using
    the reusable fieldless `AliasState{N,D}`; for something already dependent
    it just names the canonical form. DESIGN.md's "Symmetric summarizers" is
    the rule to follow when adding another one
  - `LinearRegression` is the outlier among them in two ways: it is the only
    one emitting more than one column (a coefficient and a t statistic per
    term, plus `r2`/`stderr`/`n`, so its output names are a tuple parameter
    rather than a single `N`), and the only one that allocates — `K >= 2`
    builds a `K x K` workspace per emitted row for the Cholesky, while `K = 1`
    takes a closed form over scalars. It reads its dependencies back through
    `NamedTuple{names}(vals)` projections rather than by indexing with a
    symbol, so no name is a runtime value on the emission path
  - the sum family `Sum`/`SumPower`/`DotProduct` shares one plain and one
    compensated state over a term functor (`ColumnTerm`/`PowerTerm`/
    `PairProductTerm`, terms formed at accumulator width). The `Compensated`
    pair runs Neumaier summation over the finite terms and counts NaN/±Inf
    separately, reconstructing the IEEE result in `value` — that separation is
    what lets a rolling window evict a nonfinite row cleanly and stay on the
    running path. `BigFloat` is excluded on purpose: compensation buys nothing
    at arbitrary precision, and a non-isbits `Compensated` would allocate per
    row
  - `FitModel` (MLJ) is the one summarizer whose value is an object: its state
    buffers the folded rows in concretely typed vectors and fits at `value`
    time through the `fitmodel` hook (src/models.jl), building the
    `NamedTuple{(N,),Tuple{FittedModel{P,M}}}` from type parameters so the
    column is concrete though the fit is opaque. It is plain `Summarizer`
    (re-fold everywhere). `fresh!` empties the buffers keeping capacity, which
    is only safe because `value` hands the model copies
  - a `Missing`-admitting accumulator type gets the flat `Optional*` counting
    states over the non-missing type, counting `missing` terms exactly as the
    compensated states count nonfinites, so the accumulator stays invertible —
    no `Union{Missing,_}` accumulation field, only in `value`'s return
- `src/summarize.jl` — the folding kernels and the transforms `summarize`,
  `summarizecycles`, `addsummarycolumns`; `prototypes` expands dependencies
  topologically and returns the requested output names, which ride through
  the kernels in a `Val` to project hidden dependencies out of the output;
  per-run mutable state lives in the `SummaryFold` struct, never in
  reassigned closure captures (those get boxed). The per-key states live in a
  `GroupTable{K,S}` (both parameters concrete, as the bare `Dict` was) that
  also carries the reused emission buffer and the pool of retired state
  tuples `closecycle!` retires into and `groupstates!` zeroes back out of —
  the keyed cycle fold closes a table per timestamp, so rebuilding one per
  cycle was the package's heaviest allocation site
- `src/join.jl` — `asofjoin`, the binary as-of join transform: a chunkmap
  over the left stream pulls right chunks on demand (two-pointer merge, per
  left row) into a concretely typed per-key store; `tolerance` widens the
  right context by `start - tolerance` (the one place times are subtracted).
  The store is a `Dict{K,Int}` of slot numbers over a `Vector{V}` of rows, and
  a match is a `Vector{V}` plus a `Vector{Bool}` mask (unmatched slots left
  undefined, hence `convertmatches` for a mid-chunk widening) — both to keep
  `V` out of a `Union`, which Julia cannot store inline unless every member is
  isbits, so a `String` or `Missing`-admitting right column cost one box per
  left row. See DESIGN.md's "Representing a match"; the two zero-allocation
  kernel tests are what stop it regressing
- `src/lastrow.jl` — `lastrow`, the last-row-per-key transform: `summarize`'s
  fold-and-flush-at-`stop` shape over `join.jl`'s store rather than a
  `GroupTable`, since it keeps whole rows and a `Dict{K,V}` would box one per
  row for any non-isbits `V`. No `found` mask (every slot is written the instant
  it is claimed, so widening is a plain `convert`), and the keyless path skips
  the store entirely — the chunk's last row is the last row so far, so it costs
  nothing per row. Output schema is the input's exactly, `:time` overwritten
  with `stop` via `merge`, which preserves the column position
- `src/fill.jl` — the two missing-value fills. `fillmissing` is row-wise and
  stateless (a constant per column, so `fillcolumn` is the whole hot path, and
  the output eltype narrows — the one place in the package that does, via
  `promote_type(nonmissingtype(T), typeof(value))`). `forwardfill` carries the
  last non-missing value across rows, chunks and keys, so it joins the stateful
  operators DESIGN.md's streaming section lists. Its state is a `FillCell` per
  (key, *column*), not a row per key: a forward fill is column-independent, so
  `lastrow`'s whole-row store cannot serve it. The cells are mutable, which is
  why the store is a plain `Dict{K,NamedTuple}` rather than join.jl's
  `Dict{K,Int}` over a slot vector — a lookup already answers with a pointer,
  so there is no `Union{Nothing,V}` to box (`KeyBuffer`'s reason). A selected
  column whose promoted type admits no `Missing` gets `nothing` instead of a
  replacement column, which folds the write out of the unrolled kernel.
  `tolerance` widens the input context as `asofjoin` does, so `clipstart!` has
  to drop the pre-`start` rows the fill was allowed to see
- `src/segtree.jl` — the monoid segment tree behind the rolling tree mode:
  implicit array tree of `combine!`d partial state tuples, append-only rows,
  logical front expiry (`head`), amortized rebuilds, order-preserving
  two-accumulator range queries (`treepush!`, `treequery`, `windowstart`).
  The tree owns its two query accumulators and its node vector: `treequery`
  returns **borrowed** scratch, valid only until that tree's next query, and
  `rebuild!` reuses the nodes and compacts the rows in place whenever the
  capacity has not moved — the steady state under a short window, where
  rebuilds fire every few appends rather than amortizing away
- `src/rolling.jl` — `addrollingcolumns` picks its window algorithm from the
  expanded prototype tuple's structure: all-group → per-key running states
  with per-window eviction heads, O(1)/row (`rollsegmentrunning!`);
  all-monoid → per-key segment trees, O(log n)/row (`rollsegmenttree!`);
  otherwise the re-fold baseline (`rollsegment!`, the differential-test
  oracle). Running demotes to tree when widening lets `missing` into an
  accumulator (`isinvertible`); float sums stay running because the
  compensated states evict NaN/±Inf rows cleanly; widening rebuilds
  structures from live rows
- `src/intervalize.jl` — `intervalize`, the third binary transform: summarize
  over the intervals a `clock` pipeline defines (`[bₖ, bₖ₊₁)`, timestamped at
  `bₖ₊₁`). The `summarizecycles` fold with the close trigger driven by clock
  boundaries; keyless emits a regular grid (empty intervals get `emptyvalue`s,
  output types promoted via the shared `promotedvaluetype`), keyed is sparse
  (present keys only, via `closecycle!`). The `chunkmap` over the data stream
  pulls clock boundaries into a concrete `Vector{T}` per chunk (the pull is the
  only dynamism; the per-row kernel stays dispatch-free), reusing `SummaryFold`
  whole; `closelast` closes the trailing partial at `stop`
- `src/windows.jl` — `summarizewindows`, the clock-sampled trailing window
  (`[τ - lookback, τ)` at each tick): `intervalize`'s driver (`IntervalCursor`,
  a concrete tick vector per chunk) over `addrollingcolumns`' row buffer and
  eviction head. Running mode (`RunningGroup`s, update!/downdate!) for
  invertible group sets, re-fold through a `GroupTable` pool otherwise — no
  tree mode, since windows are queried per tick rather than per row. Keyed
  output is sparse plus one *vanish* row of empty values when a key's window
  empties, decided against the previous tick's *emitted* keys; that row is what
  stops a per-key as-of consumer (`applymodels`) from using a stale summary
- `src/models.jl` — the MLJ operators (`applymodels`, `addpredictions`,
  `modelreports`) and the five hooks the extension implements (`ismodel`,
  `fitmodel`, `predictmodel`, `savefitresult`, `restorefitresult`). It names no
  MLJ type, the `parquet.jl` split.
  - `applymodels` is `asofjoin`'s store and kernel unchanged
    (`AsofJoinConfig.op` names it in errors), plus a predict step: matched rows
    are grouped by model identity behind a function barrier, each distinct
    model gets one `predictmodel` call per chunk over views, and the results
    are scattered into one promoted `Union{Missing,E}` column.
  - `addpredictions` is pure composition — `summarizewindows` into
    `applymodels(strict = false)`, which is sound because the windows are
    half-open.
  - `FittedModel`'s custom serializer routes fitresults through MLJ's
    save/restore.
- `src/acausal.jl` — the `Acausal` submodule (`futurejoin`, `lead`, `settime`),
  reached only through `using CausalFrames.Acausal` and never re-exported, so
  acausality is always an explicit opt-in. `settime` goes further and is not in
  the submodule's `export` list at all: the top level exports that name too, and
  a name exported by two `using`d modules is an error to use unqualified — so
  exporting it here would break the *causal* `settime` for anyone who also
  imported `Acausal`. Reach it as `CausalFrames.Acausal.settime`; a test pins
  its absence from `names(Acausal)`. `futurejoin` mirrors `asofjoin`'s
  streaming machinery with the match direction, the tie-break, and the context
  widening (`stop + tolerance`) all inverted; because it matches the *earliest*
  qualifying right row it must buffer right rows per key until a left row
  consumes or outruns them, and proving a key has no future match drains the
  right stream — worst case O(right rows), against `asofjoin`'s O(keys) store.
  It shares the `Vector{V}` + mask match buffer but needs no store rework of
  its own: `KeyBuffer` is mutable, so a lookup already answers with a pointer
  and never boxed
- `src/precompile.jl` — the PrecompileTools workload over the main pipeline
  paths; every new operator adds a path here
