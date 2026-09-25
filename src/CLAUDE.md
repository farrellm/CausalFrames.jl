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
  per chunk), `scan` (drains for side effects such as `writecsv`, running
  `load`'s O(1)-per-chunk `checkchunk` guards without building a frame)
- `src/operators.jl` — sources return a `CausalPipeline`; transforms are
  curried (`filterrows(pred)` returns `CausalPipeline -> CausalPipeline`)
  so both chain with `|>`; row functions run over concretely typed column
  table rows behind a per-chunk function barrier, never `DataFrameRow`s.
  - shared with the parquet and JLS operators: `clipchunk!` (rename, resolve
    `:time`, missing times via `table.jl`'s `presentrows` barrier, sortedness,
    clip, convert) and the `ChunkSink` background writer. `sinkchunk` queues
    the chunk for the writer and passes downstream `DataFrame(c; copycols =
    false)`, a private index over the same vectors: consumers may mutate a
    chunk's column index, but no operator mutates a column vector in place, so
    sharing the vectors is sound. That is the `writecsv` hand-off argument
    other entries cite
  - `CSVProducer` is the stateful-source shape (pull state in mutable fields
    behind a `ChunkSource`) reused by `ConcatProducer`, `HeadProducer`,
    `TableProducer` and `JLSProducer`. `concatenate` is sequential and checks
    that its pipelines arrive in time order; `concatenate()` is `emptyframe()`.
    `clock` is the timestamp-only tick source that `intervalize` and
    `summarizewindows` sample on
  - `readcsv` never infers types — every column is `String` unless `types`
    opts it into a concrete one (`notes/readcsv-stringtype.md` records why
    CSV.jl's own `stringtype` default stays out)
  - the three column operators share one selector vocabulary and one per-run,
    schema-keyed resolution memo; `reordercolumns` is the only one that reads
    the selectors as an *order*, which is why `foreachselector` exists beside
    `foreachliteral` — matching can stop at the first hit and ignore pattern
    leaves, ordering can do neither
  - `lag` shifts times via the shared `shiftchunk!` and widens the context in
    `lagcontext`, the mirror of `Acausal.lead`; `settime` is the general,
    per-row form of that shift, sharing `settimechunk!` with `Acausal.settime`
    the same way, but it cannot widen the context (the shift is
    data-dependent) and so needs three independent order checks instead — see
    DESIGN.md's "Retiming"
  - `head` is the one transform not built on `chunkmap`, which cannot
    terminate early: it drives the upstream iterator from a mutable
    `HeadProducer` behind a `ChunkSource`. If a second early-exit operator
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
  working on parquet means loading `DuckDB`/`Parquet2` in the session first.
  `sort = true` (here and on `readcsv`) cannot stream, so its read path is
  `gatherchunk!`/`sortgathered` in `operators.jl` — scan every chunk for
  in-window rows, stable-sort once, emit one chunk — except that DuckDB pushes
  the sort into SQL, tie-broken by its virtual `file_row_number` (its `ORDER
  BY` is unstable); see DESIGN.md's "Sorting a file source"
- `src/jls.jl` — `writejls`/`readjls`, the untyped persistence pair: a header
  record then one `serialize`d DataFrame per chunk, each its own `serialize`
  call so records are independent. The sink is `ChunkSink` with a serializing
  write loop; the source is a `CSVProducer`-shaped `JLSProducer` over
  `clipchunk!`. It exists for columns CSV and parquet cannot encode (fitted
  models), so the file is Julia/package-version-fragile by design — not an
  interchange format
- `src/table.jl` — `readtable`, the in-memory source, on three paths sharing the
  `windowbounds` clip. A generic Tables.jl table is resolved per run and per
  partition by a `CSVProducer`-shaped `TableProducer`, with `tablerows` the typed
  barrier and only in-window rows copied. An `AbstractDataFrame` is resolved once
  at construction into a private index over the caller's vectors and then read
  by the frame path with `share = false` — every run slices, so a loaded frame
  never aliases the caller — until a sort has made owned copies. A `CausalFrame`
  is read by `framepipeline`: a binary search over chunk bounds, whole-window
  chunks handed on as `copycols = false` indexes over the frame's own vectors
  (sound by the `writecsv` hand-off argument). The frame-context check and the
  auto-`closed` rule live there too; see DESIGN.md's "Tables as sources"
- `src/summarizers.jl` — `Summarizer` (immutable config, output column name in
  a type parameter) and `SummarizerState` (running state, typed from the input
  schema), plus their unexported interface: `emptyvalue`, `fresh`, `fresh!`,
  `freshwindowed`, `update!`, `value`, `widenstate`, `dependencies`,
  `combine!`, `downdate!`, `isinvertible`. Within that:
  - `fresh!` zeroes a state in place and returns it (default `fresh(st)`, so
    it is opt-in for a custom summarizer). It exists because the transforms
    zero a state tuple per cycle, per interval, and per window query — see
    DESIGN.md's "Reusing state". Callers must use the return value
  - the structured subtypes are `MonoidSummarizer` (states combine
    associatively over stream-ordered ranges via `combine!`, fresh state as
    the identity) and `GroupSummarizer <: MonoidSummarizer` (the windowed state
    removes rows via `downdate!`). This split is what `tiers.jl` puts each
    accumulator's window tier on, so which one a new summarizer claims is a
    performance decision, not a taxonomy one
  - `downdate!` is only ever handed the **oldest** row still folded — every
    caller evicts FIFO per group, and any new caller must too. States rely on
    it: `AgeWeightedSum` takes back the evicted row's weight `n - 1`, and the
    windowed deques pop their front by sequence number, and windowed `Last`
    only counts
  - `freshwindowed` (default `fresh`) is the state only the running window tier
    slides, so a group whose ordinary state cannot invert keeps a fixed-size
    state everywhere else. `Min`/`Max`/`First`/`Last` share one ordinary state
    (`TrackState`, parameterized by the combiner) and one windowed state
    (`WindowTrackState`, a monotonic deque under the same combiner; First keeps
    the whole window) — except Last, whose windowed state is a count plus the
    newest value (`WindowLastState`), measured 3-4x cheaper per row than the
    deque holding its one value. `CountDistinct` folds a `Set` but slides
    a `Dict{T,Int}` of counts: measured, a public-API count increment hashes
    twice and is 1.4-1.7x slower than `push!` on the non-window folds (DESIGN.md
    has the table). A windowed state is never combined or widened. `Product`
    is the only built-in monoid that is not a group
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
  - `AgeWeightedSum` (`Σ k·y`, k the row's age) cannot be a term functor — its
    update reads its own `S₁` — so it has its own plain and compensated states,
    reusing the `Compensated` helpers, with `missing` counted through a type
    flag `M` rather than two more Optional* types. `S₂`'s nonfinite
    classification is `S₁`'s minus the newest row's (`newest`), whose weight is
    an exact 0
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
  cycle was the package's heaviest allocation site. A declared `keyset` (a
  `KeySet`: the declared keys plus a `Dict{K,Int}` slot index, built at
  construction so the output key type is the declaration's) swaps the table
  for `DenseGroups`, a state tuple per slot plus a `folded` flag. Dense output
  closes *every* key every time, so there is nothing for the table's Dict,
  sort and pool to save; `closedense!` just walks the slots. The slot lookup is
  also the undeclared-key check, which is why it costs the dense fold nothing
  over the sparse one. Data keys are looked up without conversion, so the
  declared key type need only be `isequal` to the data's
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
- `src/lookupjoin.jl` — `lookupjoin`, the key-only join against a timeless
  in-memory table. Not a stream: the table is copied and indexed once, at
  construction, into a `LookupJoin` (a `Dict{K,Int}` of table rows plus the value
  columns as a concrete NamedTuple, captured by the closures so each chunk's call
  is statically dispatched), and the transform is a stateless `chunkmap` — so,
  unlike the as-of joins, it keeps chunk concatenation over split contexts. Per
  chunk `lookuprows!` fills a `Vector{Int}` of rows (0 = unmatched; `Int`s so a
  `String` key costs nothing per row — the zero-allocation test pins it), then
  each value column is gathered. `unmatched` is a singleton mode type picked at
  construction: `:missing` gathers `Union{Missing,T}` (`gathermissing`),
  `:error`/`:drop` keep `T`, so the schema is the keyword's, never the data's.
  Shares `prefixleft!`'s field form with `asofjoin`
- `src/lastrow.jl` — `lastrow`, the last-row-per-key transform: `summarize`'s
  fold-and-flush-at-`stop` shape over `join.jl`'s store rather than a
  `GroupTable`, since it keeps whole rows and a `Dict{K,V}` would box one per
  row for any non-isbits `V`. No `found` mask (every slot is written the instant
  it is claimed, so widening is a plain `convert`), and the keyless path skips
  the store entirely — the chunk's last row is the last row so far, so it costs
  nothing per row. Output schema is the input's exactly, `:time` overwritten
  with `stop` via `merge`, which preserves the column position
- `src/sortcycles.jl` — `sortcycles`, the within-timestamp stable sort. A
  `chunkmap` whose only state is the open cycle, held as a `Vector{DataFrame}`
  of pieces and concatenated once when a later time (or flush) closes it, so a
  cycle spread over many chunks stays O(rows). Each chunk emits the cycle it
  closes plus its own complete cycles and holds back the trailing one (a chunk
  that is all one cycle is held uncopied). The emitted rows are sorted *views*
  materialized in one copy — never concatenate the tail onto the whole chunk,
  which copied every chunk twice. `cycleperm!`
  is the typed barrier over a tuple of key views and a concrete `Ordering`
  (resolved from `rev` at construction, not per comparison): per cycle an
  `issorted` pass over the index range, then a stable `sort!` of that stretch of
  a permutation allocated only once some cycle is out of order
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
- `src/segtree.jl` — the monoid segment tree behind the rolling and window
  tree tiers: implicit array tree of `combine!`d partial state tuples,
  append-only rows, logical front expiry (`head`), amortized rebuilds,
  order-preserving two-accumulator range queries (`treequery`,
  `windowstart`). Appending (`treeappend!`, a bare leaf) and recombining
  (`treesync!`, the ancestors of every leaf since the last sync) are separate,
  so a per-tick caller pays about one combine per row; `treepush!` is the two
  together, for rolling's per-row queries. Only nodes over appended leaves are
  maintained: the sync combines the right edge with the tree's `ident`, so
  nothing past the end ever needs zeroing. The tree owns its two query
  accumulators and its node vector: `treequery` returns **borrowed** scratch,
  valid only until that tree's next query, and `rebuild!` at an unchanged
  capacity just swaps the live leaves' tuples to the front by reference — the
  steady state under a short window, where rebuilds fire every few appends
  rather than amortizing away
- `src/tiers.jl` — the per-accumulator window tiers shared by
  `addrollingcolumns` and `summarizewindows`. `tiering` partitions the realized
  states once per run and per widening: running (a group whose `freshwindowed`
  state is `isinvertible`), tree (another monoid), refold (anything else), and
  no tier for a singleton-typed (fieldless, dependent) state — unless every
  state is one, when they keep their structural tier so something tracks
  membership. `mergestates` splices the tiers' tuples back into topological
  order by a generated, compile-time permutation, so emission is the ordinary
  `summaryvalues` and dependents read across tiers. An absent tier is
  `nothing`/`()`, so a single-tier call compiles to the old single-mode code.
  `RunningTable` pools retired groups (windowed trackers own vectors). Tiers
  only demote, so a rebuild's buffer was already there; `replayrunning!`,
  `replaytrees!`, `replayoldtrees!` rebuild from the live rows
- `src/rolling.jl` — `addrollingcolumns`: one kernel, `rollsegment!`, over a
  `RollTiers` (shared row buffer with per-window eviction heads, per-window
  running tables, per-key trees owning their rows, refold templates). Per row:
  admit into every tier, advance each window's head (downdating running
  groups), then emit each window from the three tiers' states for the key — a
  `nothing` from any tier is the empty window. Re-fold folds from the window's
  own head, so it needs no time test. Widening rebuilds every tier from the
  live rows, re-partitioning (a non-invertible widening demotes only that
  accumulator); float sums stay running because the compensated states evict
  NaN/±Inf rows cleanly
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
  a concrete tick vector per chunk) over `addrollingcolumns`' row buffer, eviction
  head and tiers. One admission kernel (`windowrows!`) and one tick close
  (`closewindow!`): evict (downdating the running table), move tree heads and
  drop emptied trees, sync the rest once per tick — windows are queried per
  tick, not per row, so rolling's eager per-append ancestor update (log₂ of the
  window in combines per row) would squander that — fold the refold
  `GroupTable` from the live rows, emit, retire. Presence means rows in the
  window in every tier, so the first tier present (running, tree, refold) gives
  the keys to emit and the others are looked up by key (`tiervalues`). Keyed
  output is sparse plus one *vanish* row of empty values when a key's window
  empties, decided against the previous tick's *emitted* keys; that row is what
  stops a per-key as-of consumer (`applymodels`) from using a stale summary.
  A declared `keyset` rides in `WindowConfig`'s type (`Nothing` otherwise) and
  replaces the sort and vanish merge with `emitdense!`, a lookup per declared key
  per tick. The declaration is checked where the primary tier makes a group or
  tree (a key's first row); a refold primary groups only at ticks, so it checks
  on admission instead
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
