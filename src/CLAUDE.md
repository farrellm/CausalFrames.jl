# CausalFrames.jl — module map

DESIGN.md's "Module layout" table is the canonical index; this file records the
design rationale and performance constraints behind each module.

- `src/context.jl` / `src/chunks.jl` — `Context{T}`, the `[start, stop)`
  evaluation window, with `widenstart`, which every operator reading before
  `start` (a tolerance, a look-back) uses to widen and validate it; and the
  internal chunk protocol (`ChunkSource`, `chunkmap`): a single-pass lazy
  iterator of non-empty DataFrame chunks whose consumers own what they're
  given. Empty chunks are filtered out here, so downstream code may assume a
  chunk has rows. `PullCursor` is the one way to drive an iterator by hand (a
  producer draining its input, a binary transform pulling its second stream):
  sticky `pull!`, state in fields, touched per chunk only. Not
  `Iterators.Stateful`, which on Julia 1.10 prefetches and would break
  `head`'s early exit
- `src/frame.jl` — `CausalFrame{T}`: opaque, backed by a vector of
  time-ordered DataFrame chunks. The public inner constructor checks the
  invariants; `load`/`stream` build through a `Trusted`-token constructor that
  skips the O(n) scans, keeping `checkchunk`'s O(1) boundary and bounds
  guards. Column-access Tables.jl interface with a cheap `Tables.schema`;
  `DataFrame(frame)` is the copy point
- `src/pipeline.jl` — `CausalPipeline{F}` (lazy `Context -> iterator of
  DataFrame chunks`; the run function's type is a parameter, not an abstract
  `Function` field), `load` (drains into one frame without copying, the only
  operation that materializes the whole window), `stream` (one frame per
  chunk) and `scan` (drains for side effects such as `writecsv`, with `load`'s
  guards but no frame)
- `src/operators.jl` — sources return a `CausalPipeline`; transforms are
  curried (`filterrows(pred)` returns `CausalPipeline -> CausalPipeline`) so
  both chain with `|>`. Row functions run over concretely typed column-table
  rows behind a per-chunk function barrier, never `DataFrameRow`s.
  - shared with the parquet and JLS operators: `SourceClip` (a file source's
    clip options, window, carried `prevtime` and `done` flag), `clipchunk!`
    (rename, resolve `:time`, missing times via `table.jl`'s `presentrows`,
    sortedness, clip, convert) and the `ChunkSink` background writer, which
    `sinktransform` wires into a pass-through transform for all three writers.
    `sinkchunk` queues the chunk for the writer and passes downstream
    `DataFrame(c; copycols = false)`, a private index over the same vectors:
    consumers may mutate a chunk's column index, but no operator mutates a
    column vector in place, so sharing the vectors is sound. Other entries cite
    this as the `writecsv` hand-off argument
  - `CSVProducer` is the stateful-source shape (pull state in mutable fields
    behind a `ChunkSource`), shared by `ConcatProducer`, `HeadProducer`,
    `TableProducer` and `JLSProducer`. `concatenate` is sequential and checks
    that its pipelines arrive in time order; `concatenate()` is `emptyframe()`.
    `clock` is the timestamp-only tick source that `intervalize` and
    `summarizewindows` sample on
  - `readcsv` never infers types: every column is `String` unless `types`
    gives it a concrete one (`notes/readcsv-stringtype.md` records why CSV.jl's
    own `stringtype` default stays out)
  - the three column operators share one selector vocabulary and one per-run,
    schema-keyed resolution memo (`SchemaMemo`). Only `reordercolumns` reads
    the selectors as an *order*, which is why `foreachselector` exists beside
    `foreachliteral`: matching can stop at the first hit and ignore pattern
    leaves, ordering can do neither
  - `lag` shifts times via the shared `shiftchunk!` and widens the context in
    `lagcontext`, mirroring `Acausal.lead`. `settime` is the per-row form of
    that shift, sharing `settimechunk!` with `Acausal.settime`; it cannot widen
    the context (the shift is data-dependent), so it needs three independent
    order checks instead (see DESIGN.md's "Retiming")
  - `head` is the one transform not built on `chunkmap`, which cannot stop
    early: it drives the upstream iterator from a `HeadProducer` behind a
    `ChunkSource`. A second early-exit operator would be the point to extract
    a `chunks.jl` primitive
- `src/merge.jl` — `Base.merge(ps::CausalPipeline...)`, the n-ary
  time-interleaving source (extending Base rather than shadowing it; no
  zero-argument form, which would capture `merge()`). One `MergeCursor` per
  pipeline buffers a chunk; `pickwinner` chooses by `(time, argument index)`;
  pieces are claimed as `searchsorted*` runs with no data moved until
  `batchsize` rows are pending, then copied with one allocation per output
  column. A lone whole-chunk piece is adopted, or passed through untouched
  when it already has the union schema
- `src/parquet.jl` — `readparquet`/`writeparquet`: the API, docstrings and
  backend selection (`resolvebackend`, `parquetproducer`, `parquetsink`,
  `backendloaded`), naming no backend. DuckDB and Parquet2 are **weak** deps
  behind package extensions, and **either one serves both directions**:
  reading prefers DuckDB (a filtered streaming query) and falls back to
  Parquet2 (row groups skipped by their statistics); writing prefers Parquet2
  (row groups written as the stream flows) and falls back to DuckDB (staged in
  a temp table, one `COPY` at the end). Skipping is always an optimization:
  chunks are clipped again on arrival. `backend = :duckdb`/`:parquet2` forces
  the choice, which is how the tests cover all four combinations in one
  process; working on parquet means loading `DuckDB`/`Parquet2` in the session
  first. `sort = true` (here and on `readcsv`) cannot stream, so it reads via
  `gatherchunk!`/`sortgathered` in `operators.jl` (scan every chunk for
  in-window rows, stable-sort once, emit one chunk), except that DuckDB sorts
  in SQL, tie-broken by its virtual `file_row_number` since its `ORDER BY` is
  unstable (see DESIGN.md's "Sorting a file source")
- `src/jls.jl` — `writejls`/`readjls`, the untyped persistence pair: a header
  record then one `serialize`d DataFrame per chunk, each its own `serialize`
  call so records are independent. The sink is `ChunkSink` with a serializing
  write loop; the source is `JLSProducer` over `clipchunk!`. It exists for
  columns CSV and parquet cannot encode (fitted models), so the file is tied
  to Julia and package versions by design; it is not an interchange format
- `src/table.jl` — `readtable`, the in-memory source, on three paths sharing
  the `windowbounds` clip. A generic Tables.jl table is resolved per run and
  per partition by `TableProducer`, with `tablerows` the typed barrier and only
  in-window rows copied. An `AbstractDataFrame` is resolved once at
  construction into a private index over the caller's vectors, then read by
  the frame path with `share = false` (every run slices, so a loaded frame
  never aliases the caller) unless dropping missing times or sorting made owned
  copies. A `CausalFrame` is read by `framepipeline`: a binary search over
  chunk bounds, with whole-window chunks passed on as `copycols = false`
  indexes over the frame's vectors (sound by the `writecsv` hand-off argument).
  The frame-context check and the automatic `closed` rule live there too; see
  DESIGN.md's "Tables as sources"
- `src/summarizers.jl` — `Summarizer` (immutable config, output column name in
  a type parameter) and `SummarizerState` (running state, typed from the input
  schema), plus their unexported interface: `emptyvalue`, `fresh`, `fresh!`,
  `freshwindowed`, `update!`, `value`, `widenstate`, `dependencies`,
  `combine!`, `downdate!`, `isinvertible`. Within that:
  - `fresh!` zeroes a state in place and returns it (default `fresh(st)`, so
    it is opt-in for a custom summarizer). The transforms zero a state tuple
    per cycle, per interval and per window query; see DESIGN.md's "Reusing
    state". Callers must use the return value
  - the structured subtypes are `MonoidSummarizer` (states combine
    associatively over stream-ordered ranges via `combine!`, a fresh state
    being the identity) and `GroupSummarizer <: MonoidSummarizer` (the windowed
    state removes rows via `downdate!`). `tiers.jl` picks each accumulator's
    window tier from this, so which one a new summarizer claims is a
    performance decision, not a taxonomy one
  - `downdate!` is only ever handed the **oldest** row still folded: every
    caller evicts FIFO per group, and any new caller must too. States rely on
    it: `AgeWeightedSum` takes back the evicted row's weight `n - 1`, the
    windowed deques pop their front by sequence number, and windowed `Last`
    only counts
  - `freshwindowed` (default `fresh`) is the state only the running tier
    slides, so a group whose ordinary state cannot invert keeps a fixed-size
    state everywhere else. `Min`/`Max`/`First`/`Last` share one ordinary state
    (`TrackState`, parameterized by the combiner) and one windowed state
    (`WindowTrackState`, a monotonic deque under the same combiner; First keeps
    the whole window), except Last, whose windowed state is a count plus the
    newest value (`WindowLastState`), measured 3-4x cheaper per row than the
    deque. `CountDistinct` folds a `Set` but slides a `Dict{T,Int}` of counts,
    because a count increment hashes twice and is 1.4-1.7x slower than `push!`
    on the non-window folds (DESIGN.md has the table). A windowed state is
    never combined or widened. `Product` is the only built-in monoid that is
    not a group
  - the dependent summarizers (`Moment`, `Mean`, `Variance`, `Std`,
    `Covariance`, `Correlation`, `LinearRegression`) have empty state structs:
    they declare `dependencies` and read those values back through the
    two-argument `value(st, vals)`. Such a state joins the `DerivedState`
    union, which gives it no-op `fresh`, `update!`, `combine!` and
    `downdate!`. States are shared by shape, not summarizer (`Mean` and
    `Moment` are one `CountRatioState`, and `Variance` is `Covariance`'s state
    over a column and itself)
  - a summarizer whose value is symmetric in two columns (`DotProduct`,
    `Covariance`, every pairwise term in `LinearRegression`) folds under the
    `isless`-sorted argument order via `canonicaldot`/`canonicaldotname`, but
    emits the column name the caller asked for. For an accumulator, the
    reversed form becomes a dependent over the canonical one through the
    fieldless `AliasState{N,D}`; something already dependent just names the
    canonical form. DESIGN.md's "Symmetric summarizers" is the rule to follow
    when adding another
  - `LinearRegression` is the outlier among them: the only one emitting
    several columns (a coefficient and t statistic per term, plus
    `r2`/`stderr`/`n`, so its output names are a tuple parameter), and the
    only one that allocates (`K >= 2` builds a `K x K` workspace per emitted
    row for the Cholesky; `K = 1` takes a closed form over scalars). It reads
    its dependencies through `NamedTuple{names}(vals)` projections, so no name
    is a runtime value on the emission path
  - `AgeWeightedSum` (`Σ k·y`, k the row's age) can't be a term functor, since
    its update reads its own `S₁`, so it has its own plain and compensated
    states, reusing the `Compensated` helpers and counting `missing` through
    the flag `M`. `S₂`'s nonfinite classification is `S₁`'s minus the newest
    row's (`newest`), whose weight is exactly 0
  - the sum family `Sum`/`SumPower`/`DotProduct` shares one state,
    `AccumState{N,A,T,M,S}`, over a term functor `T` (`ColumnTerm`/`PowerTerm`/
    `PairProductTerm`, terms formed at accumulator width). Its storage `S` is
    the plain `A` or a `Compensated{A}`, and every fold operation (`accadd`,
    `accsub`, `accmerge`, `accvalue`) dispatches on it. `Compensated` runs
    Neumaier summation over the finite terms and counts NaN/±Inf separately,
    reconstructing the IEEE result in `value`, which lets a rolling window
    evict a nonfinite row cleanly and stay on the running path. `BigFloat` is
    excluded: compensation buys nothing at arbitrary precision, and a
    non-isbits `Compensated` would allocate per row
  - a `Missing`-admitting accumulator type sets the flag `M` and folds at the
    non-missing type, counting `missing` terms as the compensated storage
    counts nonfinites, so the accumulator stays invertible. `Union{Missing,_}`
    appears only in `value`'s return, never in an accumulation field. With `M`
    false the count is never touched and its tests compile away
  - the order statistics (`Quantile`, `Median`, `PercentRank`) are fieldless
    dependents over one accumulator, the unexported `SortedValues`: a sorted
    `Vector` (binary search plus memmove: under 300 ns per row to a 1,000-row
    window, growing linearly past ~10,000; DESIGN.md has the table) with
    `missing` and NaN counted, not stored, so it stays a group and every window
    slides it. Its value is the state itself, *borrowed* like `treequery`'s
    scratch: dependents read it within the emission and the projection drops
    it, so emission allocates nothing. `combine!` merges into `scratch` and
    swaps, which tolerates `dest` aliasing an input. `linearquantile` copies
    Statistics' type-7 arithmetic because `quantile(v; sorted = true)` scans
    all of `v` per call; `nearestrank` corrects `ceil(p * n)` to TA-Lib's exact
    rank. `PercentRank` reads the newest value from `Last`, a group via
    `WindowLastState`, so it keeps the running tier too
  - `FitModel` (MLJ) is the one summarizer whose value is an object: its state
    buffers the folded rows in concretely typed vectors and fits at `value`
    time through the `fitmodel` hook (src/models.jl), building
    `NamedTuple{(N,),Tuple{FittedModel{P,M}}}` from type parameters so the
    column is concrete though the fit is opaque. It is a plain `Summarizer`
    (refolded everywhere). `fresh!` empties the buffers keeping capacity,
    which is safe only because `value` hands the model copies
- `src/summarize.jl` — the folding kernels and the transforms `summarize`,
  `summarizecycles` and `addsummarycolumns`; also the key validators every
  keyed transform shares, `keycolumns` (eager: unique, never `:time`) and
  `checkkeycolumns` (first chunk: present in the input, naming which input for
  the binary transforms). `prototypes` expands dependencies topologically and
  returns the requested output names, which ride through the kernels in a
  `Val` to project hidden dependencies out of the output. Per-run state lives
  in `SummaryFold`, never in reassigned closure captures (those get boxed).
  The per-key states live in a `GroupTable{K,S}` (both parameters concrete),
  which also holds the reused emission buffer and the pool of retired state
  tuples that `closecycle!` retires into and `groupstates!` zeroes back out
  of: the keyed cycle fold closes a table per timestamp, and rebuilding it per
  cycle would be the package's heaviest allocation site. A declared `keyset`
  (a `KeySet`: the declared keys plus a `Dict{K,Int}` slot index, built at
  construction so the output key type is the declaration's) swaps the table
  for `DenseGroups`, a state tuple per slot plus a `folded` flag. Dense output
  closes *every* key every time, so there is nothing for the table's Dict,
  sort and pool to save; `closedense!` just walks the slots. The slot lookup
  doubles as the undeclared-key check, so it costs the dense fold nothing over
  the sparse one. Data keys are looked up without conversion, so the declared
  key type need only be `isequal` to the data's
- `src/join.jl` — `asofjoin`, the binary as-of join: a chunkmap over the left
  stream pulls right chunks on demand (a two-pointer merge, per left row) into
  a concretely typed per-key store; `tolerance` widens the right context back
  by the tolerance (`widenstart`). It is also the join engine `applymodels`
  and `Acausal.futurejoin` run on: `JoinConfig` (with `op` for messages and a
  direction singleton, `Backward` here), `JoinState`, `pullright!`,
  `matchchunk!` and `joinchunk!`, with the store and kernel picked by dispatch
  (`newstore`/`widenstore`/`segment!`). A new join direction adds those three
  methods, never a second driver. The store is a `SlotStore` (a `Dict{K,Int}`
  of slot numbers over a `Vector{V}` of rows, shared with `lastrow` through
  `admitslot!`), and a match is a `Vector{V}` plus a `Vector{Bool}` mask
  (unmatched slots left undefined, hence `convertmatches` for a mid-chunk
  widening). Both keep `V` out of a `Union`, which Julia stores inline only if
  every member is isbits, so a `String` or `Missing`-admitting right column
  would otherwise cost a box per left row. See DESIGN.md's "Representing a
  match"; the zero-allocation kernel tests guard it
- `src/lookupjoin.jl` — `lookupjoin`, the key-only join against a timeless
  in-memory table. Not a stream: the table is copied and indexed once, at
  construction, into a `LookupJoin` (a `Dict{K,Int}` of table rows plus the
  value columns as a concrete NamedTuple, captured by the closures so each
  chunk's call dispatches statically), and the transform is a stateless
  `chunkmap`, so unlike the as-of joins it keeps chunk concatenation over split
  contexts. Per chunk, `lookuprows!` fills a `Vector{Int}` of rows (0 =
  unmatched; `Int`s so a `String` key costs nothing per row, which the
  zero-allocation test pins), then each value column is gathered. `unmatched`
  is a singleton mode type picked at construction: `:missing` gathers
  `Union{Missing,T}` (`gathermissing`), `:error`/`:drop` keep `T`, so the
  schema follows the keyword, never the data. Shares `prefixleft!`'s field
  form with `asofjoin`
- `src/lastrow.jl` — `lastrow`, the last-row-per-key transform: `summarize`'s
  fold-and-flush-at-`stop` shape over `join.jl`'s `SlotStore` rather than a
  `GroupTable`, since it keeps whole rows and a `Dict{K,V}` would box one per
  row for any non-isbits `V`. No `found` mask (every slot is written the
  instant it is claimed, so widening is a plain `convert`), and the keyless
  path skips the store: only the chunk's last row matters, so it costs nothing
  per row. The output schema is the input's exactly, `:time` overwritten with
  `stop` via `merge`, which keeps the column position
- `src/sortcycles.jl` — `sortcycles`, the within-timestamp stable sort. A
  `chunkmap` whose only state is the open cycle, held as a `Vector{DataFrame}`
  of pieces and concatenated once when a later time (or flush) closes it, so a
  cycle spread over many chunks stays O(rows). Each chunk emits the cycle it
  closes plus its own complete cycles and holds back the trailing one (a chunk
  that is all one cycle is held uncopied). The emitted rows are sorted *views*
  materialized in one copy; concatenating the tail onto the whole chunk would
  copy every chunk twice. `cycleperm!` is the typed barrier over a tuple of key
  views and a concrete `Ordering` (resolved from `rev` at construction): per
  cycle an `issorted` pass over the index range, then a stable `sort!` of that
  stretch of a permutation, allocated only once some cycle is out of order
- `src/fill.jl` — the two missing-value fills. `fillmissing` is row-wise and
  stateless (a constant per column, so `fillcolumn` is the whole hot path, and
  the output eltype narrows, the one place in the package that does, via
  `promote_type(nonmissingtype(T), typeof(value))`). `forwardfill` carries the
  last non-missing value across rows, chunks and keys, so it is one of the
  stateful operators DESIGN.md's streaming section lists. Its state is a
  `FillCell` per (key, *column*), not a row per key: columns carry
  independently, so `lastrow`'s whole-row store cannot serve. The cells are
  mutable, so the store is a plain `Dict{K,NamedTuple}` rather than a
  `SlotStore`: a lookup already answers with a pointer, with no
  `Union{Nothing,V}` to box (as with `KeyBuffer`). A selected column whose
  promoted type admits no `Missing` gets `nothing` instead of a replacement
  column, which folds the write out of the unrolled kernel. `tolerance` widens
  the input context as `asofjoin` does, so `clipstart!` drops the pre-`start`
  rows the fill was allowed to see
- `src/segtree.jl` — the monoid segment tree behind the rolling and window
  tree tiers: an implicit array tree of `combine!`d partial state tuples,
  append-only rows, logical front expiry (`head`), amortized rebuilds, and
  order-preserving two-accumulator range queries (`treequery`,
  `windowstart`). Appending (`treeappend!`, a bare leaf) and recombining
  (`treesync!`, the ancestors of every leaf since the last sync) are separate,
  so a per-tick caller pays about one combine per row; `treepush!` does both,
  for rolling's per-row queries. Only nodes over appended leaves are
  maintained: the sync combines the right edge with the tree's `ident`, so
  nothing past the end needs zeroing. The tree owns its two query
  accumulators and its node vector: `treequery` returns **borrowed** scratch,
  valid only until that tree's next query, and `rebuild!` at an unchanged
  capacity swaps the live leaves' tuples to the front by reference. That is
  the steady state under a short window, where rebuilds fire every few
  appends rather than amortizing away
- `src/tiers.jl` — the per-accumulator window tiers shared by
  `addrollingcolumns` and `summarizewindows`. `tiering` partitions the realized
  states once per run and per widening: running (a group whose `freshwindowed`
  state is `isinvertible`), tree (another monoid), refold (anything else), and
  no tier for a singleton-typed (fieldless, dependent) state, unless every
  state is one, when they keep their structural tier so something tracks
  membership. `mergestates` splices the tiers' tuples back into topological
  order by a generated, compile-time permutation, so emission is the ordinary
  `summaryvalues` and dependents read across tiers. An absent tier is
  `nothing`/`()`, so a single-tier call compiles to a single-algorithm kernel.
  `RunningTable` pools retired groups (windowed trackers own vectors).
  `rebuildbuffer` and `rebuildtrees` choose a rebuild's buffer and trees for
  both transforms, and `replayrunning!`, `replaytrees!` and `replayoldtrees!`
  rebuild from the live rows. Built-in states only demote, so the buffer a
  rebuild needs already exists; `treebuffer` gathers one from the trees for a
  custom state that promotes
- `src/rolling.jl` — `addrollingcolumns`: one kernel, `rollsegment!`, over a
  `RollTiers` (a shared row buffer with per-window eviction heads, per-window
  running tables, per-key trees owning their rows, refold templates). Per row:
  admit into every tier, advance each window's head (downdating running
  groups), then emit each window from the tiers' states for the key; a
  `nothing` from any tier is the empty window. The refold tier folds from the
  window's own head, so it needs no time test. Widening rebuilds every tier
  from the live rows, re-partitioning (a non-invertible widening demotes only
  that accumulator); float sums stay running because the compensated states
  evict NaN/±Inf rows cleanly
- `src/intervalize.jl` — `intervalize`, the third binary transform: summarize
  over the intervals a `clock` pipeline defines (`[bₖ, bₖ₊₁)`, timestamped at
  `bₖ₊₁`). It is the `summarizecycles` fold, closing on clock boundaries.
  Keyless output is a regular grid (empty intervals get `emptyvalue`s, output
  types promoted via the shared `promotedvaluetype`); keyed output is sparse
  (present keys only, via `closecycle!`). The `chunkmap` over the data stream
  pulls clock boundaries into a concrete `Vector{T}` per chunk (the pull is
  the only dynamism; the per-row kernel stays dispatch-free), reusing
  `SummaryFold` whole; `closelast` closes the trailing partial at `stop`
- `src/windows.jl` — `summarizewindows`, the clock-sampled trailing window
  (`[τ - lookback, τ)` at each tick): `intervalize`'s driver (`IntervalCursor`
  and `pullpast!`, a concrete tick vector per chunk) over `addrollingcolumns`'
  row buffer, eviction head and tiers. One admission kernel (`windowrows!`)
  and one tick close (`closewindow!`): evict (downdating the running table),
  move tree heads and drop emptied trees, sync the rest once per tick (windows
  are queried per tick, not per row, so rolling's per-append ancestor update
  would waste log₂(window) combines per row), fold the refold `GroupTable`
  from the live rows, emit, retire. A key is present in a tier exactly when
  its window has rows, so the first tier present (running, tree, refold) gives
  the keys to emit and the others are looked up by key (`tiervalues`). Keyed
  output is sparse, plus one *vanish* row of empty values when a key's window
  empties, decided against the previous tick's *emitted* keys; that row stops
  a per-key as-of consumer (`applymodels`) from using a stale summary. A
  declared `keyset` rides in `WindowConfig`'s type (`Nothing` otherwise) and
  replaces the sort and vanish merge with `emitdense!`, a lookup per declared
  key per tick. The declaration is checked where the primary tier makes a
  group or tree (a key's first row); a refold primary groups only at ticks, so
  it checks on admission instead
- `src/models.jl` — the MLJ operators (`applymodels`, `addpredictions`,
  `modelreports`) and the five hooks the extension implements (`ismodel`,
  `fitmodel`, `predictmodel`, `savefitresult`, `restorefitresult`). It names no
  MLJ type, as `parquet.jl` names no backend.
  - `applymodels` is `asofjoin`'s join engine unchanged, driven through
    `matchchunk!` (`JoinConfig.op` names it in errors), plus a predict step:
    matched rows are grouped by model identity behind a function barrier, each
    distinct model gets one `predictmodel` call per chunk over views, and the
    results are scattered into one promoted `Union{Missing,E}` column.
  - `addpredictions` is pure composition: `summarizewindows` into
    `applymodels(strict = false)`, sound because the windows are half-open.
  - `FittedModel`'s custom serializer routes fitresults through MLJ's
    save/restore.
- `src/acausal.jl` — the `Acausal` submodule (`futurejoin`, `lead`,
  `settime`), reached only through `using CausalFrames.Acausal` and never
  re-exported, so acausality is always an explicit opt-in. `settime` is not
  even in the submodule's `export` list: the top level exports that name too,
  and a name exported by two `using`d modules is an error to use unqualified,
  so exporting it would break the *causal* `settime` for anyone who also
  imported `Acausal`. Reach it as `CausalFrames.Acausal.settime`; a test pins
  its absence from `names(Acausal)`. `futurejoin` runs on `asofjoin`'s join
  engine, supplying only the `Forward` direction's `newstore`/`widenstore`/
  `segment!` methods (so the forward-looking code stays in the submodule),
  with the match direction, tie-break and context widening (`stop +
  tolerance`) inverted. Because it matches the *earliest* qualifying right
  row, it buffers right rows per key until a left row consumes or outruns
  them, and proving a key has no future match drains the right stream: worst
  case O(right rows), against `asofjoin`'s O(keys) store. It shares the
  `Vector{V}` + mask match buffer, and `KeyBuffer` is mutable, so a store
  lookup answers with a pointer and never boxes
- `src/precompile.jl` — the PrecompileTools workload over the main pipeline
  paths; every new operator adds a path here
