# CausalFrames.jl — module map

DESIGN.md's "Module layout" table is the canonical index; this file records the
design rationale and performance constraints behind each module.

- `src/context.jl` / `src/chunks.jl` — `Context{T}`, the `[start, stop)`
  evaluation window, with `widenstart`, which every operator reading before
  `start` (a tolerance, a look-back) uses to widen and validate it, its mirror
  `widenstop` for the acausal operators reading past `stop`, and the shared
  integer-argument checks `checkpositive`/`checknonnegative`; and the
  internal chunk protocol (`ChunkSource`, `chunkmap`): a single-pass lazy
  iterator of non-empty DataFrame chunks whose consumers own what they're
  given. Empty chunks are filtered out here, so downstream code may assume a
  chunk has rows. `PullCursor` is the one way to drive an iterator by hand (a
  producer draining its input, a binary transform pulling its second stream):
  sticky `pull!`, state in fields, touched per chunk only. Not
  `Iterators.Stateful`, which on Julia 1.10 prefetches and would break
  `head`'s early exit. `LeadInDrop`/`dropleadin!` drop the rows before a
  start, for `warmup`, `forwardfill`'s `tolerance` and a shared run's
  later-starting side
- `src/share.jl` — the `CausalPipeline{F}` node and the plan of one
  evaluation (issue #105). A node is immutable (a `mutable struct` of `const`
  fields, for cheap identity): its run function `(ctx, ins) -> iterator`, its
  inputs (an erased `Vector{CausalPipeline}`, so a pipeline's type never nests
  its inputs'), `contexts` (each input's context from its own) and the
  `startfree`/`stopfree` flags; build it with `pipesource`, `pipenode` or
  `pipetransform() do upstream, ctx … end`, each and-ing the operator's own
  flags with its inputs'. Operators reach inputs only through
  `openinput(ins, i)` (each at most once), never another node's `run`: that
  is what lets `planrun` see every request before anything runs, and it is
  the seam a parallel executor would use. `planrun` visits nodes in reverse postorder and groups each node's
  requests (`RunSlot`s) into `RunInstance`s by `sharekey` (time type, then
  start unless startfree, stop unless stopfree, and the isolating
  `sharing(false)` run if any), each over the union of its slots' contexts.
  A pipeline with no node reached twice skips the plan (`istree`,
  `DirectInputs`): it cost 4-6 µs per evaluation, +20-90% on the
  sub-millisecond benchmarks. A
  lone slot reads its run directly; several read a `ChunkTee`, which queues
  per side only the rows that side's `SideClip` keeps (`copycols = false`
  indexes, the `writecsv` hand-off argument), so a side starting later queues
  none of the lead-in. A planned run's chunks come wrapped in an
  `InstanceRun`, which abandons every input slot (`abandonslot!`, opened or
  not) once the run ends or is abandoned, so operators wire up no abandoning
  of their own: only `head`, stopping while it still runs, calls `abandon!`
  (through its `PullCursor`). A tee then stops queueing for that side, and
  abandons the run once no side is live; off the plan nothing is shared and
  `abandon!` is a no-op. The flags'
  per-operator table is in DESIGN.md's "Shared runs"; a wrong `true` is a
  silent wrong answer, so every flag is covered by `test/sharing.jl`'s
  property test. Everything that takes a pipeline once per build or per
  evaluation (the builders, `UnaryTransform`, the curried operators' closures,
  the planner, `load`/`scan`/`stream`) takes it `@nospecialize`: specialized
  on each `CausalPipeline{F}`, the planner alone compiled ~80 ms per node
  type, and `operators.jl` + `join.jl` went from 37 to 56 s; erased, they take
  35 s, since a run function no longer captures (and nests) its inputs' types
- `src/compaction.jl` — `compactdead!`, the one dead-prefix rule for every
  FIFO that pops by advancing a head (the window transforms' row buffer, the
  monoid queues' rows, the windowed extremes' deque, `WindowValues`,
  `futurejoin`'s `KeyBuffer`): drop the prefix when it is everything, or past
  `COMPACTMIN` (32) rows and half the vector. Below 8, a steady 1-row FIFO paid
  3-5x per row in compactions, and from 32 up the cost is flat; the buffers
  are mostly per key, each holding up to `COMPACTMIN` dead rows, so don't raise
  it per site. It drops with a prefix `deleteat!`, which only
  advances the vector's start and lets a later push reuse the freed front;
  moving the live rows down with `copyto!` measured 7-14% slower on
  `futurejoin`, for the same allocations, so don't "fix" it into an in-place
  move
- `src/frame.jl` — `CausalFrame{T}`: opaque, backed by a vector of
  time-ordered DataFrame chunks. The public inner constructor checks the
  invariants; `load`/`stream` build through a `Trusted`-token constructor that
  skips the O(n) scans, keeping `checkchunk`'s O(1) boundary and bounds
  guards. Column-access Tables.jl interface with a cheap `Tables.schema`;
  `DataFrame(frame)` is the copy point
- `src/pipeline.jl` — `CausalPipeline`'s public constructor (a source,
  dependent unless it declares `startfree`/`stopfree`), `load` (drains into one frame without copying, the only
  operation that materializes the whole window), `stream` (one frame per
  chunk) and `scan` (drains for side effects such as `writecsv`, with `load`'s
  guards but no frame)
- the basic operators, one file per job: `src/sources.jl` (`emptyframe`,
  `concatenate`, `clock`), `src/sourceclip.jl` (the file sources' shared
  clip), `src/sinks.jl` (the file sinks' shared writer), `src/csv.jl`,
  `src/rowwise.jl` (`filterrows`, `addcolumns`), `src/retime.jl` (`lag`,
  `warmup`, `head`, `settime`) and `src/columns.jl`. Sources return a
  `CausalPipeline`; transforms are curried (`filterrows(pred)` returns
  `CausalPipeline -> CausalPipeline`, built by `share.jl`'s
  `pipetransform() do upstream, ctx … end`) so both chain with `|>`. Row functions
  run over concretely typed column-table rows behind a per-chunk function
  barrier, never `DataFrameRow`s.
  - shared with the parquet and JLS operators: `SourceClip` (a file source's
    clip options, window, carried `prevtime` and `done` flag), `clipchunk!`
    (`resolvechunk!` — rename, keep `columns`, resolve `:time`, missing times
    via `table.jl`'s `presentrows`, shared with `gatherchunk!` — then
    sortedness, clip, convert). `readscolumn` is the one predicate every
    reader pushes `columns` down by (through `keptfilecolumns`, the one carrier
    rule; CSV.jl's `select` is that list over the header, read first), over
    `keepscolumn`, which `readtable` shares, and `keepcolumns!` holds each
    chunk to it. Also the `ChunkSink`
    background writer, which
    `sinktransform` wires into a pass-through transform for all three writers.
    `sinkchunk` queues the chunk for the writer and passes downstream
    `DataFrame(c; copycols = false)`, a private index over the same vectors:
    consumers may mutate a chunk's column index, but no operator mutates a
    column vector in place, so sharing the vectors is sound. Other entries cite
    this as the `writecsv` hand-off argument
  - `CSVProducer` is the stateful-source shape (pull state in mutable fields
    behind a `ChunkSource`), shared by `ConcatProducer`, `HeadProducer`,
    `MergeProducer`, `TableProducer`, `JLSProducer` and the extensions'
    `QueryProducer`/`RowGroupProducer`. The three file producers pull through
    `sourceclip.jl`'s `pullclipped!`, which streams (`nextclipped!`) or, under
    `sort = true`, gathers (`gathersorted!`). `concatenate` is sequential and checks
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
  - `warmup` runs `f(p)` (built once, at application) over `widenstart`'s
    context and drops the lead-in with `dropleadin!`, which `forwardfill`'s
    `tolerance` shares. Its `LeadInDrop` is created per run, never per
    construction, and passes chunks untouched once one reaches `start`; the
    straddling chunk is sliced, never `deleteat!`ed (the `writecsv` hand-off
    argument)
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
  `gatherchunk!`/`sortgathered` in `sourceclip.jl` (scan every chunk for
  in-window rows, stable-sort once, emit one chunk), except that DuckDB sorts
  in SQL, tie-broken by its virtual `file_row_number` since its `ORDER BY` is
  unstable, and in time slices of about `sortrows` rows, since one sort over a
  key-ordered file's window holds the whole file (see DESIGN.md's "Sorting a
  file source")
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
- `src/summarizers.jl` (the include list for `src/summarizers/`, one file per
  family, the interface in `interface.jl`) — `Summarizer` (immutable config, output column name in
  a type parameter) and `SummarizerState` (running state, typed from the input
  schema), plus their unexported interface: `emptyvalue`, `fresh`, `fresh!`,
  `freshwindowed`, `update!`, `value`, `widenstate`, `dependencies`,
  `inputcolumns`, `combine!`, `downdate!`, `isinvertible`. Within that:
  - `inputcolumns` (the columns a state reads, `nothing` for any) is declared
    for every built-in in one file, `summarizers/inputcolumns.jl`, a dependent one
    as `()`; a test checks that no built-in falls back to `nothing`, which
    would silently keep every column in the window transforms' stored rows
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
    deque. The deque extremes run alone 0–30% faster on the monoid queue, but
    stay groups: a mixed call (OHLC) would then compile two tiers, +9–17% for
    little runtime (`notes/extremes-monoid-tier.md`). `CountDistinct` folds a
    `Set` but slides a `Dict{T,Int}` of counts,
    because a count increment hashes twice and is 1.4-1.7x slower than `push!`
    on the non-window folds (DESIGN.md has the table). A windowed state is
    never combined or widened. `MinIndex`/`MaxIndex` and
    `MinWithIndex`/`MaxWithIndex` fold `IndexState` (the extreme, its
    rows-since and a row count, for `combine!`) and slide the same deque, a
    type parameter `K` (`:value`, `:index`, `:both`) choosing what `value`
    reports, so wanting both costs one deque, not two. `Product` is the only
    built-in monoid that is not a group
  - the dependent summarizers (`Moment`, `Mean`, `Variance`, `Std`,
    `Covariance`, `Correlation`, `LinearRegression`, `Quantile`, `Median`,
    `PercentRank`, `MeanAbsDev`, and a reversed `DotProduct`'s `AliasState`)
    have empty state structs: they declare `dependencies` and read those
    values back through the two-argument `value(st, vals)`, naming each
    dependency by `outname`. Such a state subtypes the abstract
    `DerivedState`, which gives it no-op `fresh`, `update!`, `combine!` and
    `downdate!`. States are shared by shape, not summarizer (`Mean` and
    `Moment` are one `CountRatioState`, and `Variance` is `Covariance`'s state
    over a column and itself)
  - a summarizer whose value is symmetric in two columns (`DotProduct`,
    `Covariance`, every pairwise term in `LinearRegression`) folds under the
    `isless`-sorted argument order via `canonicaldot`/`canonicalcomoment`, but
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
  - `CoMoment{a,b}` (unexported, no docstring, canonical `isless` order only)
    is the centred co-moment `Σ(a−ā)(b−b̄)` that `Variance`, `Covariance` and
    an intercept `LinearRegression` read instead of the raw sums, which cancel
    catastrophically under a large level (issue #89; compensating the raw sums
    can't help, since each `x*x` term and the read-back `Σx²` are already
    rounded at the scale of `x²`). Its state folds compensated sums of `x − K`
    over rows with both values present and finite, counting `missing` (flag
    `M`) and nonfinite rows apart so it stays a group. It recentres `K` to the
    newest row once more than half its rows postdate the shift — the newest
    row, not the mean, so deviations stay exact input differences and the
    running tier agrees exactly with a re-fold. Every product entering the sums
    is error-free (`twoprod`, its error added to the compensation) and `value`
    reads the compensated pairs as double-doubles (`comomentvalue`), since the
    sums sit at the scale of the squared distance to the shift and the
    co-moment at the spread's (issue #91). `combine!` re-expresses one side at
    the other's shift. Sliding still kept a residue at the scale of a dominant
    row once it had left (issue #97), so the windowed float state
    (`freshwindowed`, flag `W`; tracking on the ordinary state cost
    `summarize` 8%) tracks its largest product (`peak`) and, past `REFOLDRATIO` times the window's scale (`Σd²`,
    or `√(qa·qb)` from plain squares kept only for a covariance), asks to be
    refolded (`needsrefold`, internal, `false` for every other state; the
    generated `anyrefold` reads a group's asks as straight-line `||`, so a
    group that can't ask compiles the check away). It costs 20-26% on regressions and
    `Correlation` over the old raw sums, and the error-free products up to
    ~13% more; `notes/variance-cancellation.md` has
    the numbers and the rejected designs (Welford, error-free squares on the
    raw sums, a per-fit multivariate accumulator)
  - `AgeWeightedSum` (`Σ k·y`, k the row's age) can't be a term functor, since
    its update reads its own `S₁`, so it has its own plain and compensated
    states, reusing the `Compensated` helpers and counting `missing` through
    the flag `M`. `S₂`'s nonfinite classification is `S₁`'s minus the newest
    row's (`newest`), whose weight is exactly 0. The weight products in
    `downdate!` and `combine!` are error-free (`twoprod`), as the co-moments'
    are: a rounded `(n - 1)·y` drifted linearly on a flat series (issue #94).
    Its windowed compensated state (`freshwindowed`, flag `W`) tracks a peak
    `|y|` and plain live `Σ|y|` and asks for a refold as the sums do (issue
    #97): `S₂` re-adds `S₁`'s lost compensation every row, so a spike left a
    76% error
  - the sum family `Sum`/`SumPower`/`DotProduct` shares one state,
    `AccumState{N,A,T,M,S}`, over a term functor `T` (`ColumnTerm`/`PowerTerm`/
    `PairProductTerm`, terms formed at accumulator width). Its storage `S` is
    the plain `A` or a `Compensated{A}`, and every fold operation (`accadd`,
    `accsub`, `accmerge`, `accvalue`) dispatches on it. `Compensated` runs
    Neumaier summation over the finite terms and counts NaN/±Inf separately,
    reconstructing the IEEE result in `value`, which lets a rolling window
    evict a nonfinite row cleanly and stay on the running path. `BigFloat` is
    excluded: compensation buys nothing at arbitrary precision, and a
    non-isbits `Compensated` would allocate per row. The compensation is a
    plain float and loses what lies far below a dominant term's round-off
    (issue #97, pandas GH#52407), so the windowed state (`freshwindowed`,
    storage `Tracked` from `trackedstorage`) also keeps its largest finite
    `|term|` (`peak`) and a plain live `Σ|term|` (`mag`) and asks for a
    refold, as the co-moment does, once `peak > REFOLDRATIO · mag`
    (`REFOLDRATIO` lives in `compensated.jl`, shared by both). Only the
    windowed state tracks: on every path it cost `summarize` up to 24%
  - a `Missing`-admitting accumulator type sets the flag `M` and folds at the
    non-missing type, counting `missing` terms as the compensated storage
    counts nonfinites, so the accumulator stays invertible. `Union{Missing,_}`
    appears only in `value`'s return, never in an accumulation field, except
    `Product`'s, a monoid that never downdates, whose total absorbs a
    `missing`. With `M`
    false the count is never touched and its tests compile away. An
    all-`Missing` column (accumulator type `Missing`) has no type to fold at,
    so it gets a count-only `MissingState` instead (also for `CoMoment`,
    `AgeWeightedSum` and `Product`). It is valued `missing`, its fresh state is still the
    identity, and on widening it rebuilds the real state from its `recipe`,
    handing over the count (`withmissings!`)
  - the order statistics (`Quantile`, `Median`, `PercentRank`) are fieldless
    dependents over one accumulator, the unexported `SortedValues`: a sorted
    `Vector` (binary search plus memmove: under 300 ns per row to a 1,000-row
    window, growing linearly past ~10,000; DESIGN.md has the table) with
    `missing` and NaN counted, not stored, so it stays a group and every window
    slides it. Its value is the state itself, *borrowed* like `queuestates`'
    scratch: dependents read it within the emission and the projection drops
    it, so emission allocates nothing. `combine!` merges into `scratch` and
    swaps, which tolerates `dest` aliasing an input. `linearquantile` copies
    Statistics' type-7 arithmetic because `quantile(v; sorted = true)` scans
    all of `v` per call; `nearestrank` corrects `ceil(p * n)` to TA-Lib's exact
    rank. `PercentRank` reads the newest value from `Last`, a group via
    `WindowLastState`, so it keeps the running tier too
  - `MeanAbsDev` is a fieldless dependent over `Mean` and the unexported
    `WindowValues`, the values in arrival order: a `Vector` with a moving
    head, appended by `update!` and advanced by `downdate!` (exact under the
    oldest-first law), compacted once the dead prefix is half the vector, so
    both are O(1) amortized and a steady window allocates nothing. It scans
    the values once per emission (an `@simd` loop): deviations from a moving
    centre have no inverse, so O(window) per row is the floor, and it is
    TA-Lib's cost. It deliberately does not read `SortedValues`, whose
    insert/delete cost twice the whole indicator at short windows (DESIGN.md
    has the numbers). Its value is borrowed and `combine!` swaps a scratch
    vector, as for `SortedValues`
  - row terms (`name => f`, a `ColumnSpec`) are virtual input columns, not a
    summarizer feature: a constructor builds the summarizer over `name` and
    hands the specs to `withterms`, which wraps it in the inert `Termed`
    carrier only when a spec is a pair. Every constructor takes one
    `ColumnSpec` per column argument (never a method per Symbol/Pair
    combination). `prototypes` strips the carrier, so states, tiers and
    kernels never see it; don't add behaviour to `Termed` beyond `emptyvalue`.
    A custom summarizer's `rowterms` hook is collected beside the carrier's
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
  the binary transforms), and `checknewcolumns`, every appending transform's
  check that its output columns are new; and `rowsdataframe`, every
  summarizing transform's (and `addcolumns`') rows-to-frame step, generated
  per row type over DataFrames' vector-of-columns constructor, which compiles
  once where `DataFrame(rows)` compiles Tables.jl's path per row type (and is
  faster). `prototypes` expands dependencies topologically and
  returns the requested output names, which ride through the kernels in a
  `Val` to project hidden dependencies out of the output, plus the row terms
  it unwrapped. Every summarizing transform (these three, `intervalize`,
  `addrollingcolumns`, `summarizewindows`) reads its summarizers' input through
  `terminput`: the chunk's column table plus one vector per term (per chunk,
  behind `termvalues`' barrier, over `NamedTuple` rows built by `rowat`,
  typed by `termtype` (`promote_op` over that row type, as an unhoisted
  `barwindow` infers its terms; the row type is generated, `termrowtype`,
  since a tuple `map` over a wide table stops inferring) and by
  the values only when inference gives up, so a term's type is the same in
  every chunk and never abstract when it infers), with input types taken from
  that table so term types widen like columns. `addcolumns` stays value-typed
  (`rowvalues`; DESIGN.md says why). The term vectors must never reach an
  output.
  The two transforms that store rows (`addrollingcolumns`,
  `summarizewindows`) then cut that table to `storedcolumns` (`:time`, the
  keys and the summarizers' `inputcolumns`) with `projectinput`, per chunk:
  an unread `Union{Missing,T}` column would box every stored row.
  Per-run state lives in `SummaryFold`, never in reassigned closure captures
  (those get boxed). Its `groups` are built by the transform's `grouping`
  (`summaryspec`; `newgroups`/`widengroups`): the state tuple itself when
  keyless, a `GroupTable` when keyed, a `DenseGroups` over a declared
  `KeySet`. The fold kernels (`foldgroups!`, `foldcycles!`, `foldrunning!`,
  `foldintervals!`, and the flushes) are written once over the groups and
  reach them through helpers with a method per kind (`rowstates!`/`foldrow!`
  to fold a row, `closerows`/`closegroups!` to close), so the transforms
  dispatch rather than branch on the key mode. Only the keyless
  `foldcycles!`/`flushcycle` (which rebind the state tuple) and the keyed and
  dense `flushintervals!` (which close different intervals) keep a method per
  kind; there is no keyless `foldintervals!`, since `nokeyset` makes that grid
  dense. `preparechunk!`
  checks the key columns against the first chunk for all of them. The
  transform's `FoldConfig` (`protos`, `grouping`, `outs`) sits in
  `SummaryFold`'s abstract `cfg` field, so `preparechunk!` and every chunk step
  and flush (`summarizechunk!`, `summarizeflush`, `cyclechunk!`, `cycleflush`,
  `runningchunk!`, intervalize's) read it untyped and reach the kernels by a
  dispatch per chunk: the chunk iterators and drivers compile once, and only
  the kernels (and `sortedrows`, `summarize`'s keyed emission) specialize per
  summarizer set.
  The per-key states live in a `GroupTable{K,S}` (both parameters concrete,
  `K` the promoted schema's `storekeytype`, re-keyed by `widengroups` when a
  key column widens),
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
  `matchchunk!` and `joinchunk!`, built by `jointransform`, with the store,
  kernel, comparison and right window picked by dispatch
  (`newstore`/`widenstore`/`segment!`/`joincmp`/`rightcontext`, and
  `joinfreedom`, its flags). A new join direction adds those six methods,
  never a second driver. `joinnode` builds the node from a `JoinConfig` and a
  chunk step, for the joins and `applymodels` alike. The store is a
  `SlotStore` (a `Dict{K,Int}` of slot numbers over a `Vector{V}` of rows,
  shared with `lastrow` through `admitslot!`), and a match is a `Vector{V}`
  plus a `Vector{Bool}` mask (unmatched slots left undefined, hence
  `convertmatches` for a mid-chunk widening). Both keep `V` out of a `Union`,
  which Julia stores inline only if every member is isbits, so a `String` or
  `Missing`-admitting right column would otherwise cost a box per left row.
  See DESIGN.md's "Representing a match"; the zero-allocation kernel tests
  guard it. As in `addrollingcolumns`, the chunk step reads `cfg` through
  `JoinState`'s abstract field and the per-chunk driver (`joinchunk!`,
  `pullright!`, `matchchunk!`, `checknames`, `assemble`, `predictchunk!`)
  takes it `@nospecialize`, so only `segment!`'s kernels specialize per key,
  tolerance and direction
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
  `stop` via `merge`, which keeps the column position. The key names reach
  `storekeytype` and `lastsegment!` through `LastRowState`'s abstract
  `keynames` field, so the chunk iterator and driver compile once for every
  key; the flush builds its frame with `rowsdataframe`
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
  the input context as `asofjoin` does, so `warmup`'s `dropleadin!` drops the pre-`start`
  rows the fill was allowed to see. As in `lastrow`, the key names (and the
  tolerance) reach the kernels through `ForwardFillState`'s abstract fields,
  so the chunk iterator and driver compile once for every key
- `src/ring.jl` — `RowRing`, the package's one ring buffer: preallocated,
  overwriting its oldest row when full, with `ringback(r, j)` reading the j-th
  newest. A caller that downdates the row leaving sizes it one past its window
  and reads that row back after the push
- `src/monoidqueue.jl` — the monoid tier behind `addrollingcolumns` and
  `summarizewindows`: per key, `MonoidQueues` holds the key's rows and one
  Two-Stacks Lite `PartialQueue` per look-back (a front vector of suffix
  combines, a back state, the window's `lo`). Rows are addressed by sequence
  number, so compacting the dead prefix (once it is half the rows) never
  shifts a window's indices. `queuepush!` folds a row into every back and
  expires every window against the row's own time (sound because every later
  query is at or after it, and it bounds an unqueried key's rows, so the
  lead-in needs no expiry of its own); `expirewindow!` advances `lo` by the
  kernels' membership test verbatim (rolling.jl adds the `Bars` method, over
  `keepnewest!`). Callers name a window by its index (`expirequeue!`,
  `hasrows`, `queuestates`, `livestates`), never reaching into a
  `PartialQueue`;
  `queuestates` flips the live rows into the front once `lo` passes it and
  returns the window's fold, **borrowed** (the front's suffix or the `out`
  scratch), valid until that window's next query; `livestates` is its form
  for a window known to hold rows, which keeps `summarizewindows`' emission
  free of a `Union{Nothing,…}`, whose dispatch allocated per key per tick.
  Everything goes through the transforms' tuple folds (`freshall!`,
  `updateall!`, `combinenodes!`), never a new generated fold. The queues are a
  `Vector`, not an `NTuple{W}`: with `W` a type parameter, every window count
  recompiled the queue methods (a harness's lead-in case, sharing a key and
  row type with a three-window case, compiled 0.2 s slower). `BarQueue` is
  the same algorithm over `barwindow`'s own folds and stays separate (see
  `barwindow.jl`)
- `src/tiers.jl` — the per-accumulator window tiers shared by
  `addrollingcolumns` and `summarizewindows`. `tiering` partitions the realized
  states once per run and per widening (each tier's templates picked by the
  generated `tierpick` past a `Val` of the permutation: a tuple generator there
  compiled a widening `grow_to!` chain per state type): running (a group whose `freshwindowed`
  state is `isinvertible`), monoid (another monoid), refold (anything else), and
  no tier for a singleton-typed (fieldless, dependent) state, unless every
  state is one, when they keep their structural tier so something tracks
  membership. `mergestates` splices the tiers' tuples back into topological
  order by a generated, compile-time permutation, so emission is the ordinary
  `summaryvalues` and dependents read across tiers. An absent tier is
  `nothing`/`()`, so a single-tier call compiles to a single-algorithm kernel.
  `RunningTable` pools retired groups (windowed trackers own vectors).
  `evictgroup!` returns the group while it still holds rows, so the caller can
  `refoldgroup!` it from its live rows when a state asks (issue #97): from the
  buffer past the head for a time window (`evictbuffered!`), from the key's
  ring for `Bars`. The refold is the whole group through `freshall!`/
  `updateall!`, already compiled for admission, behind a `@noinline` barrier:
  a generated fold over only the asking states took the window kernels'
  compile time from about 0.4 to 3 s.
  `rebuildbuffer` and `rebuildmonoids` choose a rebuild's buffer and queues
  for both transforms, and `replayrunning!`, `replaymonoids!` and
  `replayoldmonoids!` rebuild from the live rows. Built-in states only demote,
  so the buffer a rebuild needs already exists; `monoidbuffer` gathers one
  from the queues for a custom state that promotes
- `src/barwindow.jl` — `barwindow`, the count-window state (unexported
  extension API for recursive states needing their own trailing n rows). It
  reuses `prototypes` and `tiering`: groups slide windowed states over a
  `RowRing` of `n + 1` stored rows (row terms evaluated on admission, their
  types inferred from `intypes`), the other monoids sit in `BarQueue`, a
  preallocated two-stack queue (the algorithm of `monoidqueue.jl`, over its
  own folds; measured 3-13x faster than a segment tree and flat in `n`;
  DESIGN.md has the table), and a plain summarizer is rejected.
  `newbarwindow` builds it unspecialized (its type is decided at runtime
  anyway), once per state and per widening; a key's new state is `fresh(st)`,
  a copy.
  A stored row is cut to `:time`, the terms and the columns `readcolumns`
  names, hoisted or not. It must not call the transforms' tuple folds
  (`updateall!`, `freshall!`, `summaryvalues`, `combinenodes!`): it runs inside an outer state's
  `update!`/`value`, which those folds call, and inference's recursion limit
  on a repeated caller-to-callee edge would leave the inner fold dispatching
  and allocating per row. Its generated `bar*` folds have edges of their own
  (`barrefold!`, its refold for a state that asks, uses them too). It shares
  only `anyrefold`, which an outer group reads through the window's own
  `needsrefold`, the constant `false`, so no edge repeats;
  `test/jet.jl` guards the embedded path. `barinputcolumns` is the outer
  state's `inputcolumns`, sharing `storedcolumns`' union (`readcolumns`);
  `barterms` is its `rowterms`, hoisting the window's terms (wrapped in
  `BarTerm`, which `termvalues` holds to the window's typing rule up front)
  into the transform's `terminput` under reserved `hoistednames`, scoped by the outer
  summarizer's first output name (`#barterm#lastn#d`) so two windows' terms
  never share a namespace, after which `hoistedby = outer` makes the window
  read them from those columns (`HoistedTerms`) instead of evaluating them.
  Never key a hoisted term by its own name: a wrapper that drops `rowterms`
  would then have the window read a real input column of that name as the
  term, silently
- `src/rolling.jl` — `addrollingcolumns`, a node over `(summarized, p)`,
  the summarized side `p` itself unless `from` names another, read over the
  widened context and `ctx`: the plan shares them as for any diamond (a run
  per side made a chain of d stages run its source 2^d times, issue #98), and
  the tee drops the lead-in from the output side. A `Bars` window counts back
  from the run's start, so it makes the stage not `startfree`. Whenever the
  context widens (shared or not), before the output side's first pull
  (`afterleadin`), `admitleadin!` reads the whole
  lead-in through the summarized side with its own kernel,
  `leadinsegment!`, evicting against the newest admitted time (the monoid
  queues expire on admission anyway); an output pull first would queue the
  lead-in at full input width. `afterleadin`'s `LeadInGate` keeps abstract
  fields on purpose: a wrapper capturing its input's iterator type nests every
  earlier stage's type in a chain's, and `chain8-keyed` then took inference
  over five minutes to compile, against 8 s erased. The chunk step reads
  `cfg` back through the gate, and the per-chunk driver (`rollchunk!`,
  `pullsummarized!`, `setvaltype!`, `admitleadin!`, `assemble`,
  `assembleempty`) takes it `@nospecialize`, so the iterator and the driver
  compile once for every summarizer set; only the kernels and `rolltiers`'
  typed half (past the schema's type values) specialize. The kernels' inlining
  is load-bearing: a `@noinline` barrier around `emitroll!`'s `summaryvalues`
  saved 5% of their compile time and cost the refold benchmark 12% at runtime
  (`notes/compile-time.md`). Summarizing its own input (`LeadInGate.self`),
  the lead-in's first chunk takes the augmented side's checks. One
  kernel, `rollsegment!`, over a `RollTiers` (a shared row buffer with
  per-window eviction heads, per-window running tables, per-key monoid queues
  owning their rows, refold templates). Per row:
  admit into every tier, advance each window's head (downdating running
  groups), then emit each window from the tiers' states for the key; a
  `nothing` from any tier is the empty window. The refold tier folds from the
  window's own head, so it needs no time test. Widening rebuilds every tier
  from the live rows, re-partitioning (a non-invertible widening demotes only
  that accumulator); float sums stay running because the compensated states
  evict NaN/±Inf rows cleanly. `Bars(n)` windows change membership only: each
  key keeps a `RowRing` of its newest rows (sized for the longest count, plus
  one), a row leaves on admission when pushed `n + 1` deep (`admitrings!`
  downdates it), the window's queue holds the key's last `n` rows, the refold
  tier
  folds the ring, and a tier seeing fewer than `n` rows returns `nothing`,
  whose empty row is all `missing`. The shared buffer serves only time windows;
  a Bars window's head sits at its end. Windows now differ in value type, so
  `wins` peels each window's (look-back, empty row, value vector) together:
  never index the value vectors by a runtime window number. A widening with
  Bars windows rebuilds rings and queues from each key's longest live suffix
  (`barsuffixes`). Every admitted row passes through `join.jl`'s `rowat`, which
  is generated to write each field straight into the row type: Base's tuple
  `map` stops unrolling at 32 columns, a 10x cliff on a wide input (issue
  #98), and a row built as a tuple first is boxed per row when it is passed
  on (a row term's argument) rather than stored, once a column admits
  `missing`. Each emitted row's key comes from `keyat`, generated too so it
  builds no closure: the kernel can outgrow the compiler's inlining budget (a
  `missing`-admitting input column and a compensated float sum suffice), and a
  closure that isn't inlined allocates per row. `keyat` (and `keyvalues`, its
  row form) type the key by its columns, not its values (DESIGN.md, issue #114)
- `src/intervalize.jl` — `intervalize`, the third binary transform: summarize
  over the intervals a `clock` pipeline defines (`[bₖ, bₖ₊₁)`, timestamped at
  `bₖ₊₁`). It is the `summarizecycles` fold, closing on clock boundaries.
  Keyless output is a regular grid (empty intervals get `emptyvalue`s, output
  types promoted via the shared `promotedvaluetype`), run as the declared-key
  path over `nokeyset`, the one empty key, whose lookups compile away; keyed
  output is sparse (present keys only, via `closecycle!`). The `chunkmap` over
  the data stream pulls clock boundaries into a concrete `Vector{T}` per chunk
  (the pull is the only dynamism; the per-row kernel stays dispatch-free),
  reusing `SummaryFold` whole; `closelast` closes the trailing partial at
  `stop`. The chunk step and flush read the config through the
  `SummaryFold`'s erased `cfg`, so only the kernels specialize per summarizer
  set
- `src/windows.jl` — `summarizewindows`, the clock-sampled trailing window
  (`[τ - lookback, τ)` at each tick): `intervalize`'s driver (`chunks.jl`'s `IntervalCursor`
  and `pullpast!`, a concrete tick vector per chunk) over `addrollingcolumns`'
  row buffer, eviction head and tiers. One admission kernel (`windowrows!`)
  and one tick close (`closewindow!`): evict (downdating the running table),
  expire each key's queue and drop emptied ones, fold the refold `GroupTable`
  from the live rows, emit, retire. A key is present in a tier exactly when
  its window has rows, so the first tier present (running, monoid, refold) gives
  the keys to emit and the others are looked up by key (`tiervalues`). Keyed
  output is sparse, plus one *vanish* row of empty values when a key's window
  empties, decided against the previous tick's *emitted* keys; that row stops
  a per-key as-of consumer (`applymodels`) from using a stale summary. A
  declared `keyset` rides in `WindowConfig`'s type (`Nothing` otherwise) and
  replaces the sort and vanish merge with `emitdense!`, a lookup per declared
  key per tick; keyless output is `emitdense!` over `nokeyset`, as in
  `intervalize`, and `admissionkeys` skips its admission check. The
  declaration is checked where the primary tier makes a group or queue (a key's
  first row); a refold primary groups only at ticks, so it checks on admission
  instead. As in `addrollingcolumns`, the chunk step and flush read `cfg`
  through `WindowState`'s abstract field and the per-chunk driver takes it
  `@nospecialize`, so only the kernels specialize per summarizer set
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
    Both read the input, a diamond the plan shares (issue #105), so it has no
    sharing code of its own.
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
  `segment!`/`joincmp`/`rightcontext` methods (so the forward-looking code
  stays in the submodule), with the match direction, tie-break and context
  widening (`stop + tolerance`) inverted. Because it matches the *earliest*
  qualifying right row, it buffers right rows per key until a left row
  consumes or outruns them, and proving a key has no future match drains the
  right stream: worst case O(right rows), against `asofjoin`'s O(keys) store. It shares the
  `Vector{V}` + mask match buffer, and `KeyBuffer` is mutable, so a store
  lookup answers with a pointer and never boxes
- `src/precompile.jl` — the PrecompileTools workload over the main pipeline
  paths; every new operator adds a path here
