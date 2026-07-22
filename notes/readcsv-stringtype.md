# `readcsv` and CSV.jl's `stringtype`

An investigation record, not design law — `DESIGN.md` remains the source of
truth. This note exists so the next person considering a `stringtype` argument
for `readcsv` does not have to re-derive the constraint or re-run the
measurements. **No code was changed as a result of it.**

The question was: add a `stringtype` argument to `readcsv` and default it to
CSV.jl's own default. It turns out the literal version of that is impossible
without changing how `readcsv` suppresses type inference, and the measurements
argue against changing the default anyway.

Everything below was verified against CSV 0.10.16.

## What `readcsv` does today

`src/operators.jl` pins the string representation in two places:

- `typesfunction` returns `String` for every column the user's `types` argument
  does not name — this is what guarantees "types are **not** inferred".
- `csvchunks` passes `stringtype = String` to `CSV.Chunks` / `CSV.read`.

## Finding 1 — CSV.jl's default `stringtype` cannot be a per-column type

CSV.jl's default is the **abstract** `InlineString` (`CSV/src/CSV.jl`,
`DEFAULT_STRINGTYPE = InlineString`). It is accepted as the `stringtype`
keyword (the `StringTypes` union in `CSV/src/utils.jl`) but not as a column
type: `parserow`'s type dispatch chain (`CSV/src/file.jl`) branches only on the
concrete `InlineString1`…`InlineString255`, and `nonstandardtype` returns
`Union{}` for the abstract type, so it is not routed to `parsecustom!` either.
It falls through to `error("Column $i bad column type")`.

| `types` fallback | `stringtype` | result |
|---|---|---|
| `InlineString` | `InlineString` | **ERROR** `Column 1 bad column type: InlineString` |
| `String` | `InlineString` | all `String` (the `types` entry wins per column) |
| `PosLenString` | `PosLenString` | all `PosLenString` |
| `InlineString7` | `InlineString` | all `String7` |
| *(none — detection)* | `InlineString` | `String7`/`String15`, but numbers infer as `Int64`/`Float64` |

So `stringtype` alone cannot reach CSV's default while `types` forces every
unnamed column to a string. The two mechanisms are in direct conflict.

## Finding 2 — CSV's default is reachable only via detection + `typemap`

Leaving unnamed columns to detection and adding a `typemap` that maps every
detected scalar back to a string does work, and preserves raw cell text:

```julia
typemap = IdDict{Type,Type}(T => stringtype for T in
    (Int8, Int16, Int32, Int64, Int128, Float16, Float32, Float64,
     Bool, Date, DateTime, Time))
```

`"3.50"` survives as `"3.50"` (not `3.5`); dates and bools come back as their
source text. The abstract `InlineString` *is* legal as a `typemap` value — the
detection path routes it through `pickstringtype` (`CSV/src/detection.jl`,
`CSV/src/utils.jl`), yielding adaptive `String3`/`String7`/`String15` widths.

Costs of that route:

- Type detection runs on every untyped column — work `readcsv` currently avoids
  entirely.
- Widths are detected **per chunk** (`CSV/src/chunks.jl` re-initializes column
  types for each chunk), so one column can be `String7` in one chunk and
  `String15` in the next.
- An all-empty column comes back as `SentinelArrays.MissingVector` (eltype
  `Missing`) instead of `Union{Missing,String}`.

## Finding 3 — per-chunk eltype divergence is *not* an invariant violation

Counterintuitive, so worth recording: the frame's "shared schema" invariant is
names-only (`src/frame.jl` compares `names(c)`), and `Tables.schema` explicitly
promotes each column's per-chunk eltypes — the same promotion
`reduce(vcat, frame.chunks)` performs. Mixed `String7`/`String15` chunks are
therefore tolerated by design. That removes the strongest objection to inline
strings, but not the cost of detection.

## Finding 4 — `PosLenString` is a viable drop-in, with one sharp edge

`PosLenString` is concrete, so it needs no new machinery: the `typesfunction`
fallback and the `stringtype` keyword would both simply change value.

Pipeline compatibility, all verified:

- `df[lo:hi, :]` (the clip in `CSVProducer`) keeps a lazy `PosLenStringVector`;
  boolean and index subsetting too.
- `Tables.columntable` plus row iteration works behind the function barrier.
- `reduce(vcat, chunks)` in `DataFrame(frame)` materializes to
  `Vector{PosLenString}`.
- Pooling still applies (`PooledVector{PosLenString}`).
- `PosLenString("10.0") == "10.0"` is `true`, so value assertions such as
  `df.bid == ["10.0", …]` in `test/operators.jl` would keep passing;
  `eltype(df.bid) == String` would not.

Measured on a generated 61.1 MB / 2M-row CSV (`time` typed `Int`, four string
columns):

| fallback | parse | allocations | retained by a 1000-row clip |
|---|---|---|---|
| `String` (today) | 0.549 s | 360 MB | 0.1 MB |
| `PosLenString` | 0.238 s | 116 MB | **61.1 MB** |

The sharp edges:

- **Retention.** A `PosLenString` is a view into the source buffer, so a frame
  clipped to a small window pins the whole file buffer for its lifetime. The
  buffer is an `Mmap.mmap` (`CSV/src/utils.jl`), so this is file-backed pages
  rather than private heap — but it pins address space and the file, and on
  Windows modifying or deleting a mapped file can fail. This lands directly on
  the package's central use case: a narrow context window over a large file.
- **Read-only columns escape the copy point.** `DataFrame(frame)` returns
  `copy(only(chunk))` for a one-chunk frame — a read-only `PosLenStringVector`
  — but `vcat`s to a mutable `Vector{PosLenString}` for a multi-chunk frame.
  Mutability of the user's DataFrame would depend on the chunk count, exposing
  an internal detail.
- **`eltype` is no longer `String`**, breaking `::String` annotations,
  String-dispatching user code, and anything persisting the columns without an
  explicit `String.(col)`. In `asofjoin`, key columns become `Dict` keys of
  `PosLenString`: functional, but every stored key pins its buffer for the run.

## Recommendation, if this is revisited

Add `stringtype` as a keyword that **defaults to `String`** (today's behavior,
safe for the clip-a-small-window case) and require a *concrete* type —
`String`, `CSV.PosLenString`, or a sized `InlineString` such as `String15` —
raising a `readcsv` `ArgumentError` naming the valid choices rather than letting
CSV's `Column 1 bad column type` surface. That keeps a single code path and zero
inference while making the measured 2.3x available opt-in.

Matching CSV.jl's default literally means adopting the detection + `typemap`
route from Finding 2, with the costs listed there.
