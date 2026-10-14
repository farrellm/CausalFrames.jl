"""
    LinearRegression(predictors, response::ColumnSpec; intercept = true,
                     name = nothing) -> Summarizer

An ordinary least squares fit of `response` on `predictors`.

# Arguments
- `predictors`: a column name or row term `name => f` (see [`ColumnSpec`](@ref)),
  or a collection of distinct ones (`K` of them). None may be named
  `intercept`.
- `response`: the response column, or a row term.

# Keywords
- `intercept = true`: fit a constant term. Without it, `r2` is the uncentered
  R².
- `name = nothing`: a prefix for every output column, `{name}_{column}`.
  Needed to request two regressions together, since both would otherwise
  produce `n`, `r2` and `stderr`.

# Output columns

| column | meaning |
|---|---|
| `n` | rows folded (`Int`, never `missing`) |
| `r2` | coefficient of determination |
| `stderr` | residual standard error |
| `intercept_beta`, `intercept_tstat` | the constant term and its t statistic (only with `intercept`) |
| `{p}_beta`, `{p}_tstat` | per predictor `p`, in order |

```jldoctest
df = DataFrame(time = 1:5, x = [1.0, 2.0, 3.0, 4.0, 5.0], z = [0.0, 1.0, 0.0, 1.0, 1.0],
               y = [3.1, 7.9, 7.2, 12.1, 13.8])
p = readtable(df) |> summarize(LinearRegression([:x, :z], :y; name = :m1))
names(load(Context(0, 10), p))

# output

10-element Vector{String}:
 "time"
 "m1_n"
 "m1_r2"
 "m1_stderr"
 "m1_intercept_beta"
 "m1_intercept_tstat"
 "m1_x_beta"
 "m1_x_tstat"
 "m1_z_beta"
 "m1_z_tstat"
```

The statistics share one element type (`Float64` for integer input) and admit
`missing` if an input does. No rows, or a `missing` in any input, gives
`missing` statistics. A rank-deficient fit (collinear predictors, or too few
rows) gives `NaN` rather than an error; with no residual degrees of freedom the
coefficients are exact but `stderr` and the t statistics are `NaN`.

With an intercept the fit is computed from the same centred accumulators as
[`Variance`](@ref) and [`Covariance`](@ref), so large levels don't cancel, plus
[`Count`](@ref) and each column's [`Sum`](@ref) for the means. Without one it
is computed from [`Count`](@ref), [`SumPower`](@ref)`(·, 2)` and
[`DotProduct`](@ref). Either way, regressions over overlapping columns, and the
statistics beside them, share accumulators. With `K ≥ 2` each emitted row allocates a `K × K`
workspace; `K = 1` allocates nothing.
"""
struct LinearRegression{P,Y} <: GroupSummarizer
    intercept::Bool
    name::Union{Nothing,Symbol}
end

function LinearRegression(predictors, response::ColumnSpec;
    intercept::Bool = true, name::Union{Nothing,Symbol} = nothing)
    specs = Tuple(p isa Pair ? p : Symbol(p) for p in predictors)
    ps = map(colname, specs)
    isempty(ps) &&
        throw(ArgumentError("LinearRegression requires at least one predictor"))
    allunique(ps) || throw(ArgumentError(
        "LinearRegression predictors must be unique, got $ps"))
    # Catches, e.g., a predictor named `intercept`, which would otherwise fail
    # later in the NamedTuple constructor with no mention of this summarizer.
    outs = regnames(ps, name, intercept)
    allunique(outs) || throw(ArgumentError(
        "LinearRegression output columns must be unique, got $outs"))
    return withterms(LinearRegression{ps,colname(response)}(intercept, name),
        specs..., response)
end
# A lone name or row term; a string is one name, not a collection of
# characters.
LinearRegression(predictor::Union{ColumnSpec,AbstractString},
    response::ColumnSpec; kwargs...) =
    LinearRegression((predictor,), response; kwargs...)

# Fieldless, with every name a type parameter so the two-argument `value`
# infers. NN and SN are the output names (the row count, then the statistics);
# AN every accumulator name read, deduplicated (a predictor may be the
# response); SP and SY the predictor and response sums, QY the response's power
# sum, GN the packed upper triangle of the cross-product matrix, DN the
# predictor-response cross products; I the intercept flag.
struct LinearRegressionState{NN,SN,AN,SP,SY,QY,GN,DN,I} <: SummarizerState end

regname(::Nothing, base::Symbol) = base
regname(prefix::Symbol, base::Symbol) = Symbol(prefix, :_, base)

# The output names in emission order: the row count, `r2`, `stderr`, the
# intercept's pair if any, then a (beta, tstat) pair per predictor.
function regnames(P::Tuple, name, intercept::Bool)
    ns = Symbol[regname(name, :n), regname(name, :r2), regname(name, :stderr)]
    if intercept
        push!(ns, regname(name, :intercept_beta))
        push!(ns, regname(name, :intercept_tstat))
    end
    for p in P
        push!(ns, regname(name, Symbol(p, :_beta)))
        push!(ns, regname(name, Symbol(p, :_tstat)))
    end
    return Tuple(ns)
end

# The cross products a fit reads. With an intercept they are the centred
# co-moments, which Variance, Covariance and Correlation share and which don't
# cancel under a large level. Without one the fit is on the raw products by
# definition: the canonical dot product, with a squared term as SumPower(c, 2)
# rather than DotProduct(c, c), the same value under a name Moment shares.
crossdep(a::Symbol, b::Symbol, intercept::Bool) =
    intercept ? canonicalcomoment(a, b) :
    a === b ? SumPower(a, 2) : canonicaldot(a, b)
crossname(a::Symbol, b::Symbol, intercept::Bool) =
    intercept ? canonicalcomomentname(a, b) :
    a === b ? Symbol(a, :_sumpower_2) : canonicaldotname(a, b)

# Entry (i, j), i <= j, of a K x K matrix packed row-major over its upper
# triangle — the layout GN is built in.
@inline gramindex(i::Int, j::Int, K::Int) =
    (i - 1) * K - ((i - 1) * (i - 2)) ÷ 2 + (j - i + 1)

function dependencies(r::LinearRegression{P,Y}) where {P,Y}
    I = r.intercept
    ds = Summarizer[Count(), crossdep(Y, Y, I)]
    for (i, p) in pairs(P)
        push!(ds, crossdep(p, p, I))
        push!(ds, crossdep(p, Y, I))
        for j in (i+1):length(P)
            push!(ds, crossdep(p, P[j], I))
        end
    end
    if r.intercept
        push!(ds, Sum(Y))
        for p in P
            push!(ds, Sum(p))
        end
    end
    return Tuple(ds)
end

function emptyvalue(r::LinearRegression{P,Y}) where {P,Y}
    ns = regnames(P, r.name, r.intercept)
    return NamedTuple{ns}((0, ntuple(_ -> missing, length(ns) - 1)...))
end

function fresh(r::LinearRegression{P,Y}, ::NamedTuple) where {P,Y}
    ns = regnames(P, r.name, r.intercept)
    I = r.intercept
    gn = Tuple(crossname(P[i], P[j], I) for i in eachindex(P) for j in i:length(P))
    dn = map(p -> crossname(p, Y, I), P)
    qy = (crossname(Y, Y, I),)
    sy = r.intercept ? (Symbol(Y, :_sum),) : ()
    sp = r.intercept ? map(p -> Symbol(p, :_sum), P) : ()
    an = Tuple(unique((qy..., gn..., dn..., sy..., sp...)))
    return LinearRegressionState{first(ns),Base.tail(ns),an,sp,sy,qy,gn,dn,
        r.intercept}()
end

# The dependency values this regression reads, and the tuple type they are
# *declared* to have. `typeof` of the values depends on them (a `missing`
# reports `Missing`, not `Union{Missing,_}`), which would narrow the output
# eltype and push the emission into runtime dispatch; `Base.promote_op` gives
# the declared type, a compile-time constant.
@inline depvalues(vals::NamedTuple, ::Val{NS}) where {NS} =
    values(NamedTuple{NS}(vals))
@inline deptypes(::Type{V}, ::Val{NS}) where {V<:NamedTuple,NS} =
    Base.promote_op(depvalues, V, Val{NS})

# The statistics' shared element type: the declared types promoted, run through
# the centered identity's arithmetic, then made floating point for NaN. The
# promotion is generated, so it is a constant however many fields there are: a
# recursive one stops folding once the fields mix `Union{Missing,Float64}`
# co-moments with `Union{Missing,Int}` sums (a K >= 2 fit over integers).
@generated promotefieldtypes(::Type{T}) where {T<:Tuple} =
    promote_type(fieldtypes(T)...)

tofloat(x) = float(x)
tofloat(::Missing) = missing

@inline function regtype(::Type{T}) where {T<:Tuple}
    A = promotefieldtypes(T)
    Q = Base.promote_op(-, A, Base.promote_op(/, Base.promote_op(*, A, A), Int))
    return Base.promote_op(tofloat, Base.promote_op(/, Q, Q))
end

@inline anymissing(::Tuple{}) = false
@inline anymissing(t::Tuple) = ismissing(first(t)) || anymissing(Base.tail(t))

# Narrow a dependency value to the compute type. The caller has already
# returned if anything was missing, but the explicit `ismissing` split keeps
# `convert` static; `convert` alone goes dynamic over tuples of two or more
# Union-typed values (the K >= 2 cross products).
@inline convdep(::Type{Vc}, x) where {Vc} = ismissing(x) ? zero(Vc) : convert(Vc, x)
@inline convall(::Type{Vc}, ::Tuple{}) where {Vc} = ()
@inline convall(::Type{Vc}, t::Tuple) where {Vc} =
    (convdep(Vc, first(t)), convall(Vc, Base.tail(t))...)

# A dependency read back as a scalar. The empty name tuple is the
# no-intercept case, whose value is never used.
@inline regscalar(::Type{Vc}, ::NamedTuple, ::Val{()}) where {Vc} = zero(Vc)
@inline regscalar(::Type{Vc}, vals::NamedTuple, ::Val{NS}) where {Vc,NS} =
    convdep(Vc, only(depvalues(vals, Val(NS))))

# Simple regression in closed form, on scalars with no workspace: the common
# case, run per row under addsummarycolumns and addrollingcolumns. Every
# possible 0/0 is in floating point, and sqrt arguments are clamped at zero, so
# a degenerate window gives NaN rather than an error. With an intercept (I)
# the cross products `g`, `d` and `qy` arrive centred, and the sums `s`, `sy`
# serve only the means; without one they are raw.
@inline function regstats(::Type{Vc}, ::Val{I}, ::Val{M}, n::Vc,
    g::NTuple{G,Vc}, d::NTuple{1,Vc}, s::Tuple{Vararg{Vc}}, sy::Vc,
    qy::Vc) where {Vc,I,M,G}
    nan = Vc(NaN)
    sxx = g[1]
    sxy = d[1]
    sst = qy
    dof = I ? n - 2 : n - 1
    sxx > zero(Vc) || return ntuple(_ -> nan, Val(M))
    beta = sxy / sxx
    sse = sst - beta * sxy
    r2 = one(Vc) - sse / sst
    positive = dof > zero(Vc)
    sigma2 = positive ? max(sse, zero(Vc)) / dof : nan
    stderr = sqrt(sigma2)
    tbeta = beta / sqrt(sigma2 / sxx)
    if I
        xbar = s[1] / n
        alpha = sy / n - beta * xbar
        talpha = alpha / sqrt(sigma2 * (one(Vc) / n + xbar * xbar / sxx))
        return (r2, stderr, alpha, talpha, beta, tbeta)
    end
    return (r2, stderr, beta, tbeta)
end

# Multiple regression by Cholesky of the positive semidefinite cross-product
# matrix. `check = false` turns rank deficiency into a flag rather than an
# exception; the inverse gives the standard errors and the intercept's
# quadratic form.
function regstats(::Type{Vc}, ::Val{I}, ::Val{M}, n::Vc, g::NTuple{G,Vc},
    d::NTuple{K,Vc}, s::Tuple{Vararg{Vc}}, sy::Vc, qy::Vc) where {Vc,I,M,G,K}
    nan = Vc(NaN)
    rhs = d
    sst = qy
    dof = I ? n - K - 1 : n - K
    A = Matrix{Vc}(undef, K, K)
    for i in 1:K, j in i:K
        A[i, j] = g[gramindex(i, j, K)]
    end
    F = cholesky!(Symmetric(A, :U); check = false)
    issuccess(F) || return ntuple(_ -> nan, Val(M))
    b = Vector{Vc}(undef, K)
    copyto!(b, rhs)
    ldiv!(F, b)
    ssr = zero(Vc)
    for i in 1:K
        ssr += b[i] * rhs[i]
    end
    sse = sst - ssr
    r2 = one(Vc) - sse / sst
    sigma2 = dof > zero(Vc) ? max(sse, zero(Vc)) / dof : nan
    stderr = sqrt(sigma2)
    Ci = inv(F)
    coefs = ntuple(Val(2 * K)) do k
        i = (k + 1) >> 1
        isodd(k) ? b[i] : b[i] / sqrt(sigma2 * Ci[i, i])
    end
    if I
        xbar = ntuple(i -> s[i] / n, Val(K))
        alpha = sy / n
        for i in 1:K
            alpha -= b[i] * xbar[i]
        end
        quad = zero(Vc)
        for i in 1:K, j in 1:K
            quad += xbar[i] * Ci[i, j] * xbar[j]
        end
        talpha = alpha / sqrt(sigma2 * (one(Vc) / n + quad))
        return (r2, stderr, alpha, talpha, coefs...)
    end
    return (r2, stderr, coefs...)
end

@inline function value(::LinearRegressionState{NN,SN,AN,SP,SY,QY,GN,DN,I},
    vals::NamedTuple) where {NN,SN,AN,SP,SY,QY,GN,DN,I}
    V = regtype(deptypes(typeof(vals), Val(AN)))
    M = length(SN)
    nrow = NamedTuple{(NN,),Tuple{Int}}((vals.count,))
    if Missing <: V && anymissing(depvalues(vals, Val(AN)))
        return merge(nrow,
            NamedTuple{SN,NTuple{M,V}}(ntuple(_ -> missing, Val(M))))
    end
    Vc = nonmissingtype(V)
    stats = regstats(Vc, Val(I), Val(M), convert(Vc, vals.count),
        convall(Vc, depvalues(vals, Val(GN))),
        convall(Vc, depvalues(vals, Val(DN))),
        convall(Vc, depvalues(vals, Val(SP))),
        regscalar(Vc, vals, Val(SY)), regscalar(Vc, vals, Val(QY)))
    return merge(nrow, NamedTuple{SN,NTuple{M,V}}(stats))
end
