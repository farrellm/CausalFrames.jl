# The agent skill in `skills/causalframes/`, generated from the docs so there is
# no second copy to drift: SKILL.md is the README behind a short preamble, and
# `references/` holds each API page with its docstrings rendered in place, plus
# the Recipes. Included by `make.jl`, which defines `DOCS_ANCHOR_LINK` and
# `DOCS_PAGE_LINK` and loads every module the `@docs` blocks name.

const SKILL_FRONTMATTER = """
---
name: causalframes
description: >-
  How to use CausalFrames.jl, the Julia package for time-series tables built
  from lazy `|>` pipelines: CSV/parquet sources, filterrows, addcolumns,
  asofjoin, forwardfill, rolling and clock-windowed summaries, bars, and MLJ
  model fitting. Use this skill whenever Julia code does or should do
  `using CausalFrames`, or the user wants time-series, tick or bar data
  processed in Julia (resampling, rolling statistics, as-of joins, OHLC bars,
  point-in-time features free of lookahead), even if they don't name the
  package.
---
"""

const SKILL_PREAMBLE = """
## Using this skill

This page is the package README: the concepts, every operator and summarizer
in one table, and worked examples. Each name links to its full docstring in
`references/` (signature, every argument and keyword with its default and
constraints, and an example with its output), and `references/recipes.md`
covers patterns that aren't obvious from the tables. Read an operator's
docstring before relying on its keywords; the details (what `key` accepts, how
`tolerance` widens the input, which output columns appear) are there, not here.

- Stay in the pipeline. Build with `|>` and materialize once, with `load(ctx,
  p)` or `DataFrame(load(ctx, p))`. Most things that seem to need a detour
  through DataFrames have a pipeline form; check the recipes before leaving.
- Causality is the point of the package: an operator's output at time `t`
  uses only rows at or before `t`, so features computed with it carry no
  lookahead. Reach for `CausalFrames.Acausal` only when forward-looking data
  is the goal (labels, targets), and say so to the user.
- Sources clip to the context's `[start, stop)`, and `:time` must be
  non-decreasing; the readers' `sort` keyword handles unsorted input.
- Examples print their result after `# output`. That is the expected output,
  not code to run.
- Check a pipeline by running it on a few rows from
  `readtable(DataFrame(...))` and comparing against what you expect.

"""

const SKILL_GENERATED_NOTE = """
<!-- Generated from the CausalFrames.jl docs by docs/skill.jl; edit README.md,
     docs/src and the docstrings instead, then run docs/make.jl. -->
"""

# Documenter syntax that means nothing outside the docs site: doctests become
# plain Julia blocks and cross-references become code spans.
function plainmarkdown(text)
    text = replace(text, r"```jldoctest[^\n]*" => "```julia")
    return replace(text, r"\[([^\[\]]*)\]\(@ref(?: [^)]*)?\)" => s"\1")
end

# Docstring headers (`# Arguments`) sit under the reference's `## name`.
function demoteheaders(text)
    infence = false
    lines = map(split(text, '\n')) do line
        startswith(line, "```") && (infence = !infence)
        return !infence && startswith(line, "#") ? "###" * line : line
    end
    return join(lines, '\n')
end

# The raw docstrings of a `@docs` entry, as Documenter reads it: a name,
# possibly qualified, or a call giving the method signature, as in
# `Base.merge(::CausalPipeline, ...)`. Raw, because `Markdown` loses text when
# it round-trips some code spans (`[start, stop)` among them).
function docstring(entry)
    ex = Meta.parse(entry)
    if Meta.isexpr(ex, :call)
        argtypes = map(ex.args[2:end]) do arg
            if Meta.isexpr(arg, :...)
                return Vararg{Core.eval(Main, arg.args[1].args[1])}
            end
            return Core.eval(Main, arg.args[1])
        end
        sig = Tuple{argtypes...}
        ex = ex.args[1]
    else
        sig = Union{}
    end
    mod, name =
        Meta.isexpr(ex, :.) ? (Core.eval(Main, ex.args[1]), ex.args[2].value) :
        (Main, ex)
    binding = Docs.aliasof(Docs.Binding(mod, name))
    texts = String[]
    for m in Docs.modules
        multidoc = get(Docs.meta(m), binding, nothing)
        multidoc === nothing && continue
        for msig in multidoc.order
            sig <: msig || continue
            parts = multidoc.docs[msig].text
            all(p -> p isa AbstractString, parts) ||
                error(
                    "docstring of $entry interpolates values, which the skill can't render",
                )
            push!(texts, demoteheaders(rstrip(join(parts))))
        end
    end
    isempty(texts) && error("no docstring found for $entry")
    return join(texts, "\n\n---\n\n")
end

# An API page with each `@docs` block replaced by its docstrings, separated by
# rules where a block holds several. `## Internals` sections are left out.
function apireference(file)
    out = String[]
    entries = String[]
    skipping = false
    indocs = false
    for line in split(read(file, String), '\n')
        if startswith(line, "## ")
            skipping = line == "## Internals"
        end
        skipping && continue
        if line == "```@docs"
            indocs = true
            empty!(entries)
        elseif indocs && line == "```"
            indocs = false
            push!(out, join(docstring.(entries), "\n\n---\n\n"))
        elseif indocs
            push!(entries, line)
        else
            push!(out, line)
        end
    end
    # The longer pages run to hundreds of lines; a contents line under the
    # title lets an agent jump to the section it needs.
    sections = [line[4:end] for line in out if startswith(line, "## ")]
    insert!(out, 2, "\nContents: " * join(sections, ", ") * ".")
    return plainmarkdown(join(out, '\n'))
end

function recipesreference(file)
    text = replace(read(file, String), r"```@meta\n[\s\S]*?```\n\n" => "")
    return plainmarkdown(text)
end

# Links into the docs site point at the matching reference file instead.
function skilllink(page)
    page == "api" && return "references/"
    return "references/" * replace(page, r"^api/" => "") * ".md"
end

function skillmarkdown(readme)
    text = read(readme, String)
    text = replace(text, r"^# CausalFrames\n" => "# CausalFrames.jl\n")
    text = replace(text, r"^\[!\[[^\n]*\n"m => "")
    text = replace(text, r"\n{3,}" => "\n\n")
    text = replace(
        text,
        "[DESIGN.md](DESIGN.md)" => "[DESIGN.md](https://github.com/farrellm/CausalFrames.jl/blob/master/DESIGN.md)",
    )
    text = replace(
        text,
        DOCS_ANCHOR_LINK => function (s)
            m = match(DOCS_ANCHOR_LINK, s)
            return "](" * skilllink(m[1]) * "#" * m[2] * ")"
        end,
    )
    text = replace(
        text,
        DOCS_PAGE_LINK => s -> "](" * skilllink(match(DOCS_PAGE_LINK, s)[1]) * ")",
    )
    text = replace(
        text,
        "\n## Concepts\n" => "\n" * SKILL_PREAMBLE * "## Concepts\n";
        count = 1,
    )
    return SKILL_FRONTMATTER * SKILL_GENERATED_NOTE * "\n" * plainmarkdown(text)
end

# The skill's files, relative path => contents.
function generateskill(docsdir)
    files = Dict{String,String}()
    files["SKILL.md"] = skillmarkdown(joinpath(docsdir, "..", "README.md"))
    apidir = joinpath(docsdir, "src", "api")
    for file in sort(readdir(apidir))
        (endswith(file, ".md") && file != "index.md") || continue
        files["references/"*file] = apireference(joinpath(apidir, file))
    end
    files["references/recipes.md"] =
        recipesreference(joinpath(docsdir, "src", "recipes.md"))
    return Dict(path => rstrip(text) * "\n" for (path, text) in files)
end

function skillfiles(dir)
    isdir(dir) || return String[]
    return [relpath(joinpath(root, f), dir) for (root, _, fs) in walkdir(dir) for f in fs]
end

# Outside CI, writes the skill (removing files it no longer has); in CI, fails
# if the committed skill differs from what the docs generate.
function writeskill(dir, docsdir)
    files = generateskill(docsdir)
    stale = sort!(
        union(
            [
                p for (p, text) in files if
                !isfile(joinpath(dir, p)) || read(joinpath(dir, p), String) != text
            ],
            setdiff(skillfiles(dir), keys(files)),
        ),
    )
    isempty(stale) && return nothing
    if get(ENV, "CI", "") == "true"
        error(
            "skills/causalframes is out of date with the docs; run " *
            "`julia --project=docs docs/make.jl` and commit the result. Stale files:\n" *
            join("  - " .* stale, "\n"),
        )
    end
    for p in setdiff(skillfiles(dir), keys(files))
        rm(joinpath(dir, p))
    end
    for (p, text) in files
        mkpath(dirname(joinpath(dir, p)))
        write(joinpath(dir, p), text)
    end
    return nothing
end
