using CausalFrames
using DataFrames
using Documenter
# Loaded up front so their CausalFrames extensions are compiled before the
# doctests that use them run; otherwise a first `using Parquet2` inside a doctest
# can print precompilation output into the result being compared. Imported, not
# used: MLJModelInterface's `Count` would shadow the summarizer in `@docs`.
import MLJModelInterface
import Parquet2

DocMeta.setdocmeta!(CausalFrames, :DocTestSetup,
    :(using CausalFrames, DataFrames, Dates);
    recursive = true)

# Links from the README into the deployed docs, with and without an anchor.
const DOCS_ANCHOR_LINK =
    r"\]\(https://farrellm\.github\.io/CausalFrames\.jl/dev/([\w/-]+)/#([^)\s]+)\)"
const DOCS_PAGE_LINK = r"\]\(https://farrellm\.github\.io/CausalFrames\.jl/dev/([\w/-]+)/\)"

# Every operator and summarizer gets its own `## `name`` header on one of the
# `docs/src/api/` pages; prose headers are for everything else. Returns the
# header name => the page it lives on, e.g. "readcsv" => "api/sources".
function operatorheaders(apidir)
    headers = Dict{String,String}()
    for file in sort(readdir(apidir))
        (endswith(file, ".md") && file != "index.md") || continue
        page = "api/" * file[1:(end-3)]
        for m in eachmatch(r"^## `([^`]+)`$"m, read(joinpath(apidir, file), String))
            headers[m[1]] = page
        end
    end
    return headers
end

# The README's operator and summarizer tables and the API overview must each
# link to every one of those headers; this check is the only thing keeping the
# hand-maintained tables in sync. It runs before the README is rewritten into
# the home page. The README links are absolute (as GitHub renders them), so the
# page must match too.
function checkapilinks(readme, apidir, overviewfile)
    headers = operatorheaders(apidir)
    overview = read(overviewfile, String)
    problems = String[]
    linked = Set{String}()
    for m in eachmatch(DOCS_ANCHOR_LINK, read(readme, String))
        page, anchor = m[1], m[2]
        if !haskey(headers, anchor)
            push!(
                problems,
                "README links to '#$anchor', which is not a header on any API page",
            )
        elseif headers[anchor] != page
            push!(problems,
                "README links to $page/#$anchor, but that header is on $(headers[anchor])")
        end
        push!(linked, anchor)
    end
    for (anchor, page) in sort!(collect(headers))
        anchor in linked || push!(problems, "no README link to $page/#$anchor")
        occursin("(@ref \"$anchor\")", overview) ||
            push!(problems, "no link to $page/#$anchor in api/index.md")
    end
    isempty(problems) ||
        error("API links out of sync:\n" * join("  - " .* problems, "\n"))
    return nothing
end

# The home page is the README, generated here so there is no second copy to
# drift. `docs/src/index.md` is generated and gitignored; edit README.md.
#
# Five rewrites separate the GitHub audience from the docs site: the title
# gains the .jl, the badges (GitHub furniture) go, the DESIGN.md link (a repo
# path that would 404) points at GitHub, links into the site (Recipes and
# the table links) become site-relative, and the examples become doctests.
# GitHub renders a `jldoctest` block as plain text, so the README writes them
# as `julia` blocks with a `# output` section, and they become one doctest
# session here, sharing their variables in order.
function readme_as_index(readme, index)
    text = read(readme, String)
    text = replace(text, r"^# CausalFrames\n" => "# CausalFrames.jl\n")
    text = replace(text, r"^\[!\[[^\n]*\n"m => "")
    text = replace(text, r"\n{3,}" => "\n\n")   # the blank run the badges left
    text = replace(
        text,
        "[DESIGN.md](DESIGN.md)" => "[DESIGN.md](https://github.com/farrellm/CausalFrames.jl/blob/master/DESIGN.md)",
    )
    text = replace(text, DOCS_ANCHOR_LINK => s"](\1.md#\2)")
    # The API overview is a directory index, not `api.md`; every other page maps
    # to its own file.
    text = replace(text,
        "](https://farrellm.github.io/CausalFrames.jl/dev/api/)" => "](api/index.md)")
    text = replace(text, DOCS_PAGE_LINK => s"](\1.md)")
    text = replace(text,
        r"```julia\n((?:(?!```)[\s\S])*?\n# output\n[\s\S]*?)```" =>
            s"```jldoctest readme\n\1```")
    return write(index, text)
end

checkapilinks(joinpath(@__DIR__, "..", "README.md"),
    joinpath(@__DIR__, "src", "api"),
    joinpath(@__DIR__, "src", "api", "index.md"))

readme_as_index(joinpath(@__DIR__, "..", "README.md"),
    joinpath(@__DIR__, "src", "index.md"))

makedocs(;
    modules = [CausalFrames],
    authors = "Matthew Farrell",
    sitename = "CausalFrames.jl",
    format = Documenter.HTML(;
        canonical = "https://farrellm.github.io/CausalFrames.jl",
        edit_link = "master",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Recipes" => "recipes.md",
        "API" => [
            "Overview" => "api/index.md",
            "Frames and pipelines" => "api/core.md",
            "Sources" => "api/sources.md",
            "File I/O" => "api/io.md",
            "Row transformations" => "api/rows.md",
            "Column transformations" => "api/columns.md",
            "Summarizing transforms" => "api/summarizing.md",
            "Model fitting (MLJ)" => "api/models.md",
            "Summarizers" => "api/summarizers.md",
        ],
    ],
)

deploydocs(;
    repo = "github.com/farrellm/CausalFrames.jl",
    devbranch = "master",
)
