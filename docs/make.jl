using CausalFrames
using DataFrames
using Documenter

DocMeta.setdocmeta!(CausalFrames, :DocTestSetup, :(using CausalFrames);
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
# link to every one of those headers. Nothing else enforces those tables — they
# are hand-maintained prose that used to drift silently — so the docs build
# does, before the README is rewritten into the home page. The README links are
# absolute (that is what GitHub renders), so the page has to match too.
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

# The home page IS the README, generated here rather than kept as a second copy.
# The two were hand-maintained duplicates and had silently drifted several
# operators and a whole section apart; nothing enforced the sync, so the fix is
# to remove the opportunity rather than to re-sync. `docs/src/index.md` is
# generated and gitignored — edit README.md.
#
# Four rewrites separate the GitHub audience from the docs-site one: the docs
# title carries the .jl, the badges are GitHub furniture (one of them links to
# this very site), the DESIGN.md link is a repo path that would 404 here, and
# the README's links into the deployed site — Recipes, and every operator and
# summarizer link in its tables — should stay inside the site rather than
# round-tripping through the deployed URL.
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
            "Sinks" => "api/sinks.md",
            "Row-wise transforms" => "api/rowwise.md",
            "Filling and truncation" => "api/filling.md",
            "Joins and time shifts" => "api/time.md",
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
