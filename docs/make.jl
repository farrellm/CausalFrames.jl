using CausalFrames
using DataFrames
using Documenter

DocMeta.setdocmeta!(CausalFrames, :DocTestSetup, :(using CausalFrames);
    recursive = true)

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
        "API" => "api.md",
    ],
)

deploydocs(;
    repo = "github.com/farrellm/CausalFrames.jl",
    devbranch = "master",
)
