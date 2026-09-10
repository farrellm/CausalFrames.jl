using CausalFrames
using DataFrames
using Documenter

DocMeta.setdocmeta!(CausalFrames, :DocTestSetup, :(using CausalFrames);
    recursive = true)

# The home page IS the README, generated here rather than kept as a second copy.
# The two were hand-maintained duplicates and had silently drifted several
# operators and a whole section apart; nothing enforced the sync, so the fix is
# to remove the opportunity rather than to re-sync. `docs/src/index.md` is
# generated and gitignored — edit README.md.
#
# Four rewrites separate the GitHub audience from the docs-site one: the docs
# title carries the .jl, the badges are GitHub furniture (one of them links to
# this very site), the DESIGN.md link is a repo path that would 404 here, and
# the Recipes link should stay inside the site rather than round-tripping
# through the deployed URL.
function readme_as_index(readme, index)
    text = read(readme, String)
    text = replace(text, r"^# CausalFrames\n" => "# CausalFrames.jl\n")
    text = replace(text, r"^\[!\[[^\n]*\n"m => "")
    text = replace(text, r"\n{3,}" => "\n\n")   # the blank run the badges left
    text = replace(
        text,
        "[DESIGN.md](DESIGN.md)" => "[DESIGN.md](https://github.com/farrellm/CausalFrames.jl/blob/master/DESIGN.md)",
    )
    text = replace(text,
        "[Recipes](https://farrellm.github.io/CausalFrames.jl/dev/recipes/)" => "[Recipes](recipes.md)",
    )
    return write(index, text)
end

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
        "API" => "api.md",
    ],
)

deploydocs(;
    repo = "github.com/farrellm/CausalFrames.jl",
    devbranch = "master",
)
