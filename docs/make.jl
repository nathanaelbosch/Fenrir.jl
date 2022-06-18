using Fenrir
using Documenter

DocMeta.setdocmeta!(Fenrir, :DocTestSetup, :(using Fenrir); recursive=true)

makedocs(;
    modules=[Fenrir],
    authors="Nathanael Bosch <nathanael.bosch@uni-tuebingen.de> and contributors",
    repo="https://github.com/nathanaelbosch/Fenrir.jl/blob/{commit}{path}#{line}",
    sitename="Fenrir.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://nathanaelbosch.github.io/Fenrir.jl",
        assets=String[],
    ),
    pages=["Home" => "index.md", "Fenrr.jl in a nutshell" => "gettingstarted.md"],
)

deploydocs(; repo="github.com/nathanaelbosch/Fenrir.jl", devbranch="main")
