using JDQMCMeasurements
using Documenter

DocMeta.setdocmeta!(JDQMCMeasurements, :DocTestSetup, :(using JDQMCMeasurements); recursive=true)

makedocs(;
    modules=[JDQMCMeasurements],
    authors="Benjamin Cohen-Stead <benwcs@gmail.com>",
    repo="https://github.com/SmoQySuite/JDQMCMeasurements.jl/blob/{commit}{path}#{line}",
    sitename="JDQMCMeasurements.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://SmoQySuite.github.io/JDQMCMeasurements.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/SmoQySuite/JDQMCMeasurements.jl",
    devbranch="master",
)
