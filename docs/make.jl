using Documenter, LinearOperators

makedocs(
  modules = [LinearOperators],
  doctest = true,
  linkcheck = true,
  format = Documenter.HTML(
    assets = ["assets/style.css"],
    ansicolor = true,
    prettyurls = get(ENV, "CI", nothing) == "true",
  ),
  sitename = "LinearOperators.jl",
  pages = Any["Home" => "index.md", "Tutorial" => "tutorial.md", "Reference" => "reference.md"],
)

deploydocs(
  repo = "github.com/JuliaSmoothOptimizers/LinearOperators.jl.git",
  push_preview = true,
  devbranch = "main",
)
