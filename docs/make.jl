using Documenter
using LinearOperators

makedocs(
  modules = [LinearOperators],
  assets = ["assets/style.css"],
  format = :html,
  sitename = "LinearOperators.jl",
  pages = Any["Home" => "index.md",
              "Tutorial" => "tutorial.md",
              "Reference" => "reference.md"]
)

deploydocs(deps = nothing, make = nothing,
  repo = "github.com/JuliaSmoothOptimizers/LinearOperators.jl.git",
  target = "build",
  julia = "0.6",
  latest = "master"
)
