using Documenter
using LinearOperators

makedocs(
  modules = [LinearOperators]
)

deploydocs(deps = Deps.pip("pygments", "mkdocs", "mkdocs-material", "python-markdown-math"),
  repo = "github.com/JuliaSmoothOptimizers/LinearOperators.jl.git",
  julia = "release",
  latest = "master"
)
