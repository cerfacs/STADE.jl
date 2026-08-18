# STADE.jl 🏟️
*Low entry ticket to Scalable Differentiable Computing*

**Source Transformation Automatic Differentiation Engine** (STADE) generates source-verifiable GPU-ported (CUDA, AMDGPU, Metal, JACC)

- Adjoints
- Tangents
- Hessian-vector products (hvp)
- Primals

of monoprocessor numerical kernels written using the Julia programming language.

[Try STADE.jl out in our website!](https://cerfacs.github.io/STADE.jl)

[Here](https://github.com/cerfacs/STADE.jl/blob/main/claude/skill-stade-kernels.md) are defined STADE-differentiable kernels.

## Why STADE.jl ?

This is a legitimate question since the Automatic Differentiation (AD) ecosystem in Julia is already rich (see [DifferentiationInterface.jl](https://github.com/JuliaDiff/DifferentiationInterface.jl)).

1. Auditable adjoints / tangents / hvp

STADE outputs human-readable Julia source code.


2. Reproducible gradient-based experiments (e.g., AI models training)

STADE-generated adjoints / tangents / hvp are bundled with your primal source codes, such that your experiments are reproducible regardless of STADE availability or version in the machine they run.


3. Make differentiable computing as approachable as classic scientific computing




## First use


## Wishlist 💡

- [ ] Replace reduction-related atomic writes with more performant alternatives
- [ ] Wrap common subexpressions into temporary variables
- [ ] Add `bgen_` stage for mini-batch execution code via GPU-aware MPI
- [ ] Add support to `:while` statement
- [ ] Add support to (binomial) checkpointing
- [ ] Add support to efficient differentiation of fixed-point loops 
- [ ] Add support to un-inlined call graphs

### Acknowledgements

This project has been mainly inspired by [Inria Tapenade AD engine](https://team.inria.fr/ecuador/en/tapenade/).

#### If you use or build upon this software, please cite us:

```bibtex
@software{drozda_STADE,
  author       = {Drozda, Luciano},
  title        = {STADE.jl: Source Transformation Automatic Differentiation Engine},
  year         = 2026,
  publisher    = {Zenodo},
  doi          = {},
  url          = {},
}
```