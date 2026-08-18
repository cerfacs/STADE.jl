# STADE.jl 🏟️
*Low entry ticket to Scalable Differentiable Computing*

**Source Transformation Automatic Differentiation Engine** (STADE) generates source-verifiable GPU-ported adjoints, tangents, Hessian-vector products, and primals of monoprocessor numerical kernels written using the Julia programming language.

[Try STADE.jl out in our website!](https://cerfacs.github.io/STADE.jl)

[Here](https://github.com/cerfacs/STADE.jl/blob/main/claude/skill-stade-kernels.md) are defined STADE-differentiable kernels.

## Why STADE.jl ?

This is a legitimate question since the Automatic Differentiation (AD) ecosystem in Julia is already rich (see [DifferentiationInterface.jl](https://github.com/JuliaDiff/DifferentiationInterface.jl)).

## First use


## Wishlist 💡

- [ ] Common subexpressions wrapping into temporary variables
- [ ] `bgen_` stage for mini-batch runs via GPU-aware MPI
- [ ] `:while` statement support
- [ ] (Binomial) checkpointing support
- [ ] Fixed-point loops efficient differentiation

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