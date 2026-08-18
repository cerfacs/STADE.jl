# STADE.jl 🏟️
*Low entry ticket to Scalable Differentiable Programming*

**Source Transformation Automatic Differentiation Engine** (STADE) generates GPU-ported adjoints, tangents, and Hessian-vector products from monoprocessor numerical kernels written using the Julia programming language.

[Try STADE.jl out in our website!](https://cerfacs.github.io/STADE.jl)

[Here](https://github.com/cerfacs/STADE.jl/blob/main/claude/skill-stade-kernels.md) are defined STADE-differentiable kernels.

## Why STADE.jl ?

This is a legitimate question as STADE.jl would stand as yet another Automatic Differentiation (AD) engine in the already rich Julia AD ecosystem (see DifferentiationInterface.jl).

## First use


## Wishlist 💡

- [ ] Common subexpressions wrapping into temporary variables
- [ ] `:while` statement support
- [ ] `bgen_` stage for mini-batch runs via GPU-aware MPI
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