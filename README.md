# STADE.jl 🏟️
*Low entry ticket to Scalable Differentiable Computing*

**Source Transformation Automatic Differentiation Engine** (STADE) generates source-verifiable GPU-ported (CUDA, AMDGPU, Metal, JACC)

- Adjoints
- Tangents
- Hessian-vector products (hvp)
- Primals

of monoprocessor numerical kernels written using the Julia programming language.

[Try STADE.jl out in our website!](https://cerfacs.github.io/STADE.jl)

## Why STADE.jl ?

This is a legitimate question since the Automatic Differentiation (AD) ecosystem in Julia is already rich (see [DifferentiationInterface.jl](https://github.com/JuliaDiff/DifferentiationInterface.jl)). In a nutshell, our motivations lean on obtaining:

- **Auditable** adjoints / tangents / hvp\
  STADE outputs human-readable Julia source code. This builds trust around AD tool usage to compute derivatives. It also eases debugging and development of new features.


- **Reproducible** gradient-based experiments (e.g., AI models training)\
  STADE-generated adjoints / tangents / hvp are bundled with primal source codes, such that numerical experiments are reproducible regardless of STADE availability or version in the machine they run.


- **Accessible** scalable differentiable computing\
  STADE generates GPU-ported adjoints / tangents / hvp from monoprocessor kernels, which are easy to craft and debug.


For now, the aforementioned goals come at the cost of supporting only a subset of the language abstractions. In any case, STADE focuses on differentiable computing, which itself is a subset of differentiable programming. [Here are defined STADE-differentiable kernels.](https://github.com/cerfacs/STADE.jl/blob/main/claude/skill-stade-kernels.md)

## First use


## Wishlist 💡

- [ ] Wrap common subexpressions into auxiliary variables
- [ ] Add `bgen_` stage for mini-batch execution code via GPU-aware MPI
- [ ] Add support to `:while` statement
- [ ] Add support to efficient differentiation of fixed-point loops 
- [ ] Add support to un-inlined call graphs
- [ ] Benchmark against mainstream frameworks (e.g., JAX, PyTorch)

### Acknowledgements

This project has been mainly inspired by [Inria Tapenade AD engine](https://team.inria.fr/ecuador/en/tapenade/) and received funding support from [ANITI EXPLEARTH](https://aniti.univ-toulouse.fr/en/explainable-and-physics-informed-ai-for-regional-weatherprediction/), [ROSAS Horizon Europe](https://www.rosas-project.eu/), and [PHLUSIM ANR](https://anr.fr/Project-ANR-23-CE23-0025).

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