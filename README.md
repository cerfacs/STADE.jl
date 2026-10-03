# STADE.jl 🏟️

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22061808.svg)](https://doi.org/10.5281/zenodo.22061808)

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
  STADE-generated adjoints / tangents / hvp are bundled with primal source codes, such that numerical experiments are reproducible regardless of STADE availability or version in the machine they run.\
  One limit is worth stating: an MPI implementation chooses its own summation order, so a mini-batch gradient reduced with `stade_batch_file` is not bit-reproducible across rank counts. A 4-rank run on two nodes agreed with the single-rank result to one Float32 ulp.


- **Accessible** scalable differentiable computing\
  STADE generates GPU-ported adjoints / tangents / hvp from monoprocessor kernels, which are easy to craft and debug.


For now, the aforementioned goals come at the cost of supporting only a subset of the language abstractions. In any case, STADE focuses on differentiable computing, which itself is a subset of differentiable programming. [Here are defined STADE-differentiable kernels.](https://github.com/cerfacs/STADE.jl/blob/main/claude/skill-stade.md)

## First use

```bash
julia -e ' \
begin
  import Pkg
  Pkg.add(url="https://github.com/cerfacs/STADE.jl")
  using STADE
  in_    = pkgdir(STADE, "test", "val-corpus", "affine_loss.jl")
  outdir = mktempdir()
  out_b  = joinpath(outdir, "affine_loss_b.jl")
  out_c  = joinpath(outdir, "affine_loss_b_cuda.jl")
  try
    stade_adjoint_file(in_, out_b)
    stade_cuda_file(out_b, out_c)
    println(read(out_b, String))
    println(read(out_c, String))
  catch e
    error("STADE.jl failed: $e")
  end
end
'
```

## Public API

- `stade_tangent/adjoint/hvp_file(in_path::String, out_path::String; ...)`: writes to `out_path` a Julia source result of differentiation by STADE of the (multi-)kernel file at `in_path`. In the multi-kernel case, inlining is performed in the kernel which is not called by any other one in the file (named root kernel). The inlined root kernel is the one subject to differentiation.

- `stade_cuda/amdgpu/metal/jacc_file(in_path::String, out_path::String; ...)`: writes to `out_path` a Julia source result of GPU porting by STADE of the (multi-)kernel file at `in_path`.

  `keep_all_atomic = false` is the default. A loop whose whole body is one scalar accumulation, reading arrays indexed exactly by the loop variable, gets BOTH lowerings and a run-time test on its own trip count: below 1024 iterations it launches the atomic kernel, at or above it evaluates `target = target + mapreduce(f, +, view(a, lo:step:hi), ...)`. Neither wins everywhere on a Tesla V100, measured on `dotprod`'s adjoint: 8.66 vs 53.73 us at n=16, 164.18 vs 72.11 us at n=4096, 2506.31 vs 150.12 us at n=1048576. A tree reduction pays a fixed multi-pass cost a short loop cannot amortise, and an atomic on one address serialises its threads. The view matches the loop's own range, so a loop that does not start at 1 reduces only the elements it visited, and an explicit `init` keeps a zero-trip loop contributing zero. Pass `keep_all_atomic = true` to emit the atomic kernel unconditionally.

- `stade_batch_file(in_path::String, out_path::String; per_sample, mode, reduced = Symbol[])`: writes to `out_path` a Julia source epilogue that runs the kernel at `in_path` over a mini-batch, one sample per rank, with GPU-aware MPI. The kernel needs no batch loop and no rewriting: every rank runs it unchanged on its own sample, and the epilogue sums the parameter gradients across ranks. `per_sample` names the read-only arrays whose values change from one sample to the next; every other read-only array is treated as replicated. The generated file carries the derived role of every buffer as a comment, so the communication plan is auditable without re-running STADE.

## Wishlist 💡

- [x] ✅ [v0.2.2] Wrap common subexpressions into auxiliary variables
- [x] ✅ [v0.2.4] Add `bgen_` stage for mini-batch execution code via GPU-aware MPI
- [ ] Replace reduction-related atomic writes with more performant alternatives\
  Partly done in v0.2.6. `keep_all_atomic` now defaults to `false`, so a whole-array reduction lowers to a view-bounded `mapreduce` instead of an atomic kernel: 72.6 us against 132.5 us for `dotprod` on a Tesla V100, and 2.64x on the full `mlp1d` adjoint. That covers 29 of the corpus's 160 atomic writes, and only above the trip-count threshold. The remaining 131 are scatter-accumulates at a data-dependent index (`res[i_cell_to_node[e]] += ...`), multi-statement loop bodies, and reductions nested inside an offloaded outer loop, none of which a `mapreduce` over aligned views can express.

  Replacing every atomic is not the goal. An atomic is expensive only under contention: on a V100, one million threads accumulating into a single address cost 2400.78 us, into 4096 addresses 13.85 us. A scatter-accumulate spreads its writes and is already near-free, so the case worth replacing is exactly the single-target reduction, which is the case a tree reduction serves.
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
  doi          = {https://doi.org/10.5281/zenodo.22061808},
  url          = {https://github.com/cerfacs/STADE.jl},
}
```