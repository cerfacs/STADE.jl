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
  Two limits are worth stating. An MPI implementation chooses its own summation order, so a mini-batch gradient reduced with `stade_batch_file` is not bit-reproducible across rank counts. A 4-rank run on two nodes agreed with the single-rank result to one Float32 ulp.

  On a GPU, an atomic accumulation sums its contributing threads in whatever order the scheduler runs them, so a reduction below `reduction_threshold` is not bit-reproducible between runs either. Above the threshold the tree reduction is deterministic. Set `reduction_threshold = 0` for a deterministic reduction at every size, at the cost of the atomic path's speed on short loops.


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

  A loop whose whole body is one scalar accumulation, reading arrays indexed exactly by the loop variable, gets BOTH lowerings and a run-time test on its own trip count: below `reduction_threshold` iterations it runs the loop as it stands, as an atomic kernel when the accumulator is an array element and as a host loop when it is a local scalar; at or above the threshold it evaluates `target = target + mapreduce(f, +, view(a, lo:step:hi), ...)`. Neither wins everywhere. A tree reduction pays a fixed multi-pass cost a short loop cannot amortise, and an atomic on one address serialises its threads. On an NVIDIA A30, best-of-5 after warmup, `dotprod`'s adjoint crosses over near 24000: atomic 11.72 vs mapreduce 71.59 us at n=1024, 62.25 vs 72.12 us at n=16384, 876.21 vs 85.36 us at n=262144.

  The JACC backend applies the same test, choosing between its atomic kernel and `JACC.@parallel_reduce`. Measured on a Tesla V100, its crossover sits near 65000: atomic 735.31 vs reduce 1043.96 us at n=1024, 780.64 vs 726.00 us at n=65536, 1592.85 vs 641.37 us at n=262144. One shared default is used rather than one per backend, because that number and CUDA's bracket it.

  The same test governs a reduction into a local scalar, where the alternative is a host loop rather than an atomic kernel. An earlier version lowered that case unconditionally, on the reasoning that a device reduction must beat a host loop reading one scalar per element. Three consecutive same-process comparisons on a V100 at n = 16 say otherwise: the host loop at 1600.9, 1599.1 and 1524.6 us against 1820.5, 1872.0 and 1668.0 us.

  `reduction_threshold` defaults to 32768, chosen to avoid a regression rather than to capture every win. The same benchmark on a Tesla V100 does not reproduce: two identical runs gave 1160.74 us and 88.60 us for the reduction at n=262144, a 13x swing, while the atomic timings agreed within a few percent. No threshold can be fitted to that, so the default sits above the crossover of the one measurement that was stable. Measure your own hardware and lower it if that is worth doing. The view matches the loop's own range, so a loop that does not start at 1 reduces only the elements it visited, and an explicit `init` keeps a zero-trip loop contributing zero. `keep_all_atomic` was removed in v0.3.0. It selected between the two lowerings at generation time, which the run-time test now does per call; its two settings are reachable as `reduction_threshold = 0` and as any value above every trip count.

- `stade_batch_file(in_path::String, out_path::String; per_sample, mode, reduced = Symbol[])`: writes to `out_path` a Julia source epilogue that runs the kernel at `in_path` over a mini-batch, one sample per rank, with GPU-aware MPI. The kernel needs no batch loop and no rewriting: every rank runs it unchanged on its own sample, and the epilogue sums the parameter gradients across ranks. `per_sample` names the read-only arrays whose values change from one sample to the next; every other read-only array is treated as replicated. The generated file carries the derived role of every buffer as a comment, so the communication plan is auditable without re-running STADE.

## Multi-kernel files

A file may define several kernels. The one no other kernel calls is the entry
point, and every call in its body is inlined before differentiation.

A call argument must be a bare symbol when the parameter is an array, or when the
callee assigns to it, because the callee returns through it. A read-only scalar
parameter takes any expression: `stage(t, w, x, i_n - 1)` and `stage(t, w, x, 3)`
are both accepted, and the expression is bound to a temporary at the call site so
it is evaluated once per call.

Recursion is refused, directly or through a cycle. Inlining a recursive call
would need a depth that is not known before the code runs, so its tape has no
closed-form size. Write the recursion as a loop over an explicit level index, as
`mg_vcycle` does.

## Wishlist 💡

- [x] ✅ [v0.2.2] Wrap common subexpressions into auxiliary variables
- [x] ✅ [v0.2.4] Add `bgen_` stage for mini-batch execution code via GPU-aware MPI
- [ ] Replace reduction-related atomic writes with more performant alternatives\
  Partly done in v0.3.0. A whole-array reduction lowers to a view-bounded `mapreduce` instead of an atomic kernel above the trip-count threshold: 72.6 us against 132.5 us for `dotprod` on a Tesla V100, and 2.64x on the full `mlp1d` adjoint. That covers 29 of the corpus's 160 atomic writes, and only above the trip-count threshold. The remaining 131 are scatter-accumulates at a data-dependent index (`res[i_cell_to_node[e]] += ...`), multi-statement loop bodies, and reductions nested inside an offloaded outer loop, none of which a `mapreduce` over aligned views can express.

  Replacing every atomic is not the goal. An atomic is expensive only under contention: on a V100, one million threads accumulating into a single address cost 2400.78 us, into 4096 addresses 13.85 us. A scatter-accumulate spreads its writes and is already near-free, so the case worth replacing is exactly the single-target reduction, which is the case a tree reduction serves.
- [x] ✅ [v0.4.0] Add support to `:while` statement\
  Fixed in v0.4.3: activity did not propagate into a `while` body, so a body local was treated as passive at every use and the tangent came out silently zero. Any kernel whose `while` body assigns a scalar that then feeds an array must be regenerated.\
  The forward sweep records each loop's trip count and the backward sweep replays it as a `for`. The forward sweep still contains the `while`, since it has to run the loop to learn the count. The gradient is therefore taken at a fixed trip count, which is correct almost everywhere and wrong where the count jumps. A `while` runs on the host and needs `keep_push_pop = true`, since no launch can size itself around an unknown trip count and that mode has no closed-form stack size to offer; loops inside the body still offload. A condition that assigns, or that reads nothing the body writes, is refused at parse time.
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