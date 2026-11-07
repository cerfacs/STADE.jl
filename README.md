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

  `stade_adjoint`, `stade_adjoint_file` and `stade_adjoint_corpus` take `primal_outputs::Bool = true`. With `false` the generated adjoint does not update the primal outputs of the kernel: it drops the forward sweep when the backward sweep does not read what the forward sweep writes (a linear update, such as a stencil step), and the stores into arrays that nothing in the adjoint reads. The arrays that the kernel writes then keep their input values, and the gradients are identical. A kernel with nothing to drop is generated as with `true` (the `_file` form prints a note). A store is kept when it reads a scalar that an inner loop accumulates beside an adjoint increment (the output layer of a perceptron): that sum is what lets the GPU stage split the gradient loop, and dropping it made the adjoint 2 to 3 times slower. On the GPU the result is usually faster (1.1 to 1.7 times on the stencil, scatter and solver kernels of the benchmark set). Time both settings.

- `stade_cuda/amdgpu/metal/jacc_file(in_path::String, out_path::String; ...)`: writes to `out_path` a Julia source result of GPU porting by STADE of the (multi-)kernel file at `in_path`.

  A loop whose whole body is one scalar accumulation, reading arrays indexed exactly by the loop variable, gets BOTH lowerings and a run-time test on its own trip count: below `reduction_threshold` iterations it runs the loop as it stands, as an atomic kernel when the accumulator is an array element and as a host loop when it is a local scalar; at or above the threshold it evaluates `target = target + mapreduce(f, +, view(a, lo:step:hi), ...)`. Neither wins everywhere. A tree reduction pays a fixed multi-pass cost a short loop cannot amortise, and an atomic on one address serialises its threads. On an NVIDIA A30, best-of-5 after warmup, `dotprod`'s adjoint crosses over near 24000: atomic 11.72 vs mapreduce 71.59 us at n=1024, 62.25 vs 72.12 us at n=16384, 876.21 vs 85.36 us at n=262144.

  The JACC backend applies the same test, choosing between its atomic kernel and `JACC.@parallel_reduce`. Measured on a Tesla V100, its crossover sits near 65000: atomic 735.31 vs reduce 1043.96 us at n=1024, 780.64 vs 726.00 us at n=65536, 1592.85 vs 641.37 us at n=262144. One shared default is used rather than one per backend, because that number and CUDA's bracket it.

  The same test governs a reduction into a local scalar, where the alternative is a host loop rather than an atomic kernel. An earlier version lowered that case unconditionally, on the reasoning that a device reduction must beat a host loop reading one scalar per element. Three consecutive same-process comparisons on a V100 at n = 16 say otherwise: the host loop at 1600.9, 1599.1 and 1524.6 us against 1820.5, 1872.0 and 1668.0 us.

  `reduction_threshold` defaults to 32768, chosen to avoid a regression rather than to capture every win. The same benchmark on a Tesla V100 does not reproduce: two identical runs gave 1160.74 us and 88.60 us for the reduction at n=262144, a 13x swing, while the atomic timings agreed within a few percent. No threshold can be fitted to that, so the default sits above the crossover of the one measurement that was stable. Measure your own hardware and lower it if that is worth doing. The view matches the loop's own range, so a loop that does not start at 1 reduces only the elements it visited, and an explicit `init` keeps a zero-trip loop contributing zero. `keep_all_atomic` was removed in v0.3.0. It selected between the two lowerings at generation time, which the run-time test now does per call; its two settings are reachable as `reduction_threshold = 0` and as any value above every trip count.

  Two CUDA-only keywords, both on by default, change how a reduction is lowered; `warp_atomics = false` and `sync_free_reduce = false` restore the previous output exactly. `warp_atomics`: an array-element atomic add `A[i] += v` becomes a call to a generated helper that, when all 32 lanes of the warp are active and agree on the index, adds the 32 values with a shuffle tree and lets lane 0 issue one atomic; in every other case it issues the plain atomic. The helper assumes one-dimensional thread blocks, which the generated launches use. It changes the order of the float summation, so results differ from the plain atomics in the last bits, and an accumulation into a single address (a loss) is much cheaper. `sync_free_reduce`: the large-trip branch of an array-element reduction becomes a device-side broadcast add into a one-element view (`view(A, i:i) .= op.(view(A, i:i), reduce(...; dims = 1))`) instead of a host `@allowscalar` assignment. That branch no longer copies a scalar to the host, so the host does not wait in the middle of the call sequence. AMDGPU, Metal and JACC output are unchanged. A CUDA graph cannot capture that large-trip branch: the dimension-wise reduction of CUDA.jl fails during capture, and the previous host form fails there too. To capture the whole adjoint call, pass `reduction_threshold = typemax(Int)`. Every reduction then uses the atomic kernel, which `warp_atomics` makes cheap. On a Tesla V100 with the U-Net adjoint, this setting gave the same step time as the default and a graph launch about 1% faster; the benefit is a lower launch cost per step, not a faster device.
  JACC kernels with more than 32 free variables take them as one tuple (`__jargs`) and unpack it on the device. JACC splats the launch arguments into the kernel, and Julia cannot compile a splat of more than 32 values on the device. Kernels with 32 or fewer free variables are unchanged. This lets a loop with a very wide body, such as the `ttgc` HVP cell loop (42 free variables), run on the device instead of staying on the host.

  CUDA only, opt-in: `fresh_gradient = true` writes a gradient element instead of accumulating into it (`g[i] = e` for `g[i] = g[i] + e`) when the adjoint array provably gets one write per element per call. It saves one read pass over the array, which is bandwidth-bound in a dense layer (on a V100 the gap of `mlp3` at 79M parameters to JAX goes from 1.41 to 1.12). The caller must pass the gradient arrays zeroed and make one adjoint call per reset; the CPU adjoint still accumulates, so the two contracts differ for a flagged array. A caller that breaks the contract gets wrong gradients and no error. `fresh_check = true` (default `false`, needs `fresh_gradient`) makes the generated host function raise an error at entry if a flagged gradient array holds a non-zero value or a NaN, which catches a missing reset and a second call without a reset. It costs one reduction and one synchronization per flagged array per call, so use it for testing and turn it off for timing. The header of the generated file lists the flagged arrays.
  CUDA only: `split_coalesce` (default `true`) gives the split-reduction twin a coalesced layout when its lane loops are unit-stride (transformer adjoint d2048, V100: 82.9 to 77.0 ms). CUDA only: `cublas = W` (default `2^26` from v0.4.62; `0` turns it off) runs an adjoint product as one `CUDA.CUBLAS.gemm!` when M * N * K >= W (same kernel, d2048: 76.9 to 62.3 ms; W = 2^28 is a safe value). With `cublas > 0` the scratch of a gemm product is an uninitialized array (beta = 0) and the seed scratch of an adjoint is not zeroed (d2048: 62.3 to 61.6 ms). The split twin is now also built for the adjoint of a loop with an empty loop or a reused temporary (`ufv_gather_step`; opt-in `split_reduce = 8, split_min_inner = 4, split_max_out = 10^8`: 1e6 cells 1.88 to 1.35 ms; slower for `mlp3` and `mpnn_gather`, so per kernel).

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
- [x] ✅ [v0.4.0] Add support to `:while` statement\
  Fixed in v0.4.3: activity did not propagate into a `while` body, so a body local was treated as passive at every use and the tangent came out silently zero. Any kernel whose `while` body assigns a scalar that then feeds an array must be regenerated.\
  The forward sweep records each loop's trip count and the backward sweep replays it as a `for`. The forward sweep still contains the `while`, since it has to run the loop to learn the count. The gradient is therefore taken at a fixed trip count, which is correct almost everywhere and wrong where the count jumps. A `while` runs on the host and needs `keep_push_pop = true`, since no launch can size itself around an unknown trip count and that mode has no closed-form stack size to offer; loops inside the body still offload. A condition that assigns, or that reads nothing the body writes, is refused at parse time.
- [ ] Add support to efficient differentiation of fixed-point loops 
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