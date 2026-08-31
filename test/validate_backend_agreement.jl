include(joinpath(@__DIR__, "..", "src", "STADE.jl"))

"""
    validate_backend_agreement(dir="val-corpus")

Check that the CUDA and JACC backends make the SAME offload decisions, loop for
loop, for every corpus kernel in both `:adjoint` and `:hvp`.

`cgen_body` and `jgen_body` share `cgen_reduction_only_loop` and every gate
around it. They differ only in how an eligible loop is *emitted* -- a `@cuda`
launch against a `JACC.@parallel_for`. So the two must agree on which loops
leave the host, how many device kernels come out, and what each kernel's
argument list is. Any divergence means one backend has a guard the other lacks.

That is not hypothetical. The first full-corpus live GPU run found three
defects in one pass, all of this exact shape:

  * `jgen_body` had no `cgen_liveout_is_zeroed` gate, so it offloaded a loop
    whose scalar a later host statement reads. `ii_kill` produced silently
    wrong gradients on a real GPU -- max_rel_err 0.85 adjoint, 1.23 HVP --
    while CUDA was exact. No crash. Only a numeric comparison caught it.
  * `jgen_body` did not wrap a host branch condition reading a device array in
    `@allowscalar`, which `cgen_body` does. `entry_branch` died with "Scalar
    indexing is disallowed" on JACC and ran fine on CUDA.
  * `jgen_launch_expr` had no zero-trip guard, which `cgen_launch_expr` has.
    `JACC.@parallel_for range=0` raises DivideError from `cld`.

All three are visible from the generated ASTs alone. This script needs no GPU,
no relay, and no waiting, and it would have found all three in seconds. Divergence
is reported as a failure in both directions: neither backend is the reference.

Agreement alone catches only the first of those three. The other two do not
change which loops are offloaded -- they change how the host code around them is
written -- so each gets its own property check below:

  * every `JACC.@parallel_for` must sit inside a `> 0` trip guard, since JACC has
    no per-thread bounds check to fall back on the way a CUDA kernel does;
  * every host `if` whose condition reads an array must have that condition
    wrapped in `@allowscalar`, in both backends.

One pair, `ttgc [hvp]`, is exempt from the structural comparison entirely: JACC
cannot take a kernel wider than the splat ceiling, `jgen_body` therefore holds
that loop on the host, and the resulting restructuring is not predictable enough
to assert a relation against. That pair is covered only by the ceiling check and
by numeric parity. It is the one blind spot here.

Both were verified to fail when the corresponding guard is removed from
`jgen_`, so all three defect shapes are now covered here.

A fourth check has nothing to do with agreement: JACC kernel ARITY. JACC's
`_parallel_for_cuda(N, f, x...)` splats its varargs into the user kernel, and
once that tuple passes Julia's inference splat limit the compiler emits a
dynamic `jl_f__apply_iterate` call, which cannot be compiled for a GPU. `ttgc`'s
HVP generates a 45-parameter kernel and dies with

    InvalidIRError: ... unsupported call to an unknown function
    (call to jl_f__apply_iterate)

while the identical CUDA kernel, which takes its arguments positionally rather
than through a splat, compiles fine. The limit was bracketed on hardware, not
derived: a 25-argument kernel compiles, a 44-argument one does not. 32 is
Julia's documented MAX_TUPLE_SPLAT and the most likely exact threshold, so it is
used as the ceiling here -- treat it as a well-placed estimate rather than a
measured boundary. This is a JACC/Julia limitation that STADE triggers by
emitting wide kernels, not a STADE codegen bug, but catching it here beats
discovering it on a device.

What this does NOT check is that the agreed decision is CORRECT. Both backends
sharing a wrong gate looks identical to both sharing a right one. Numeric parity
against the CPU (`validate_corpus_gpu.jl`, or a submitted
`stade_validate_cuda_file` script) remains the only thing that can tell those
apart.
"""
function validate_backend_agreement(dir::String = joinpath(@__DIR__, "val-corpus"))
    count_for(e) = e isa Expr ? (e.head === :for) + sum(Int[count_for(a) for a in e.args]; init = 0) : 0

    # loop signature: the (lo, step, hi) of every loop the host still runs, in order.
    # Comparing shapes rather than counts catches a backend that keeps the RIGHT NUMBER
    # of loops on the host but a different set of them.
    function host_loop_shapes(e, acc = Any[])
        if e isa Expr
            if e.head === :for && length(e.args) >= 1 && e.args[1] isa Expr && e.args[1].head === :(=)
                push!(acc, string(e.args[1].args[2]))
            end
            foreach(a -> host_loop_shapes(a, acc), e.args)
        end
        return acc
    end
    kernel_arity(k) = length(k.args[1].args) - 1

    kernels = sort(filter(readdir(dir)) do f
        endswith(f, ".jl") && !any(endswith(f, s) for s in ("_b.jl", "_d.jl", "_hv.jl", "_cuda.jl", "_jacc.jl"))
    end)

    # every JACC launch must be inside a `<trip> > 0` guard
    function unguarded_launches(e, guarded = false, acc = Any[])
        if e isa Expr
            if e.head === :macrocall && any(a -> a === Symbol("@parallel_for") ||
                   (a isa QuoteNode && a.value === Symbol("@parallel_for")) ||
                   (a isa Expr && a.head === :. && a.args[2] isa QuoteNode &&
                    a.args[2].value === Symbol("@parallel_for")), e.args)
                guarded || push!(acc, e)
                return acc
            end
            if e.head === :if && e.args[1] isa Expr && e.args[1].head === :call && e.args[1].args[1] === :>
                unguarded_launches(e.args[2], true, acc)
                length(e.args) > 2 && unguarded_launches(e.args[3], guarded, acc)
                return acc
            end
            foreach(a -> unguarded_launches(a, guarded, acc), e.args)
        end
        return acc
    end

    # every host `if` reading an array must wrap its condition in @allowscalar
    function bare_array_conditions(e, acc = Any[])
        if e isa Expr
            if e.head === :if && STADE.cgen_expr_has_ref(e.args[1]) &&
               !(e.args[1] isa Expr && e.args[1].head === :macrocall)
                push!(acc, e.args[1])
            end
            foreach(a -> bare_array_conditions(a, acc), e.args)
        end
        return acc
    end

    bad = 0
    checks = 0
    for f in kernels
        name = splitext(f)[1]
        primal = STADE.io_read_corpus_entry(joinpath(dir, f))
        for mode in (:adjoint, :hvp)
            gen = mode == :hvp ? STADE.stade_hvp(primal; keep_push_pop = false, fuse_ii_loops = true) :
                                 STADE.stade_adjoint(primal; keep_push_pop = false, fuse_ii_loops = true)
            expr = mode == :hvp ? gen.hvp : gen.adjoint
            cu = STADE.stade_gpu(expr, STADE.cgen_backend_cuda())
            ja = STADE.stade_jacc(expr)

            problems = String[]
            # A CUDA kernel wider than the splat ceiling is one JACC legitimately cannot
            # take, so the two backends are EXPECTED to diverge by exactly that many
            # loops -- a real capability gap, not a missing guard. Any divergence beyond
            # it is unexplained and fails.
            over = count(k -> kernel_arity(k) > STADE.JGEN_MAX_SPLAT_ARGS, cu.kernels)
            cu_shapes = host_loop_shapes(cu.host)
            ja_shapes = host_loop_shapes(ja.host)
            if over == 0
                cu_shapes == ja_shapes ||
                    push!(problems, "host loops differ: cuda=$(length(cu_shapes)) jacc=$(length(ja_shapes))" *
                                    (length(cu_shapes) == length(ja_shapes) ? " (same count, different loops)" : ""))
                length(cu.kernels) == length(ja.kernels) ||
                    push!(problems, "device kernel count: cuda=$(length(cu.kernels)) jacc=$(length(ja.kernels))")
                if length(cu.kernels) == length(ja.kernels)
                    # JACC kernels take a leading index argument; CUDA derives it from the thread id
                    for (i, (kc, kj)) in enumerate(zip(cu.kernels, ja.kernels))
                        kernel_arity(kc) + 1 == kernel_arity(kj) ||
                            push!(problems, "kernel $(i) arity: cuda=$(kernel_arity(kc)) jacc=$(kernel_arity(kj))")
                    end
                end
            else
                # Structural comparison is abandoned here, not merely relaxed. Holding a
                # wide outer loop on the host does not subtract one kernel: its inner
                # loops then become individually splittable, so JACC ends up with MORE
                # kernels AND more host loops than CUDA (ttgc [hvp]: 23 vs 22 kernels,
                # 9 vs 4 host loops). The restructuring is not cheaply predictable, so
                # no arithmetic relation is asserted. This pair is therefore covered
                # only by the ceiling check above and by numeric parity elsewhere --
                # a real, and currently the only, blind spot in this script.
            end

            # Every JACC kernel must be under the splat ceiling; if one is not, the
            # gate in jgen_body failed and the kernel will not compile on a device.
            for (i, kj) in enumerate(ja.kernels)
                n = kernel_arity(kj) - 1   # minus the leading index argument JACC supplies
                n <= STADE.JGEN_MAX_SPLAT_ARGS ||
                    push!(problems, "jacc kernel $(i) takes $(n) splatted args (ceiling $(STADE.JGEN_MAX_SPLAT_ARGS)) -- the jgen_body gate let it through")
            end

            ug = unguarded_launches(ja.host)
            isempty(ug) || push!(problems, "$(length(ug)) JACC launch(es) with no zero-trip guard")
            for (lbl, h) in (("cuda", cu.host), ("jacc", ja.host))
                bc = bare_array_conditions(h)
                isempty(bc) || push!(problems, "$(lbl): $(length(bc)) host if-condition(s) read an array unwrapped")
            end

            checks += 1
            label = rpad("$(name) [$(mode)]", 34)
            if isempty(problems)
                println(label, " ok  host=", length(cu_shapes), " kernels=", length(cu.kernels),
                        over == 0 ? "" : "   (+$(over) loop(s) held on host by JACC's splat ceiling)")
            else
                println(label, " FAIL")
                for p in problems
                    println("      ", p)
                end
                bad += 1
            end
        end
    end

    println("\n", checks - bad, "/", checks, " (kernel, mode) pairs agree across backends",
            bad == 0 ? "" : "   *** $bad NOT OK ***")
    return bad
end

validate_backend_agreement()
