include(joinpath(@__DIR__, "..", "src", "STADE.jl"))

"""
    validate_write_overlap()

Check that a GPU-split loop writing one array at two index expressions that can
land on the same element from two different threads emits atomic accumulations,
and that provably-disjoint write sets are left as plain writes.

This is the regression gate for the data race measured on a Tesla V100 on
2026-10-03, in the generated CUDA and JACC adjoints of `stencil_loss` and
`advection`. `cuda_kernel_stencil_loss_b_4!` contained

    ub[i_x - 1] = ub[i_x - 1] + __oldb_0
    ub[i_x]     = ub[i_x] + 2.0 * -__oldb_0
    ub[i_x + 1] = ub[i_x + 1] + __oldb_0

one thread per iteration. Each index is injective in the loop variable, which is
what `cgen_expr_injective_ok` proves and all that it proves: thread `i` and
thread `i+2` both update `ub[i+1]`, the plain read-modify-writes lose updates,
and the gradient comes back wrong and different on every run. The loss itself
stayed correct, so no value oracle saw it.

The numerical oracles cannot see this at all. `validate_corpus.jl` runs on the
host, where the loop is sequential; the corpus baseline for `stencil_loss` draws
`i_n = 4`, and even on a device every size up to one warp hides the race behind
the hardware's own lockstep. So this file checks the emitted code directly.

Three groups of checks:

  * `RACY` -- shapes whose write sites can collide across threads. Every write
    to the named array must be atomic, on both the CUDA and the JACC backend.
  * `DISJOINT` -- shapes whose write sites provably cannot collide. Every write
    to the named array must stay plain. These are the false-positive gate: an
    over-eager rule makes every scatter atomic, which is correct and slow, and
    nothing else in the suite would notice.
  * the corpus witnesses, `stencil_loss` and `advection`, generated through the
    real adjoint pipeline rather than hand-written.

It also sabotage-tests the guard: the same racy body is walked once with the
overlap set the analysis computed and once with it forced empty. A guard that
never fires produces identical code both ways, and that is a failure here.

No GPU is needed. `stade_gpu` is codegen, and CUDA is never loaded.
"""

# (name, source, array under test). Each kernel is a plain skill-stade kernel whose
# single loop cgen_ splits, so the body below is the device kernel body.
function val_overlap_racy_cases()
    return [
        ("stencil_3pt", """
         function stencil_3pt(a, b, i_n)
             for i_x = 2:i_n - 1
                 a[i_x - 1] = a[i_x - 1] + b[i_x]
                 a[i_x] = a[i_x] + 2.0 * b[i_x]
                 a[i_x + 1] = a[i_x + 1] + b[i_x]
             end
             return nothing
         end
         """, :a),
        ("shift_2pt", """
         function shift_2pt(a, b, i_n)
             for i_x = 2:i_n
                 a[i_x] = a[i_x] + b[i_x]
                 a[i_x - 1] = a[i_x - 1] + -b[i_x]
             end
             return nothing
         end
         """, :a),
        # the offset reaches the write index through a scalar let-binding, so the
        # comparison only works once same-body definitions are folded in
        ("hoisted_offset", """
         function hoisted_offset(a, b, i_n)
             for i_x = 2:i_n - 1
                 i_p = i_x + 1
                 a[i_x] = a[i_x] + b[i_x]
                 a[i_p] = a[i_p] + b[i_x]
             end
             return nothing
         end
         """, :a),
    ]
end

function val_overlap_disjoint_cases()
    return [
        # one site only -- the question this analysis asks does not arise
        ("single_site", """
         function single_site(a, b, i_n)
             for i_x = 1:i_n
                 a[i_x] = a[i_x] + b[i_x]
             end
             return nothing
         end
         """, :a),
        # same stride, offsets one apart, stride 2 -- no integer solution
        ("even_odd", """
         function even_odd(a, b, i_n)
             for i_x = 1:i_n
                 a[2 * i_x] = a[2 * i_x] + b[i_x]
                 a[2 * i_x + 1] = a[2 * i_x + 1] + b[i_x]
             end
             return nothing
         end
         """, :a),
        # the split loop here is the OUTER one, so its variable pins dimension 1 and two
        # threads never share a row however the inner index moves. Which loop cgen_ splits
        # decides the answer: the same body with the INNER loop split is safe for the other
        # reason, that each thread then owns one column.
        ("row_private_2d", """
         function row_private_2d(a, b, i_n, i_m)
             for i_x = 1:i_n
                 for i_c = 1:i_m - 1
                     a[i_x, i_c] = a[i_x, i_c] + b[i_x, i_c]
                     a[i_x, i_c + 1] = a[i_x, i_c + 1] + b[i_x, i_c]
                 end
             end
             return nothing
         end
         """, :a),
        # integer-literal components of one vector field can never alias
        ("components", """
         function components(a, b, i_n)
             for i_x = 1:i_n
                 a[1, i_x] = a[1, i_x] + b[i_x]
                 a[2, i_x] = a[2, i_x] + b[i_x]
             end
             return nothing
         end
         """, :a),
    ]
end


# Shapes checked against `cgen_race_write_arrays` directly, as (name, source, array, should_be_flagged).
# cgen_ is free to refuse a loop, or to split an inner loop rather than the outer one, long before
# cgen_device_assign is reached -- both are safe outcomes and neither exercises this analysis. These
# call it on the loop body it documents as its input, so the multi-dimensional and flattened-index
# paths are covered whatever the splitter decides.
function val_overlap_plan_cases()
    return [
        # colliding in the dimension carrying the loop variable; the MATCHING integer literal in
        # dimension 1 must not be read as disjointness
        ("col_2d", """
         function col_2d(a, b, i_n)
             for i_x = 2:i_n - 1
                 a[1, i_x - 1] = a[1, i_x - 1] + b[1, i_x]
                 a[1, i_x + 1] = a[1, i_x + 1] + b[1, i_x]
             end
             return nothing
         end
         """, :a, true),
        # a literal MISmatch in any one dimension is a real proof of disjointness
        ("components_2d", """
         function components_2d(a, b, i_n)
             for i_x = 1:i_n
                 a[1, i_x] = a[1, i_x] + b[i_x]
                 a[2, i_x] = a[2, i_x] + b[i_x]
             end
             return nothing
         end
         """, :a, false),
        # the loop variable pins dimension 1, so no two threads share a row
        ("row_2d", """
         function row_2d(a, b, i_n, i_m)
             for i_x = 1:i_n
                 for i_c = 1:i_m - 1
                     a[i_x, i_c] = a[i_x, i_c] + b[i_x, i_c]
                     a[i_x, i_c + 1] = a[i_x, i_c + 1] + b[i_x, i_c]
                 end
             end
             return nothing
         end
         """, :a, false),
        # same stride, offset not divisible by it
        ("stride2_2d", """
         function stride2_2d(a, b, i_n)
             for i_x = 1:i_n
                 a[2 * i_x, 1] = a[2 * i_x, 1] + b[i_x]
                 a[2 * i_x + 1, 1] = a[2 * i_x + 1, 1] + b[i_x]
             end
             return nothing
         end
         """, :a, false),
    ]
end

# every `arr[...] = ...` and `@atomic arr[...] += ...` anywhere in `expr`, as (atomic, index_expr)
function val_overlap_writes(expr, arr::Symbol, atomic_macros, out = Tuple{Bool,Any}[])
    if expr isa Expr
        if expr.head == :macrocall && !isempty(expr.args) && expr.args[1] in atomic_macros
            for a in expr.args[2:end]
                if a isa Expr && a.head == :(+=) && a.args[1] isa Expr &&
                   a.args[1].head == :ref && a.args[1].args[1] === arr
                    push!(out, (true, a.args[1]))
                end
            end
        elseif expr.head == :(=) && expr.args[1] isa Expr &&
               expr.args[1].head == :ref && expr.args[1].args[1] === arr
            push!(out, (false, expr.args[1]))
        end
        for a in expr.args
            val_overlap_writes(a, arr, atomic_macros, out)
        end
    end
    return out
end

function val_overlap_backends()
    cuda = STADE.cgen_backend_cuda()
    return [("cuda", e -> STADE.stade_gpu(e, cuda).kernels,
             Any[cuda.atomic_macro]),
            ("jacc", e -> STADE.stade_jacc(e).kernels,
             Any[Expr(:., :Atomix, QuoteNode(Symbol("@atomic")))])]
end

function validate_write_overlap()
    bad = 0
    checks = 0
    atomic_macros = Any[STADE.cgen_backend_cuda().atomic_macro,
                        Expr(:., :Atomix, QuoteNode(Symbol("@atomic")))]

    for (group, cases, want_atomic) in (("racy", val_overlap_racy_cases(), true),
                                        ("disjoint", val_overlap_disjoint_cases(), false))
        for (name, src, arr) in cases
            expr = Meta.parse(src)
            for (label, convert, _) in val_overlap_backends()
                checks += 1
                kernels = try
                    convert(expr)
                catch e
                    println(rpad("$(name) [$(group)/$(label)]", 38), " FAIL  conversion threw: ", e)
                    bad += 1
                    continue
                end
                writes = Tuple{Bool,Any}[]
                for k in kernels
                    val_overlap_writes(k, arr, atomic_macros, writes)
                end
                if isempty(writes)
                    println(rpad("$(name) [$(group)/$(label)]", 38),
                            " FAIL  no device write to `", arr, "` found -- the loop did not offload")
                    bad += 1
                elseif want_atomic && !all(first, writes)
                    plain = [w[2] for w in writes if !w[1]]
                    println(rpad("$(name) [$(group)/$(label)]", 38),
                            " FAIL  plain write at ", join(plain, ", "), " -- lost-update race")
                    bad += 1
                elseif !want_atomic && any(first, writes)
                    atom = [w[2] for w in writes if w[1]]
                    println(rpad("$(name) [$(group)/$(label)]", 38),
                            " FAIL  needless atomic at ", join(atom, ", "))
                    bad += 1
                else
                    println(rpad("$(name) [$(group)/$(label)]", 38),
                            " ok  ", length(writes), " write(s), atomic=", want_atomic)
                end
            end
        end
    end

    for (name, src, arr, want_flagged) in val_overlap_plan_cases()
        checks += 1
        kernel = STADE.parse_kernel(Meta.parse(src))
        loop = first(filter(s -> s.kind == :for, kernel.body))
        flagged = arr in STADE.cgen_race_write_arrays(loop.body, loop.var)
        if flagged != want_flagged
            println(rpad("$(name) [plan]", 38), " FAIL  flagged=", flagged, ", expected ", want_flagged)
            bad += 1
        else
            println(rpad("$(name) [plan]", 38), " ok  flagged=", flagged)
        end
    end

    # the two kernels the live run caught, through the real adjoint pipeline
    dir = joinpath(@__DIR__, "val-corpus")
    for (name, arr) in (("stencil_loss", :ub), ("advection", :ub))
        kernel = STADE.inl_inline_calls(STADE.io_read_kernel_corpus(joinpath(dir, name * ".jl")))[Symbol(name)]
        gen = STADE.stade_adjoint(kernel; keep_push_pop = false, fuse_ii_loops = true)
        for (label, convert, _) in val_overlap_backends()
            checks += 1
            writes = Tuple{Bool,Any}[]
            for k in convert(gen.adjoint)
                val_overlap_writes(k, arr, atomic_macros, writes)
            end
            plain = [w[2] for w in writes if !w[1]]
            if isempty(writes)
                println(rpad("$(name) [corpus/$(label)]", 38), " FAIL  no device write to `", arr, "`")
                bad += 1
            elseif !isempty(plain)
                println(rpad("$(name) [corpus/$(label)]", 38),
                        " FAIL  plain write at ", join(plain, ", "))
                bad += 1
            else
                println(rpad("$(name) [corpus/$(label)]", 38), " ok  ", length(writes), " atomic write(s)")
            end
        end
    end

    # Sabotage: walk the racy body again with the overlap set forced empty. If the
    # guard is doing nothing, the two walks agree and this check fires.
    checks += 1
    kernel = STADE.parse_kernel(Meta.parse(val_overlap_racy_cases()[1][2]))
    loop = first(filter(s -> s.kind == :for, kernel.body))
    backend = STADE.cgen_backend_cuda()
    guarded = STADE.cgen_device_body(loop.body, loop.var, backend, Set{Symbol}(),
                                     STADE.cgen_race_write_arrays(loop.body, loop.var))
    unguarded = STADE.cgen_device_body(loop.body, loop.var, backend, Set{Symbol}(), Set{Symbol}())
    if guarded == unguarded
        println(rpad("sabotage [guard fires]", 38),
                " FAIL  identical code with and without the overlap set -- the guard is vacuous")
        bad += 1
    elseif any(w -> w[1], val_overlap_writes(Expr(:block, unguarded...), :a, atomic_macros))
        println(rpad("sabotage [guard fires]", 38),
                " FAIL  the unguarded walk emitted an atomic, so the guard is not what produced it")
        bad += 1
    else
        println(rpad("sabotage [guard fires]", 38), " ok  guard is load-bearing")
    end

    println("\n", checks - bad, "/", checks, " checks passed",
            bad == 0 ? "" : "   *** $bad NOT OK ***")
    return bad
end

validate_write_overlap()
