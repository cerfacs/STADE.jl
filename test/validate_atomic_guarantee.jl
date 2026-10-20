include(joinpath(@__DIR__, "..", "src", "STADE.jl"))

# One occurrence of an array name inside an emitted kernel.
# kind: :atomic_acc (CUDA.@atomic A[e] += v), :plain_acc (A[e] += v, or A[e] = A[e] + .. with A absent elsewhere in the rhs),
#       :other (every other mention: a read, an overwrite, an index use)
# The thread variable of an emitted kernel: the scalar the kernel derives from `__tid`.
function aga_thread_var(kernel_body)
    for st in kernel_body.args
        st isa Expr && st.head == :(=) && st.args[1] isa Symbol && STADE.cgen_expr_contains(st.args[2], :__tid) &&
            st.args[2] != :__tid && return st.args[1]
    end
    return nothing
end

# Affine form `a * tv + b` of an index with literal integer arithmetic only, or nothing for anything else.
function aga_affine(e, tv)
    e isa Integer && return (0, Int(e))
    e === tv && return (1, 0)
    e isa Expr && e.head == :call || return nothing
    f = e.args[1]; xs = [aga_affine(a, tv) for a in e.args[2:end]]
    any(x -> x === nothing, xs) && return nothing
    f == :+ && return (sum(x[1] for x in xs), sum(x[2] for x in xs))
    f == :- && length(xs) == 1 && return (-xs[1][1], -xs[1][2])
    f == :- && length(xs) == 2 && return (xs[1][1] - xs[2][1], xs[1][2] - xs[2][2])
    if f == :* && length(xs) == 2
        xs[1][1] == 0 && return (xs[1][2] * xs[2][1], xs[1][2] * xs[2][2])
        xs[2][1] == 0 && return (xs[2][2] * xs[1][1], xs[2][2] * xs[1][2])
    end
    return nothing
end

# One-dimensional sites `a * tv + b_s` with one stride a and offsets in distinct residue classes mod a: no two (thread, site) pairs meet.
function aga_strided_disjoint(accs, tv)
    tv === nothing && return false
    all(s -> length(s.index) == 1, accs) || return false
    forms = [aga_affine(s.index[1], tv) for s in accs]
    any(f -> f === nothing, forms) && return false
    a = forms[1][1]
    (a != 0 && all(f -> f[1] == a, forms)) || return false
    offs = unique(f[2] for f in forms)
    return length(unique(mod(b, abs(a)) for b in offs)) == length(offs)
end

function aga_sites!(sites::Dict{Symbol,Vector{NamedTuple}}, e, am)
    if e isa Expr
        if e.head == :macrocall && length(e.args) >= 3 && e.args[1] == am &&
           e.args[3] isa Expr && e.args[3].head == :(+=) && e.args[3].args[1] isa Expr && e.args[3].args[1].head == :ref
            ref = e.args[3].args[1]
            push!(get!(sites, ref.args[1], NamedTuple[]), (kind = :atomic_acc, index = ref.args[2:end]))
            foreach(a -> aga_sites!(sites, a, am), ref.args[2:end])
            aga_sites!(sites, e.args[3].args[2], am)
            return
        end
        if e.head == :call && length(e.args) == 4 && e.args[1] === STADE.cgen_warp_add_name() && e.args[2] isa Symbol
            push!(get!(sites, e.args[2], NamedTuple[]), (kind = :atomic_acc, index = Any[e.args[3]]))
            aga_sites!(sites, e.args[3], am)
            aga_sites!(sites, e.args[4], am)
            return
        end
        if e.head == :(+=) && e.args[1] isa Expr && e.args[1].head == :ref && e.args[1].args[1] isa Symbol
            ref = e.args[1]
            push!(get!(sites, ref.args[1], NamedTuple[]), (kind = :plain_acc, index = ref.args[2:end]))
            foreach(a -> aga_sites!(sites, a, am), ref.args[2:end])
            aga_sites!(sites, e.args[2], am)
            return
        end
        if e.head == :(=) && e.args[1] isa Expr && e.args[1].head == :ref && e.args[1].args[1] isa Symbol
            ref = e.args[1]
            terms = STADE.cgen_flatten_sum(e.args[2])
            k = findfirst(t -> t == ref, terms)
            if k !== nothing && !any(j -> j != k && STADE.cgen_expr_reads_array(terms[j], ref.args[1]), eachindex(terms))
                push!(get!(sites, ref.args[1], NamedTuple[]), (kind = :plain_acc, index = ref.args[2:end]))
                foreach(a -> aga_sites!(sites, a, am), ref.args[2:end])
                for (j, t) in enumerate(terms)
                    j == k || aga_sites!(sites, t, am)
                end
                return
            end
        end
        if e.head == :ref && e.args[1] isa Symbol
            push!(get!(sites, e.args[1], NamedTuple[]), (kind = :other, index = e.args[2:end]))
            foreach(a -> aga_sites!(sites, a, am), e.args[2:end])
            return
        end
        foreach(a -> aga_sites!(sites, a, am), e.args)
    end
    return
end

"""
    validate_atomic_guarantee(dir="val-corpus"; only=nothing)

Reads the generated CUDA kernels of every corpus kernel (adjoint and HVP) and checks the atomic rule
the split gate rests on: an array that a kernel only ACCUMULATES into, at two or more different index
expressions, has no plain accumulation -- except an array whose every site carries the thread variable,
bare, at one index position (each thread owns a column). Two such sites can reach one element from two threads, so a plain
`A[e] = A[e] + v` there is a lost update; the write must be `CUDA.@atomic A[e] += v`.

This reads source text only. `validate_ownership.jl` finds the same defect by running; this test finds
it without any run, at any size, and names the array.
"""
function validate_atomic_guarantee(dir::String = joinpath(@__DIR__, "val-corpus");
                                   only::Union{Vector{String},Nothing} = nothing, verbose::Bool = true)
    names = sort([splitext(f)[1] for f in readdir(dir) if endswith(f, ".jl") &&
                  !any(endswith(f, s) for s in ("_b.jl", "_d.jl", "_hv.jl", "_cuda.jl", "_jacc.jl"))])
    only !== nothing && (names = filter(n -> n in only, names))
    backend = STADE.cgen_backend_cuda()
    kernels_seen = 0; arrays_checked = 0; bad = 0; errors = 0
    for name in names, mode in (:adjoint, :hvp)
        try
            pe = STADE.io_read_corpus_entry(joinpath(dir, name * ".jl"))
            gen = mode == :hvp ? STADE.stade_hvp(pe; keep_push_pop = false) : STADE.stade_adjoint(pe; keep_push_pop = false)
            plan = STADE.stade_gpu(mode == :hvp ? gen.hvp : gen.adjoint, backend)
            for k in plan.kernels
                kernels_seen += 1
                sites = Dict{Symbol,Vector{NamedTuple}}()
                aga_sites!(sites, k.args[2], backend.atomic_macro)
                tv = aga_thread_var(k.args[2])
                for (arr, ss) in sites
                    any(s -> s.kind == :other, ss) && continue
                    accs = filter(s -> s.kind != :other, ss)
                    length(unique(s.index for s in accs)) >= 2 || continue
                    # every site names the thread variable, bare, at one index position: each thread owns its own
                    # slice of the array (a column, say) and the sites cannot meet across threads
                    minlen = minimum(length(s.index) for s in accs)
                    tv !== nothing && any(p -> all(s -> s.index[p] == tv, accs), 1:minlen) && continue
                    # one literal stride, offsets in distinct residue classes (a conv's `(ci - 1) * 3 + k`): the sites tile the array
                    aga_strided_disjoint(accs, tv) && continue
                    arrays_checked += 1
                    plain = filter(s -> s.kind == :plain_acc, accs)
                    if !isempty(plain)
                        bad += 1
                        println(rpad("$(name) [$(mode)]", 34), " NOT ATOMIC  ", k.args[1].args[1], "  array ", arr, ": ",
                                length(plain), " plain accumulation(s) among ", length(accs), " sites, ",
                                length(unique(s.index for s in accs)), " different index expressions")
                    end
                end
            end
        catch e
            errors += 1
            println(rpad("$(name) [$(mode)]", 34), " ERROR ", first(split(sprint(showerror, e), "\n")))
        end
    end
    println("\n", kernels_seen, " kernels read, ", arrays_checked, " accumulate-only arrays with 2+ index expressions, ",
            bad, " with a plain accumulation, ", errors, " errors")
    return (kernels = kernels_seen, checked = arrays_checked, bad = bad, errors = errors)
end

# The strided exemption must reject colliding offsets and non-literal strides.
function aga_selftest()
    mk(ix...) = [(kind = :plain_acc, index = Any[Meta.parse(i)]) for i in ix]
    ok1 = aga_strided_disjoint(mk("(ci - 1) * 3 + 1", "(ci - 1) * 3 + 2", "(ci - 1) * 3 + 3"), :ci)
    ok2 = !aga_strided_disjoint(mk("(ci - 1) * 3 + 1", "(ci - 1) * 3 + 4"), :ci)
    ok3 = !aga_strided_disjoint(mk("(ci - 1) * hw + 1", "(ci - 1) * hw + 2"), :ci)
    ok4 = !aga_strided_disjoint(mk("ci + 1", "ci + 2"), :ci)
    ok5 = !aga_strided_disjoint(mk("2 * ci", "2 * ci + 2"), :ci)
    return ok1 && ok2 && ok3 && ok4 && ok5
end

if abspath(PROGRAM_FILE) == @__FILE__
    aga_selftest() || (println("strided exemption self-test FAILED"); exit(1))
    r = validate_atomic_guarantee(; only = length(ARGS) > 0 ? ARGS : nothing)
    exit(r.bad == 0 && r.errors == 0 && r.checked > 0 ? 0 : 1)
end
