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

if abspath(PROGRAM_FILE) == @__FILE__
    r = validate_atomic_guarantee(; only = length(ARGS) > 0 ? ARGS : nothing)
    exit(r.bad == 0 && r.errors == 0 && r.checked > 0 ? 0 : 1)
end
