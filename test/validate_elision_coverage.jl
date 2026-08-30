include(joinpath(@__DIR__, "..", "src", "STADE.jl"))

"""
    validate_elision_coverage(dir="val-corpus")

Check that every decision branch of `cgen_snapshot_save_dead` is reached by a
real corpus kernel, in at least one of `:adjoint` and `:hvp`.

That guard decides whether a snapshot save's read of an array counts as a
genuine consumer or is dead on arrival. It gates
`cgen_array_private_to_loop`, which gates `cgen_reduction_only_loop`, which
decides whether a loop reaches the GPU at all -- so a branch that quietly stops
being reachable is a silently untested path in the middle of the offload
decision. It has four outcomes:

  `:adjacent`     the overwrite is the next statement. What an adjoint emits.
  `:separated`    the overwrite is further along with nothing touching the array
                  in between. What an HVP emits, because `hvp_double_stmt`
                  interleaves each statement's shadow twin with its primal. This
                  branch did not exist until mpnn's HVP was found leaving its
                  loops over graph edges and graph nodes on the host.
  `:refused_intervening_touch`  something between reads or writes the array, so
                  the read is real and the loop must not split.
  `:refused_no_overwrite`       no covering overwrite at all.

The last is by far the most common, and that is expected rather than
suspicious: `cgen_is_snapshot_save` matches any `other[i] = arr[j]`, so every
ordinary array-to-array copy in the corpus lands here and is correctly left
alone. The count is printed for information; only reachability is asserted.

Rule 10 in skill-stade-dev asks that every shape an analysis can encounter be
represented in the corpus. This measures that directly instead of assuming it,
and fails if a future change makes a branch dead.

Needs no GPU.
"""
function validate_elision_coverage(dir::String = joinpath(@__DIR__, "val-corpus"))
    BRANCHES = (:adjacent, :separated, :refused_intervening_touch, :refused_no_overwrite)

    # Mirrors cgen_snapshot_save_dead's own walk, reporting WHICH branch answered
    # rather than just the boolean. Kept beside it deliberately: if the two ever
    # disagree the mirror is wrong, and the disagreement is the thing worth seeing.
    function classify(body, i, arr, idx)
        for j in (i + 1):length(body)
            STADE.cgen_overwrites_same_element(body[j], arr, idx) &&
                return j == i + 1 ? :adjacent : :separated
            STADE.cgen_stmt_touches_array(body[j], arr) && return :refused_intervening_touch
        end
        return :refused_no_overwrite
    end

    function walk!(body, tally, witness, label)
        arrays = Set{Symbol}()
        writes = Dict{Any,Vector{Any}}(); reads = Dict{Any,Vector{Any}}()
        STADE.cgen_collect_array_accesses!(body, writes, reads)
        union!(arrays, keys(writes)); union!(arrays, keys(reads))
        for arr in arrays, (i, st) in enumerate(body)
            STADE.cgen_is_snapshot_save(st, arr) || continue
            b = classify(body, i, arr, st.rhs.args[2])
            # the guard's real answer must agree with the mirror's
            @assert STADE.cgen_snapshot_save_dead(body, i, arr, st.rhs.args[2]) ==
                    (b in (:adjacent, :separated)) "mirror disagrees with cgen_snapshot_save_dead at $(label)/$(arr)"
            tally[b] = get(tally, b, 0) + 1
            haskey(witness, b) || (witness[b] = label)
        end
        for st in body
            st.kind == :for && walk!(st.body, tally, witness, label)
            st.kind == :if && (walk!(st.then, tally, witness, label); walk!(st.els, tally, witness, label))
        end
    end

    kernels = sort(filter(readdir(dir)) do f
        endswith(f, ".jl") && !any(endswith(f, s) for s in ("_b.jl", "_d.jl", "_hv.jl", "_cuda.jl", "_jacc.jl"))
    end)

    tally = Dict{Symbol,Int}()
    witness = Dict{Symbol,String}()
    for f in kernels
        name = splitext(f)[1]
        primal = STADE.io_read_corpus_entry(joinpath(dir, f))
        for mode in (:adjoint, :hvp)
            gen = mode == :hvp ? STADE.stade_hvp(primal; keep_push_pop = false) :
                                 STADE.stade_adjoint(primal; keep_push_pop = false)
            gk = STADE.cgen_ingest(mode == :hvp ? gen.hvp : gen.adjoint)
            walk!(gk.body, tally, witness, "$(name) [$(mode)]")
        end
    end

    bad = 0
    for b in BRANCHES
        n = get(tally, b, 0)
        if n == 0
            println(rpad(string(b), 30), " FAIL  unreachable from the corpus -- add a kernel with this shape")
            bad += 1
        else
            println(rpad(string(b), 30), " ok  ", lpad(n, 4), " sites, first: ", witness[b])
        end
    end
    println("\n", length(BRANCHES) - bad, "/", length(BRANCHES), " elision branches covered",
            bad == 0 ? "" : "   *** $bad NOT OK ***")
    return bad
end

validate_elision_coverage()
