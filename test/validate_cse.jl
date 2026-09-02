include(joinpath(@__DIR__, "..", "src", "STADE.jl"))

"""
    validate_cse(dir="val-corpus")

Check the two invariants `emit_cse_stmts` has to hold, on every generated tangent, adjoint and HVP
in the corpus.

**The kill invariant.** A temporary is only sound while every variable its defining expression reads
still holds the value it held at the definition. `emit_cse_run` enforces that by generation-keying:
an assignment bumps its target's generation, a push!/pop! bumps its stack's, and anything that is
not an assignment is a full barrier, so two textually identical occurrences separated by a write get
different keys and are never merged. If that bookkeeping ever breaks, the generated code still runs
and still looks reasonable -- it just silently uses a stale value. That is the class of defect this
file exists for, and no oracle in `validate_corpus.jl` reliably catches it: whether a stale read
changes the answer depends on the kernel.

The check is the direct one. For each `__cse_N = rhs`, walk forward through the enclosing run; if a
statement writes any variable `rhs` reads, every later read of `__cse_N` is a violation.

**The scope invariant.** A temporary must not be read outside the straight-line run that defined it.
`emit_cse_block` ends a run at every `:for` and `:if` and processes those bodies separately, so a
name can never be read across a loop header or out of a branch. A read with no definition ahead of
it in the same run is reported.

**The shadow invariant (HVP only).** `emit_cse_stmts` runs twice on an HVP body. The `__cse_*`
temporaries are bound before `hvp_double_body` and so must each be given a second-order shadow;
without one, `hvp_tangent_expr` resolves the name to the literal 0.0 and every second-order term
flowing through it is silently dropped. The `__hcse_*` temporaries are bound after doubling, when
nothing differentiates the body again, and must NOT have one. The check: if a `__cse_N`'s defining
expression reads anything that itself has a shadow in this body, `__cse_Nd` has to be defined too.
The escape clause is what keeps this sound -- a temporary whose inputs are all shadowless has a zero
tangent and correctly gets no shadow.

    cd test && julia validate_cse.jl
"""
function validate_cse(dir::String = joinpath(@__DIR__, "val-corpus"))
    kernels = sort(filter(f -> endswith(f, ".jl") &&
                               !(endswith(f, "_b.jl") || endswith(f, "_d.jl") || endswith(f, "_hv.jl")),
                          readdir(dir)))
    bad = 0
    checks = 0
    for f in kernels
        name = splitext(f)[1]
        primal = STADE.io_read_corpus_entry(joinpath(dir, f))
        for mode in (:tangent, :adjoint, :hvp)
            expr = mode == :hvp ? STADE.stade_hvp(primal; keep_push_pop = false, fuse_ii_loops = true).hvp :
                   mode == :adjoint ? STADE.stade_adjoint(primal; keep_push_pop = false, fuse_ii_loops = true).adjoint :
                   STADE.stade_tangent(primal)
            problems = String[]
            bound = Int[0]
            check_block!(expr.args[2].args, problems, bound)
            mode == :hvp && check_shadows!(expr, problems)
            checks += 1
            if isempty(problems)
                println(rpad("$(name) [$(mode)]", 32), " ok  ", lpad(bound[1], 4), " temporaries")
            else
                bad += 1
                println(rpad("$(name) [$(mode)]", 32), " FAIL  ", length(problems),
                        " violation(s), first: ", first(problems))
            end
        end
    end
    println("\n", checks - bad, "/", checks, " (kernel, mode) pairs hold both CSE invariants",
            bad == 0 ? "" : "   *** $bad NOT OK ***")
    return bad
end

is_temp(x) = x isa Symbol && startswith(string(x), "__cse_")

# Variables an expression reads: bare symbols, plus the name of any array it indexes. Mirrors
# emit_cse_reads! deliberately rather than calling it -- an independent copy is what makes this a
# check on the pass rather than a restatement of it.
function reads_of!(e, acc)
    e isa Symbol && (push!(acc, e); return acc)
    e isa Expr || return acc
    e.head == :ref && e.args[1] isa Symbol && push!(acc, e.args[1])
    start = (e.head == :call || e.head == :ref) ? 2 : 1
    for a in e.args[start:end]
        reads_of!(a, acc)
    end
    return acc
end

# Everything a statement can invalidate: its assignment target, any stack it pushes to or pops from,
# and -- for a statement that is not an assignment at all -- everything.
function writes_of(s)
    everything = Symbol[]
    acc = Set{Symbol}()
    stack_ops!(s, acc)
    if s isa Expr && s.head == :(=)
        lhs = s.args[1]
        lhs isa Symbol && push!(acc, lhs)
        lhs isa Expr && lhs.head == :ref && lhs.args[1] isa Symbol && push!(acc, lhs.args[1])
        return (acc, false)
    end
    return (acc, true)
end

function stack_ops!(e, acc)
    if e isa Expr
        e.head == :call && length(e.args) >= 2 && e.args[1] in (:push!, :pop!) &&
            e.args[2] isa Symbol && push!(acc, e.args[2])
        for a in e.args
            stack_ops!(a, acc)
        end
    end
    return nothing
end

function check_run!(run, problems, bound)
    defs = Dict{Symbol,Set{Symbol}}()     # live temp -> the variables its definition read
    stale = Set{Symbol}()
    for s in run
        used = Set{Symbol}()
        reads_of!(s isa Expr && s.head == :(=) ? s.args[2] : s, used)
        if s isa Expr && s.head == :(=) && s.args[1] isa Expr
            for a in s.args[1].args[2:end]; reads_of!(a, used); end
        end
        for v in used
            is_temp(v) || continue
            if v in stale
                push!(problems, "$(v) read after a write to something its definition reads")
            elseif !haskey(defs, v)
                push!(problems, "$(v) read with no definition ahead of it in this run")
            end
        end
        if s isa Expr && s.head == :(=) && is_temp(s.args[1])
            defs[s.args[1]] = reads_of!(s.args[2], Set{Symbol}())
            delete!(stale, s.args[1])
            bound[1] += 1
        end
        (written, barrier) = writes_of(s)
        for (v, srcs) in defs
            (barrier || any(w in srcs for w in written)) && push!(stale, v)
        end
    end
    return nothing
end

function check_block!(stmts, problems, bound)
    run = Any[]
    for s in stmts
        s isa LineNumberNode && continue
        if s isa Expr && s.head == :for
            check_run!(run, problems, bound); empty!(run)
            check_block!(s.args[2].args, problems, bound)
        elseif s isa Expr && s.head == :if
            check_run!(run, problems, bound); empty!(run)
            for a in s.args[2:end]
                a isa Expr && a.head == :block ? check_block!(a.args, problems, bound) :
                                                 check_block!(Any[a], problems, bound)
            end
        else
            push!(run, s)
        end
    end
    check_run!(run, problems, bound)
    return nothing
end

# Every name the body brings into existence: its parameters and everything it assigns to.
function defined_names(fexpr)
    acc = Set{Symbol}()
    for a in fexpr.args[1].args[2:end]
        a isa Symbol && push!(acc, a)
    end
    collect_targets!(fexpr.args[2], acc)
    return acc
end

function collect_targets!(e, acc)
    if e isa Expr
        if e.head == :(=)
            lhs = e.args[1]
            lhs isa Symbol && push!(acc, lhs)
            lhs isa Expr && lhs.head == :ref && lhs.args[1] isa Symbol && push!(acc, lhs.args[1])
        end
        for a in e.args
            collect_targets!(a, acc)
        end
    end
    return nothing
end

# tgen_shadow appends "d" to a variable; hvp_stack_shadow appends "_d" to a STACK. The two spellings
# are not interchangeable: accepting "_d" for an ordinary variable made this predicate report that
# transformer's `n` was differentiable, on the strength of an unrelated kernel scalar named `n_d`
# (which is n * d). That is a false positive, and a guard that fires for the wrong reason is worse
# than one that does not fire -- it invites a real defect to be explained away as noise.
has_shadow(v, defined) =
    endswith(string(v), "_stack") ? Symbol(string(v) * "_d") in defined :
                                    Symbol(string(v) * "d") in defined

# `__cse_0` is bound before doubling and needs a shadow; `__cse_0d` IS that shadow and must not be
# mistaken for another temporary owing one.
is_pre_double_temp(x) = x isa Symbol && occursin(r"^__cse_\d+$", string(x))

function check_shadows!(fexpr, problems)
    defined = defined_names(fexpr)
    defs = Dict{Symbol,Set{Symbol}}()
    collect_temp_defs!(fexpr.args[2], defs)
    for (v, srcs) in defs
        any(has_shadow(x, defined) for x in srcs) || continue
        has_shadow(v, defined) && continue
        push!(problems, "$(v) has a differentiable definition but no second-order shadow")
    end
    return nothing
end

function collect_temp_defs!(e, defs)
    if e isa Expr
        if e.head == :(=) && is_pre_double_temp(e.args[1])
            defs[e.args[1]] = reads_of!(e.args[2], Set{Symbol}())
        end
        for a in e.args
            collect_temp_defs!(a, defs)
        end
    end
    return nothing
end

validate_cse()
