#!/usr/bin/env julia
# Lint a job script before it costs a queue slot.
#
# Two traps have each reached the cluster and cost a round trip. Both are
# invisible to Meta.parse, so a script that parses can still die on rank 0
# and leave the other ranks blocked in a collective until mpirun kills
# them. The resulting stack traces point at UCX and InfiniBand, which is
# where two hours went.
#
#   1. Assigning at top level to a name that shadows an exported Base
#      binding: `all = ...` hits Base.all and raises
#      "cannot assign a value to imported variable".
#   2. Reassigning inside a top-level `for` or `try` a name bound outside
#      it: soft scope makes it a new local, so `worst = max(worst, e)`
#      raises "not defined in local scope".
#
# Usage: julia lint_job_script.jl <file.jl> ...

function lint(path::String)
    src = read(path, String)
    ex = Meta.parse("begin\n" * src * "\nend")
    problems = String[]

    # --- trap 1: shadowing an exported Base binding at top level ---
    function scan_base(e, depth)
        e isa Expr || return
        if e.head in (:function, :macro)
            return                      # a function body has its own scope
        end
        if e.head == :(=) && e.args[1] isa Symbol
            n = e.args[1]
            if isdefined(Base, n) && Base.isexported(Base, n)
                push!(problems, "shadows Base.$n at top level")
            end
        end
        foreach(a -> scan_base(a, depth + 1), e.args)
    end
    scan_base(ex, 0)

    # --- trap 2: soft-scope reassignment in a top-level for/try ---
    bound = Set{Symbol}()
    function collect_bound(e)
        e isa Expr || return
        e.head in (:function, :macro) && return
        e.head == :(=) && e.args[1] isa Symbol && push!(bound, e.args[1])
        foreach(collect_bound, e.args)
    end
    function scan_soft(e, inside::Bool, outer::Set{Symbol})
        e isa Expr || return
        if e.head in (:function, :macro)
            return
        elseif e.head in (:for, :try, :while)
            seen = copy(outer)
            for a in e.args
                scan_soft(a, true, seen)
            end
            return
        elseif inside && e.head == :(=) && e.args[1] isa Symbol
            n = e.args[1]
            if n in outer
                push!(problems, "soft-scope reassignment of `$n` inside a top-level block")
            end
        end
        foreach(a -> scan_soft(a, inside, outer), e.args)
    end
    # names bound at the very top level, outside any for/try
    for a in ex.args
        a isa Expr || continue
        a.head in (:for, :try, :while) && continue
        collect_bound(a)
    end
    for a in ex.args
        scan_soft(a, false, bound)
    end

    problems = unique(problems)
    if isempty(problems)
        println(rpad(basename(path), 34), " clean")
        return 0
    end
    println(rpad(basename(path), 34), " ", length(problems), " problem(s)")
    for p in problems
        println("    ", p)
    end
    return length(problems)
end

bad = 0
for f in ARGS
    global bad += lint(f)
end
exit(bad == 0 ? 0 : 1)
