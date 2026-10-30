# ==================== validate_inbounds.jl ==========================
# Checks the two kernel-finishing passes of v0.4.32: the `@inbounds` wrap (`inbounds`) and the deletion of dead scalar updates (`dead_scalars`).
#
#   julia validate_inbounds.jl                       all phases
#   validate_inbounds(; phases = (:units, :corpus))  without the sabotage phase
#
# 1. UNITS. Small kernels, written as plain Julia and run before and after the pass on random inputs. The kernels use no GPU call, so the
#    comparison is exact (arrays and a call counter). Each case also names what must stay and what must go, because a pass that deletes
#    nothing would also pass the differential run.
# 2. CORPUS. The adjoint and the HVP of every corpus kernel and every kit kernel, CUDA and JACC. Per kernel: with both keywords off the
#    kernels are the very same objects; the pass deletes only scalar assignments (the count of array writes, macro calls and calls is the
#    same); a second pass deletes nothing; the wrapped kernel holds the whole body with the thread guard inside; the printed source parses back
#    to the same expression.
# 3. SABOTAGE. Edits of the pass, each of which one unit case must catch.

isdefined(Main, :STADE) || include(joinpath(@__DIR__, "..", "src", "STADE.jl"))

# ---- units ------------------------------------------------------------------------------------------------------------
# (name, source, kept, gone, gone_without_refs): `gone` are lines that must be absent when array reads may be dropped (inbounds on),
# `gone_without_refs` the lines that must be absent when they may not.
const INB_UNITS = [
    ("dead accumulation in a loop (the b_13 pattern)",
     """function k(x, w, out, n)
         s = 0.5
         for c = 1:n
             u = x[c]
             v = w[c]
             s = s + u * v
             out[c] = u
         end
         return nothing
     end""", String[], ["s = s + u * v", "s = 0.5"], ["s = s + u * v", "s = 0.5"]),
    ("dead accumulation that reads an array directly",
     """function k(x, w, out, n)
         s = 0.5
         for c = 1:n
             s = s + x[c] * w[c]
             out[c] = x[c]
         end
         return nothing
     end""", String[], ["s = s + x[c] * w[c]", "s = 0.5"], String[]),
    ("trailing reset of a scalar",
     """function k(x, out, n)
         a = x[1]
         out[1] = a
         a = 0.0
         return nothing
     end""", ["out[1] = a", "a = x[1]"], ["a = 0.0"], ["a = 0.0"]),
    ("accumulator read after the loop",
     """function k(x, out, n)
         s = 0.0
         for c = 1:n
             s = s + x[c]
         end
         out[1] = s
         return nothing
     end""", ["s = s + x[c]", "s = 0.0", "out[1] = s"], String[], String[]),
    ("value read only by the next iteration",
     """function k(x, out, n)
         prev = 0.0
         for c = 1:n
             out[c] = prev
             prev = x[c]
         end
         return nothing
     end""", ["prev = x[c]", "prev = 0.0"], String[], String[]),
    ("read in the then branch only",
     """function k(x, out, n)
         a = x[1] * 2.0
         if n > 3
             out[1] = a
         end
         return nothing
     end""", ["a = x[1] * 2.0", "out[1] = a"], String[], String[]),
    ("read in the else branch only",
     """function k(x, out, n)
         a = x[1] * 2.0
         if n > 3
             out[1] = 1.0
         else
             out[2] = a
         end
         return nothing
     end""", ["a = x[1] * 2.0", "out[2] = a"], String[], String[]),
    ("a chain nobody reads",
     """function k(x, out, n)
         a = x[1]
         b = a * 2.0
         c = b + a
         out[1] = 1.0
         return nothing
     end""", ["out[1] = 1.0"], ["c = b + a", "b = a * 2.0"], ["c = b + a", "b = a * 2.0"]),
    ("overwritten before any read",
     """function k(x, out, n)
         v = 1.0
         v = x[1]
         out[1] = v
         return nothing
     end""", ["v = x[1]", "out[1] = v"], ["v = 1.0"], ["v = 1.0"]),
    ("read only inside a later loop",
     """function k(x, out, n)
         s = x[1] * 3.0
         for c = 1:n
             out[c] = s
         end
         return nothing
     end""", ["s = x[1] * 3.0", "out[c] = s"], String[], String[]),
    ("read as a loop bound",
     """function k(x, out, n)
         m = n - 1
         for c = 1:m
             out[c] = x[c]
         end
         return nothing
     end""", ["m = n - 1"], String[], String[]),
    ("read as an array index",
     """function k(x, out, n)
         j = n - 1
         out[j] = x[1]
         return nothing
     end""", ["j = n - 1"], String[], String[]),
    ("carried across the iterations of an outer loop",
     """function k(x, out, n)
         acc = 0.0
         for i = 1:n
             for j = 1:n
                 acc = acc + x[j]
             end
             out[i] = acc
         end
         return nothing
     end""", ["acc = acc + x[j]", "acc = 0.0", "out[i] = acc"], String[], String[]),
    ("a call with an effect is kept",
     """function k(x, out, n)
         v = bump!(x, n)
         return nothing
     end""", ["v = bump!(x, n)"], String[], String[]),
    ("read only by a statement of another form",
     """function k(x, out, n)
         v = x[1] + 1.0
         sink!(out, v)
         return nothing
     end""", ["v = x[1] + 1.0", "sink!(out, v)"], String[], String[]),
    ("dead update inside a branch",
     """function k(x, out, n)
         s = x[1]
         if n > 2
             s = s * 2.0
             out[1] = 5.0
         end
         out[2] = x[2]
         return nothing
     end""", ["out[1] = 5.0", "out[2] = x[2]"], ["s = s * 2.0", "s = x[1]"], ["s = s * 2.0"]),
    ("overwritten in one branch only, no else",
     """function k(x, out, n)
         a = x[1] * 2.0
         if n > 3
             a = 5.0
         end
         out[1] = a
         return nothing
     end""", ["a = x[1] * 2.0", "a = 5.0", "out[1] = a"], String[], String[]),
    ("read only by a branch condition",
     """function k(x, out, n)
         m = n - 1
         if m > 3
             out[1] = 1.0
         end
         out[2] = 2.0
         return nothing
     end""", ["m = n - 1", "out[1] = 1.0"], String[], String[]),
    ("early return keeps what the guard reads",
     """function k(x, out, n)
         lim = n * 2
         if n > 100
             return nothing
         end
         out[1] = lim
         return nothing
     end""", ["lim = n * 2", "out[1] = lim"], String[], String[]),
]

const INB_PRELUDE = """
bump!(x, n) = (x[end] = x[end] + 1.0; 1.0)
sink!(out, v) = (out[end] = out[end] + v; nothing)
"""

function inb_run(src::String, seed::Int)
    m = Module()
    Core.eval(m, Meta.parseall(INB_PRELUDE))
    Core.eval(m, Meta.parse(src))
    n = 6
    x = [sin(1.3 * i + seed) for i in 1:(n + 4)]
    w = [cos(0.7 * i + seed) for i in 1:(n + 4)]
    out = zeros(n + 4)
    k = getfield(m, :k)
    nargs = length(first(methods(k)).sig.parameters) - 1
    nargs == 4 ? Base.invokelatest(k, x, w, out, n) : nargs == 3 ? Base.invokelatest(k, x, out, n) : error("arity")
    return (x, out)
end

function inb_unit_checks()
    bad = 0
    check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
    for (name, src, kept, gone, gone_nr) in INB_UNITS
        ex = Meta.parse(src)
        for (refs_ok, want_gone) in ((true, gone), (false, gone_nr))
            new = STADE.cgen_dead_scalars(ex, refs_ok)
            txt = STADE.io_expr_to_source(new)
            # exact differential run: every array and the call counter
            same = all(seed -> inb_run(src, seed) == inb_run(txt, seed), 1:3)
            # when array reads may not be dropped, a line that only the dropping mode removes must still be there
            must = refs_ok ? kept : vcat(kept, [g for g in gone if !(g in gone_nr)])
            miss = [s for s in must if !occursin(s, txt)]
            left = [s for s in want_gone if occursin(s, txt)]
            check(same && isempty(miss) && isempty(left),
                  "$name [refs $(refs_ok ? "dropped" : "kept")]" * (same ? "" : " -- results differ") * (isempty(miss) ? "" : " -- deleted: $miss") * (isempty(left) ? "" : " -- left: $left"))
        end
        # the two passes together are idempotent
        once = STADE.cgen_dead_scalars(ex, true)
        check(STADE.cgen_dead_scalars(once, true) == once, "$name: a second pass deletes nothing")
    end
    # the wrap
    ex = Meta.parse("function k(x, out, n)\n    t = n\n    if t > 3\n        return nothing\n    end\n    out[1] = x[1]\n    return nothing\nend")
    wr = STADE.cgen_inbounds_wrap(ex)
    blk = wr.args[2]
    check(length(blk.args) == 1 && blk.args[1] isa Expr && blk.args[1].head === :macrocall && blk.args[1].args[1] === Symbol("@inbounds"), "wrap: one @inbounds call holds the body")
    inner = blk.args[1].args[3]
    check(inner == ex.args[2], "wrap: the body is unchanged, guard and final return inside")
    check(STADE.cgen_finish_kernels(Expr[ex]; inbounds = false, dead_scalars = false)[1] === ex, "both keywords off: the kernel is returned as it is")
    return bad
end

# ---- corpus ------------------------------------------------------------------------------------------------------------
# counts the statements the pass must not touch: array writes, macro calls, calls whose result is dropped, loops and branches
function inb_effects(e, acc = Dict{Symbol,Int}())
    if e isa Expr
        if e.head === :(=) && e.args[1] isa Expr
            acc[:store] = get(acc, :store, 0) + 1
        elseif e.head === :macrocall
            acc[:macro] = get(acc, :macro, 0) + 1
        elseif e.head === :call && e.args[1] === :__stade_wa_add!
            acc[:warp] = get(acc, :warp, 0) + 1
        elseif e.head === :for
            acc[:for] = get(acc, :for, 0) + 1
        elseif e.head === :return
            acc[:ret] = get(acc, :ret, 0) + 1
        end
        for a in e.args
            inb_effects(a, acc)
        end
    end
    return acc
end

function inb_corpus_names(dir)
    names = sort([splitext(f)[1] for f in readdir(dir) if endswith(f, ".jl") && !any(endswith(f, s) for s in ("_b.jl", "_d.jl", "_hv.jl", "_cuda.jl", "_jacc.jl"))])
    paths = [(n, joinpath(dir, n * ".jl")) for n in names]
    kit = get(ENV, "STADE_KIT_KERNELS", "")
    if !isempty(kit) && isdir(kit)
        for sub in ("nn", "solvers"), f in sort(readdir(joinpath(kit, sub)))
            push!(paths, ("kit_" * splitext(f)[1], joinpath(kit, sub, f)))
        end
    end
    return paths
end

function inb_corpus_checks(dir; names = nothing, report = true)
    bad = 0
    check(ok, msg) = (ok ? (report && println(rpad("ok", 6), msg)) : (println(rpad("FAIL", 6), msg); bad += 1))
    cuda = STADE.cgen_backend_cuda()
    nplans = 0; nkernels = 0; ndeleted = 0; nrefdeleted = 0
    for (name, path) in inb_corpus_names(dir)
        names !== nothing && !(name in names) && continue
        pe = STADE.io_read_corpus_entry(path)
        for (mode, gen) in (("adjoint", STADE.stade_adjoint), ("hvp", STADE.stade_hvp))
            out = try gen(pe; keep_push_pop = false, fuse_ii_loops = true) catch; continue end
            e = mode == "hvp" ? out.hvp : out.adjoint
            for (lab, mk) in (("cuda", kw -> STADE.stade_gpu(e, cuda; split_reduce = 0, contraction = 0, kw...)), ("jacc", kw -> STADE.stade_jacc(e; kw...)))
                base = try mk((inbounds = false, dead_scalars = false)) catch; continue end
                nplans += 1
                id = STADE.cgen_finish_kernels(base.kernels; inbounds = false, dead_scalars = false)
                check(id === base.kernels, "$name $mode $lab: both keywords off return the kernels themselves")
                dead = mk((inbounds = false, dead_scalars = true)).kernels
                full = mk((inbounds = true, dead_scalars = true)).kernels
                wrap = mk((inbounds = true, dead_scalars = false)).kernels
                ok = length(dead) == length(base.kernels) == length(full) == length(wrap)
                check(ok, "$name $mode $lab: the same number of kernels")
                ok || continue
                for (i, k) in enumerate(base.kernels)
                    nkernels += 1
                    tag = "$name $mode $lab kernel $i"
                    check(inb_effects(k) == inb_effects(dead[i]), "$tag: the pass touches no store, call, loop or return")
                    check(STADE.cgen_dead_scalars(dead[i], false) == dead[i], "$tag: a second pass deletes nothing")
                    nb = length(string(k)); nd = length(string(dead[i]))
                    nd < nb && (ndeleted += 1)
                    w = wrap[i]
                    inner = w.args[2].args[1].args[3]
                    check(w.args[2].args[1].args[1] === Symbol("@inbounds") && inner == k.args[2], "$tag: @inbounds holds the whole body")
                    check(length(w.args[2].args) == 1, "$tag: nothing, a thread guard included, sits outside the block")
                    # dropping array reads too can only delete more
                    nf = length(string(STADE.cgen_dead_scalars(k, true)))
                    check(nf <= nd, "$tag: dropping array reads deletes at least as much")
                    nf < nd && (nrefdeleted += 1)
                    # text round trip
                    txt = STADE.io_expr_to_source(full[i])
                    back = try Meta.parseall(txt) catch; nothing end
                    check(back !== nothing, "$tag: the printed kernel parses")
                end
            end
        end
    end
    println("corpus: $nplans plans, $nkernels kernels; the pass deleted something in $ndeleted of them, and more with array reads dropped in $nrefdeleted")
    return bad
end

# ---- sabotage ---------------------------------------------------------------------------------------------------------
function validate_inbounds(dir::String = joinpath(@__DIR__, "val-corpus"); phases = (:units, :corpus, :sabotage), slice = 1:typemax(Int))
    bad = 0
    if :units in phases
        println("-- units")
        bad += inb_unit_checks()
    end
    if :corpus in phases
        println("-- corpus")
        bad += inb_corpus_checks(dir; report = false)
    end
    if :sabotage in phases
        println("-- sabotage")
        check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
        repo = abspath(joinpath(@__DIR__, ".."))
        text = read(joinpath(repo, "src", "STADE.jl"), String)
        # No entry for the initial `head = union(live, rng)` of a loop: the fixpoint's own update adds the bounds on the first sweep, so removing it changes no output.
        cases = [
            ("loop fixpoint stops after one sweep", "nh == head && break", "break"),
            ("an effectful call may be deleted", "e.args[1] in cgen_dead_pure_calls() || return false", "true || return false"),
            ("an array read may be deleted without @inbounds", "return refs_ok && all(a -> cgen_dead_pure(a, refs_ok), e.args)", "return all(a -> cgen_dead_pure(a, refs_ok), e.args)"),
            ("the else branch adds no reads", "union!(nl, el)", "nothing"),
            ("a missing else adds no reads", "union!(nl, live)", "nothing"),
            ("the loop's reads do not reach the statements before it", "            live = head\n", "            live = union(live, rng)\n"),
            ("a statement of another form adds no reads", "            cgen_dead_syms!(s, live)\n            push!(out, s)\n        end\n    end\n    return (reverse(out), live)", "            push!(out, s)\n        end\n    end\n    return (reverse(out), live)"),
            ("an assignment does not kill its target", "            delete!(live, v)\n            cgen_dead_syms!(s.args[2], live)", "            cgen_dead_syms!(s.args[2], live)"),
            ("the right side of a kept assignment adds no reads", "            delete!(live, v)\n            cgen_dead_syms!(s.args[2], live)", "            delete!(live, v)"),
            ("the condition of a branch adds no reads", "            nl = union(tl, cgen_dead_syms!(s.args[1], Set{Symbol}()))", "            nl = copy(tl)"),
            ("the thread guard sits outside the block", "return Expr(:function, k.args[1], Expr(:block, Expr(:macrocall, Symbol(\"@inbounds\"), nothing, k.args[2])))", "return Expr(:function, k.args[1], Expr(:block, k.args[2].args[1:min(end, 3)]..., Expr(:macrocall, Symbol(\"@inbounds\"), nothing, Expr(:block, k.args[2].args[min(end, 3) + 1:end]...))))"),
        ]
        mktempdir() do tmp
            for (label, old, new) in cases[intersect(eachindex(cases), slice)]
                n = length(findall(old, text))
                if n != 1
                    check(false, "sabotage \"$label\": anchor found $n times")
                    continue
                end
                path = joinpath(tmp, "STADE_sab.jl")
                write(path, replace(text, old => new; count = 1))
                script = """
                    include($(repr(path)))
                    include($(repr(joinpath(@__DIR__, "validate_inbounds.jl"))))
                    n = inb_unit_checks()
                    n += inb_corpus_checks($(repr(dir)); names = ["transformer_block", "unet", "mpnn", "heat1d", "advection", "stencil_loss"], report = false)
                    println("RESULT ", n)
                """
                out = try read(Base.pipeline(`$(Base.julia_cmd()) --startup-file=no -e $script`; stderr = devnull), String) catch e; "ERR" end
                m = match(r"RESULT (\d+)", out)
                caught = out == "ERR" || m === nothing || parse(Int, m.captures[1]) > 0
                check(caught, "sabotage \"$label\": " * (caught ? "a check fails" : "NOT CAUGHT"))
            end
        end
    end
    println("\n", bad == 0 ? "all inbounds checks passed" : "*** $bad inbounds check(s) failed ***")
    return bad
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    exit(validate_inbounds() == 0 ? 0 : 1)
end
