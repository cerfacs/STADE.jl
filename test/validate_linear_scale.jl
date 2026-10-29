isdefined(Main, :STADE) || include(joinpath(@__DIR__, "..", "src", "STADE.jl"))
using Random

# The linearity rule of the value-needed analysis, and the first-touch test that shares it.
#
# A call is linear in its operands, with a partial that holds no variable, when it is +, -, a product with exactly one operand that is not a
# literal, or a quotient with a literal denominator. A variable read only through such calls does not need its value at the reverse sweep, so a
# write to it needs no snapshot. The rule lives in three copies (snap_, agen_, ii_) that must agree. The first-touch test is a different question
# -- does the PRIMAL read the entry value -- so it counts every occurrence, linear or not.

lin_corpus_names() = ["lin_pingpong", "lin_quotient", "lin_then_quad", "lin_quad_inside", "lin_guard_products", "lin_carried_sum", "lin_carried_scale"]

function lin_value_sets(expr)
    vars = Set{Symbol}([:u, :a, :x, :s, :idx, :v])
    # snap_ and agen_ also name integer index variables read inside a nonlinear index expression; only the float names are compared
    a = Set{Symbol}(); STADE.snap_var_value_needed!(expr, a, false)
    b = Set{Symbol}(); STADE.agen_var_value_needed!(expr, b, false)
    c = Set{Symbol}(); STADE.ii_value_reads(expr, vars, c, false)
    return intersect(a, vars), intersect(b, vars), c
end

function lin_unit_checks()
    bad = 0
    check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
    cases = [
        (:(u[i]), Symbol[]),
        (:(0.25 * u[i]), Symbol[]),
        (:(u[i] * 0.25), Symbol[]),
        (:(2 * u[i]), Symbol[]),
        (:(0.25 * (u[i - 1] - 2.0 * u[i] + u[i + 1])), Symbol[]),
        (:(u[i] + 0.25 * (u[i - 1] - 2.0 * u[i] + u[i + 1])), Symbol[]),
        (:(u[i] / 4.0), Symbol[]),
        (:(u[i] / (2.0 * 2.0)), Symbol[]),
        (:((2.0 * 3.0) * u[i]), Symbol[]),
        (:(0.5 * u[i] / 2.0), Symbol[]),
        (:(-u[i]), Symbol[]),
        (:(-(0.5 * u[i])), Symbol[]),
        (:(0.5 * u[mod(i, 3) + 1]), Symbol[]),
        (:(a * u[i]), [:a, :u]),
        (:(u[i] * u[i]), [:u]),
        (:(0.5 * u[i] * u[i]), [:u]),
        (:(0.5 * (a * u[i])), [:a, :u]),
        (:(1.0 / u[i]), [:u]),
        (:(u[i] / a), [:a, :u]),
        (:(sin(0.5 * u[i])), [:u]),
        (:(2.0 * sin(u[i])), [:u]),
        (:(0.5 * u[i]^2), [:u]),
        (:(0.5 * x * s), [:s, :x]),
        (:(2.0 * u[i] + 3.0 * v[i] * v[i]), [:v]),
    ]
    for (e, want) in cases
        a, b, c = lin_value_sets(e)
        w = Set{Symbol}(want)
        check(a == w, "snap_  $(e): needs $(sort(collect(a)))")
        check(b == w, "agen_  $(e): needs $(sort(collect(b)))")
        check(c == w, "ii_    $(e): needs $(sort(collect(c)))")
    end
    # first touch: every occurrence in the primal is a read, linear or not; a while never proves a write
    body(src) = STADE.parse_kernel(Meta.parse(src)).body
    ft = [
        ("linear sum read before a branch write", "function f(x, y, n)\n s = 1.0\n for i = 1:n\n y[i] = s + 1.0\n if x[i] > 0.0\n s = x[i]\n else\n s = 2.0 * x[i]\n end\n end\n return nothing\nend", :read),
        ("literal product read before a branch write", "function f(x, y, n)\n s = 1.0\n for i = 1:n\n y[i] = 0.5 * s\n if x[i] > 0.0\n s = x[i]\n else\n s = 2.0 * x[i]\n end\n end\n return nothing\nend", :read),
        ("branch write first", "function f(x, y, n)\n s = 1.0\n for i = 1:n\n if x[i] > 0.0\n s = x[i]\n else\n s = 2.0 * x[i]\n end\n y[i] = s + 1.0\n end\n return nothing\nend", :write),
        ("while condition reads it", "function f(x, y, n)\n s = 1.0\n for i = 1:n\n k = 0\n while s > 0.3 && k < 3\n k = k + 1\n end\n if x[i] > 0.0\n s = x[i]\n else\n s = 2.0 * x[i]\n end\n end\n return nothing\nend", :read),
        ("while body reads it", "function f(x, y, n)\n s = 1.0\n for i = 1:n\n k = 0\n while k < 3\n y[i] = y[i] + s\n k = k + 1\n end\n if x[i] > 0.0\n s = x[i]\n else\n s = 2.0 * x[i]\n end\n end\n return nothing\nend", :read),
        ("while body write proves nothing", "function f(x, y, n)\n s = 1.0\n for i = 1:n\n k = 0\n while k < 3\n s = 2.0\n k = k + 1\n end\n y[i] = 1.0\n end\n return nothing\nend", :none),
    ]
    for (label, src, want) in ft
        loop = first(filter(st -> st.kind == :for, body(src)))
        got = STADE.norm_first_touch(loop.body, :s)
        check(got === want, "first touch [$label]: $got")
    end
    return bad
end

# generation and the four oracles over the corpus kernels, plus the stack each one must keep
function lin_corpus_checks(dir::String; names = lin_corpus_names(), oracles::Bool = true)
    bad = 0
    check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
    want_stacks = Dict("lin_pingpong" => Set{String}(), "lin_quotient" => Set{String}(), "lin_then_quad" => Set{String}(),
                       "lin_quad_inside" => Set(["u_stack"]), "lin_guard_products" => Set(["u_stack", "v_stack"]))
    mktempdir() do tmp
        for name in names, fuse in (false, true)
            path = joinpath(dir, name * ".jl")
            out = joinpath(tmp, "$(name)_$(Int(fuse))_b.jl")
            STADE.stade_adjoint_file(path, out; keep_push_pop = false, fuse_ii_loops = fuse)
            txt = read(out, String)
            got = Set{String}(m.captures[1] for m in eachmatch(r"\b([A-Za-z]\w*_stack)\[", txt))
            if haskey(want_stacks, name)
                check(got == want_stacks[name], "$name [fuse=$(Int(fuse))]: stacks $(sort(collect(got))), want $(sort(collect(want_stacks[name])))")
            end
            oracles || continue
            Random.seed!(hash(name))
            for (lab, fn) in (("tangent", STADE.stade_validate_tangent_file), ("adjoint", STADE.stade_validate_adjoint_file), ("hvp", STADE.stade_validate_hvp_file))
                r = fn(path; trials = 8, keep_push_pop = false, fuse_ii_loops = fuse, int_lo = 3, int_hi = 5)
                check(r.ok, "$name [fuse=$(Int(fuse))] $lab oracle: max rel err $(round(r.max_rel_err, sigdigits = 3))")
            end
            r = STADE.stade_validate_dotprod_file(path; trials = 8, keep_push_pop = false, fuse_ii_loops = fuse)
            check(r.ok, "$name [fuse=$(Int(fuse))] dot-product identity: $(round(r.max_rel_err, sigdigits = 3))")
        end
        # a while inside a for must generate under the fusion pass and pass every oracle
        wpath = joinpath(@__DIR__, "while-corpus", "while_carried_cond.jl")
        for fuse in (false, true)
            ok = try
                STADE.stade_adjoint_file(wpath, joinpath(tmp, "w_$(Int(fuse))_b.jl"); keep_push_pop = true, fuse_ii_loops = fuse)
                Random.seed!(hash("while_carried_cond"))
                r = STADE.stade_validate_adjoint_file(wpath; trials = 3, keep_push_pop = true, fuse_ii_loops = fuse, int_lo = 0, int_hi = 4)
                r.ok
            catch
                false
            end
            check(ok, "while_carried_cond [fuse=$(Int(fuse))]: generates and the adjoint oracle passes")
        end
    end
    return bad
end

"""
    validate_linear_scale(dir = "val-corpus"; sabotage = true)

  1. UNIT: 24 expressions through the three copies of the linearity rule. Each copy must name exactly the variables whose values the local
     partials need, and 6 bodies through the first-touch test.
  2. CORPUS: 7 `lin_*` kernels. The adjoint of each keeps exactly the stacks that its nonlinear reads require (none for the linear ping-pong,
     the quotient and the linear passes under a final quadratic loss), and every oracle (tangent, adjoint, HVP, dot product) passes with and
     without fusion. A `while` inside a `for` generates under the fusion pass.
  3. SABOTAGE: each of 12 edits of the source (every copy of the rule widened or narrowed, the first-touch test reverted to the value-needed
     rule, the while arms of it dropped, `cgen_loop_runs_once` applied to a `while`) must fail a check above.
"""
function validate_linear_scale(dir::String = joinpath(@__DIR__, "val-corpus"); sabotage::Bool = true)
    bad = 0
    println("-- unit checks")
    bad += lin_unit_checks()
    println("-- corpus kernels")
    bad += lin_corpus_checks(dir)
    if sabotage
        println("-- sabotage")
        repo = abspath(joinpath(@__DIR__, ".."))
        text = read(joinpath(repo, "src", "STADE.jl"), String)
        cases = [
            ("snap_: every product is linear", "op == :* && return count(a -> !snap_const_expr(a), args) == 1", "op == :* && return true"),
            ("agen_: every product is linear", "op == :* && return count(a -> !agen_const_expr(a), args) == 1", "op == :* && return true"),
            ("ii_: every product is linear", "op == :* && return count(a -> !ii_const_expr(a), args) == 1", "op == :* && return true"),
            ("snap_: every quotient is linear", "op == :/ && return length(args) == 2 && snap_const_expr(args[2])", "op == :/ && return true"),
            ("agen_: every quotient is linear", "op == :/ && return length(args) == 2 && agen_const_expr(args[2])", "op == :/ && return true"),
            ("ii_: every quotient is linear", "op == :/ && return length(args) == 2 && ii_const_expr(args[2])", "op == :/ && return true"),
            ("snap_: a variable counts as a constant", "snap_const_expr(e) = e isa Number ||", "snap_const_expr(e) = e isa Number || e isa Symbol ||"),
            ("snap_: back to +,- only", "op == :* && return count(a -> !snap_const_expr(a), args) == 1", "op == :* && return false"),
            ("first touch: linear reads do not count", "# any occurrence is a read here, linear or not: the primal executes it and sees the entry value\n            reads = Set{Symbol}()\n            agen_collect_expr_vars!(st.rhs, reads)",
                                                       "# any occurrence is a read here, linear or not: the primal executes it and sees the entry value\n            reads = Set{Symbol}()\n            agen_var_value_needed!(st.rhs, reads, false)"),
            ("first touch: a while condition is not read", "# the condition runs first and the body may not run at all\n            reads = Set{Symbol}()\n            agen_collect_expr_vars!(st.cond, reads)\n            v in reads && return :read\n", "# the condition runs first and the body may not run at all\n"),
            ("first touch: a while body read is ignored", "norm_first_touch(st.body, v) === :read && return :read\n        elseif st.kind == :if", "nothing\n        elseif st.kind == :if"),
            ("runs-once test applied to a while", "st.kind == :for && st.lo isa Number && st.hi isa Number && st.step isa Number &&", "st.lo isa Number && st.hi isa Number && st.step isa Number &&"),
        ]
        mktempdir() do tmp
            for (label, old, new) in cases
                n = length(findall(old, text))
                if n != 1
                    check(false, "sabotage \"$label\": anchor found $n times")
                    continue
                end
                path = joinpath(tmp, "STADE_sab.jl")
                write(path, replace(text, old => new; count = 1))
                script = """
                    include($(repr(path)))
                    include($(repr(joinpath(@__DIR__, "validate_linear_scale.jl"))))
                    n = lin_unit_checks()
                    n += lin_corpus_checks($(repr(dir)); names = ["lin_guard_products", "lin_carried_sum", "lin_then_quad"])
                    println("RESULT ", n)
                """
                out = try read(pipeline(`$(Base.julia_cmd()) --startup-file=no -e $script`; stderr = devnull), String) catch e; "ERR" end
                m = match(r"RESULT (\d+)", out)
                caught = out == "ERR" || m === nothing || parse(Int, m.captures[1]) > 0
                println(rpad(caught ? "ok" : "FAIL", 6), "sabotage \"$label\": ", caught ? "a check fails" : "NOT CAUGHT")
                caught || (bad += 1)
            end
        end
    end
    println("\n", bad == 0 ? "all linear scale checks passed" : "*** $bad linear scale check(s) failed ***")
    return bad
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    exit(validate_linear_scale() == 0 ? 0 : 1)
end
