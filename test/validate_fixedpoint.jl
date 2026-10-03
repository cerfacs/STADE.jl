include(joinpath(@__DIR__, "..", "src", "STADE.jl"))

# Phases 0 to 2 of the fixed-point plan: selection, derivation, refusals.
# No emission yet, so there is no gradient to check here. What is checked is
# that the stage picks the right loops, derives the right roles, and refuses
# what contradicts a declaration.
#
# The taped adjoint of every positive is ALSO checked, because a
# fixed-point loop is a `while` loop and its taped gradient is the reference
# the implicit path will be measured against. If that reference is wrong,
# every later comparison is meaningless.

const FP_DIR = joinpath(@__DIR__, "fp-corpus")

fp_load(n) = STADE.parse_kernel(STADE.io_read_corpus_entry(joinpath(FP_DIR, n * ".jl")))

function validate_fixedpoint()
    checks = 0; bad = 0
    function want(label, got, expect)
        checks += 1
        if got == expect
            println(rpad(label, 52), " ok")
        else
            bad += 1
            println(rpad(label, 52), " FAIL  got ", got, " want ", expect)
        end
    end
    function want_refusal(label, fragment, f)
        checks += 1
        msg = try
            f(); ""
        catch e
            sprint(showerror, e)
        end
        if isempty(msg)
            bad += 1; println(rpad(label, 52), " FAIL  accepted")
        elseif !occursin(fragment, msg)
            bad += 1; println(rpad(label, 52), " FAIL  wrong reason: ", first(msg, 80))
        else
            println(rpad(label, 52), " ok")
        end
    end

    sel(n; declared = nothing) = begin
        k = fp_load(n)
        [f.residual for f in STADE.fpgen_find_loops(k, STADE.act_analyze(k); declared = declared)]
    end

    println("--- selection (section 3.3)")
    # The rule takes a loop whose condition reads its own carried state.
    want("fp_mixed takes the solver, not the schedule", sel("fp_mixed"), [:d])
    want("fp_ambiguous takes both sequential loops", sel("fp_ambiguous"), [:d, :e])
    want("fp_nested_fp takes outer and inner", sel("fp_nested_fp"), [:d_out, :d_in])
    want("fp_in_outer_loop finds the loop inside a for", sel("fp_in_outer_loop"), [:d])
    want("fp_counter is not confused by its counter", sel("fp_counter"), [:d])
    # The documented miss. A schedule never reads the state, so it must be named.
    want("fp_bisection is NOT selected by the rule", sel("fp_bisection"), Symbol[])
    want("fp_bisection is selected when named", sel("fp_bisection"; declared = [:w]), [:w])

    println("\n--- derived roles (section 3.6)")
    roles(n, i = 1) = begin
        k = fp_load(n); am = STADE.act_analyze(k)
        l = STADE.fpgen_find_loops(k, am)[i]
        STADE.fpgen_state_and_params(k, l, am)
    end
    want("fp_jacobi state", roles("fp_jacobi").state, [:u])
    want("fp_jacobi params", roles("fp_jacobi").params, [:a, :f])
    want("fp_two_states derives a joint state", roles("fp_two_states").state, [:u, :v])
    # The nested case: the inner loop sees the outer state as a PARAMETER.
    want("fp_nested_fp outer state", roles("fp_nested_fp", 1).state, [:u, :v])
    want("fp_nested_fp inner state", roles("fp_nested_fp", 2).state, [:v])
    want("fp_nested_fp inner takes u as a parameter", roles("fp_nested_fp", 2).params, [:a, :u])

    println("\n--- refusals (section 5.3)")
    want_refusal("fixed_point = true selecting nothing", "selected no loop",
        () -> begin
            k = fp_load("fp_not_read")
            STADE.fpgen_require_selection(STADE.fpgen_find_loops(k, STADE.act_analyze(k)),
                                          nothing, k.sig.name)
        end)
    want_refusal("a residual that names no loop", "no `while` loop uses",
        () -> begin
            k = fp_load("fp_jacobi")
            STADE.fpgen_find_loops(k, STADE.act_analyze(k); declared = [:nosuch])
        end)
    # These need a CLAIM to contradict, so they exist only for the verbose form.
    want_refusal("a declared parameter the body writes", "An output cannot be a parameter",
        () -> begin
            k = fp_load("fp_param_written"); am = STADE.act_analyze(k)
            l = STADE.fpgen_find_loops(k, am)[1]
            STADE.fpgen_check_declared(k, l, STADE.fpgen_state_and_params(k, l, am),
                                       nothing, [:f, :a])
        end)
    want_refusal("a declared state missing a carrier", "does not include it",
        () -> begin
            k = fp_load("fp_param_written"); am = STADE.act_analyze(k)
            l = STADE.fpgen_find_loops(k, am)[1]
            STADE.fpgen_check_declared(k, l, STADE.fpgen_state_and_params(k, l, am),
                                       [:u], nothing)
        end)

    println("\n--- carriage: read-before-write AND written (section 5.3)")
    # Two corrections that the `:while` work had already needed, and that
    # phase 1 repeated. A temporary is written then read inside ONE
    # iteration and is not carried. A parameter is only read and is not
    # carried however early it is read.
    checks += 1
    let k = fp_load("fp_jacobi"), am = STADE.act_analyze(k),
        l = STADE.fpgen_find_loops(k, am)[1],
        carried = STADE.fpgen_carried(l.stmt.body)
        # `d` is NOT carried by this test, and that is right: the body
        # writes it before reading it. It is carried into the loop by the
        # CONDITION, which fpgen_carried does not walk, and it is excluded
        # from the carrier check anyway as the residual.
        if carried == [:u]
            println(rpad("un is a temporary, f is a parameter, u carries", 52), " ok")
        else
            bad += 1
            println(rpad("un is a temporary, f is a parameter, u carries", 52), " FAIL  ", carried)
        end
    end

    println("\n--- the taped reference (section 8.2)")
    # Every positive must differentiate as an ordinary `while` loop, because
    # that gradient is what the implicit path gets measured against.
    positives = ["fp_jacobi", "fp_linear", "fp_counter", "fp_two_states",
                 "fp_nested_param", "fp_ambiguous", "fp_nested_fp",
                 "fp_in_outer_loop", "fp_zero_iters", "fp_mixed"]
    for n in positives
        checks += 1
        path = joinpath(FP_DIR, n * ".jl")
        try
            STADE.stade_adjoint_file(path, joinpath(FP_DIR, n * "_b.jl"); keep_push_pop = true)
            r = STADE.stade_validate_adjoint_file(path; trials = 2, keep_push_pop = true,
                                                  fuse_ii_loops = false, int_lo = 1, int_hi = 4)
            r.ok || (bad += 1)
            println(rpad(n * " [taped adjoint]", 52), r.ok ? " ok  " : " FAIL  ",
                    round(r.max_rel_err, sigdigits = 4))
        catch e
            bad += 1
            println(rpad(n * " [taped adjoint]", 52), " ERR  ", first(sprint(showerror, e), 70))
        end
    end

    # And the one subject where taping is WRONG rather than slow. This is
    # measured with FIXED inputs, not through the validator, because
    # val_generate_baseline REJECTS a candidate that fails its own tangent
    # self-check and retries. For a piecewise-constant program it therefore
    # keeps drawing until it finds inputs where taping happens to agree, and
    # reports a pass. The disagreement is the property, so it has to be
    # asked for directly.
    checks += 1
    let path = joinpath(FP_DIR, "fp_bisection.jl")
        pr = STADE.io_read_corpus_entry(path)
        b = STADE.stade_adjoint(pr; keep_push_pop = true)
        af = STADE.val_compile(b.adjoint); initf = STADE.val_compile(b.initstacks)
        pf = STADE.val_compile(pr)
        f = [0.4, -1.3]; a = [2.0, 0.7]; n = 2
        prim(ff, aa) = begin
            l = zeros(1); uu = zeros(n); Base.invokelatest(pf, l, uu, ff, aa, n); l[1]
        end
        eps = 1.0e-6
        fd = (prim(f .+ [eps, 0.0], a) - prim(f .- [eps, 0.0], a)) / (2eps)
        sn = Symbol[x for x in b.initstacks.args[1].args[2:end]]
        st = STADE.val_init_stacks(initf, Any[n for _ in sn])
        loss = zeros(1); lossb = ones(1); u = zeros(n); ub = zeros(n)
        fb = zeros(n); ab = zeros(n)
        Base.invokelatest(af, loss, lossb, u, ub, f, fb, a, ab, n, collect(st)...)
        if fb[1] == 0.0 && abs(fd) > 1.0e-3
            println(rpad("fp_bisection: taped is 0, the truth is not", 52), " ok  fd=",
                    round(fd, digits = 4))
        else
            bad += 1
            println(rpad("fp_bisection: taped is 0, the truth is not", 52),
                    " FAIL  taped=", fb[1], " fd=", fd)
        end
    end

    for f in readdir(FP_DIR)
        occursin(r"_(b|d|hv)\.jl$", f) && rm(joinpath(FP_DIR, f); force = true)
        endswith(f, ".yaml") && rm(joinpath(FP_DIR, f); force = true)
    end

    println("\n", checks - bad, "/", checks, " checks passed",
            bad == 0 ? "" : "   *** $bad NOT OK ***")
    return bad
end

validate_fixedpoint()
