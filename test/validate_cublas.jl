# ==================== validate_cublas.jl =====================
# Checks the cuBLAS offload of the adjoint products (`cublas`, v0.4.60 candidate, opt-in): a product that the adjoint contraction recognizes as
# C[r, c] += sum_k A[r, k] * B[k, c] over dense operands runs as one CUDA.CUBLAS.gemm! (beta = 1) when M * N * K reaches the keyword, and as its tile twin otherwise.
#
#   julia validate_cublas.jl           units, direct checks, sabotage
#   validate_cublas(; phases = (:units,))
#
# The cases are the adjoint-contraction kernels of validate_adjoint_contraction.jl run through the CPU shim (ownership_shim.jl, where CUDA.CUBLAS.gemm! multiplies on the host)
# with `cublas = 1` and with `cublas = 0` from the same random inputs.
#   - a case that tiles: every product runs as a gemm or as a tile twin (their sum is the number of statements), outputs agree to 1e-12, no conflict, something changed;
#   - `cublas = 0` is the very same text as the plan without the keyword; a keyword above the work of every product keeps all the tile twins;
#   - an operand whose view would end past its array (a leading dimension larger than the rows, the array cut at the last element) keeps the tile twin and the outputs agree;
#   - the same strides with the array long enough run the gemm.
# SABOTAGE edits of the recognizer, each of which one check must catch. Two guards have no sabotage entry because the adjoint recognizer cannot reach them: the offset of a view
# is at least 1 (the plain kernel would read before its array), and the store of the epilogue is always one atomic add.

isdefined(Main, :STADE) || include(joinpath(@__DIR__, "..", "src", "STADE.jl"))
isdefined(Main, :ADJ_CASES) || include(joinpath(@__DIR__, "validate_adjoint_contraction.jl"))
isdefined(Main, :CON_CASES) || include(joinpath(@__DIR__, "validate_contraction.jl"))
using Random

function cub_run(src::String, entry::Symbol, args, cub::Union{Int,Nothing}; contraction::Int = 1)
    con_init!()
    return Base.invokelatest(cub_run_body, src, entry, args, cub, contraction)
end
function cub_run_body(src::String, entry::Symbol, args, cub, contraction)
    Main.CUDA.MAXT[] = name -> 1024
    kw = (warp_atomics = false, shared_private = false, split_reduce = 0, contraction = contraction)
    plan = cub === nothing ? STADE.stade_gpu(Meta.parse(src), STADE.cgen_backend_cuda(); kw...) : STADE.stade_gpu(Meta.parse(src), STADE.cgen_backend_cuda(); kw..., cublas = cub)
    for k in plan.kernels; Core.eval(Main, k); end
    Core.eval(Main, plan.host)
    host = plan.host.args[1].args[1]
    dev = Any[a isa AbstractArray ? Main.CUDA.TArray(copy(a)) : a for a in args]
    empty!(Main.CUDA.LAUNCHED); empty!(Main.CUDA.VIOLATIONS)
    Base.invokelatest(getfield(Main, host), dev...)
    outs = Any[a isa Main.CUDA.TArray ? copy(a.data) : a for a in dev]
    return (outs = outs, launched = copy(Main.CUDA.LAUNCHED), conflicts = copy(Main.CUDA.VIOLATIONS), plan = plan)
end
cub_ngemm(r) = count(==("cublas_gemm!"), r.launched)
cub_ntile(r) = count(adj_is_twin, r.launched)

# a kernel whose operands x and xb have the row stride `ld` (larger than the inner length k)
const CUB_LD_LOOP = """for c = 1:k
                xb[(t - 1) * ld + c] = xb[(t - 1) * ld + c] + w[(c - 1) * n + o] * gs
                wb[(c - 1) * n + o] = wb[(c - 1) * n + o] + x[(t - 1) * ld + c] * gs
            end"""
function cub_ld_args(sd; m = 67, n = 70, k = 19, ld = 25, cut = true)
    r = MersenneTwister(sd); xl = cut ? (m - 1) * ld + k : m * ld + k; L = k * n + n + 8
    return Any[randn(r, m * n + 8), randn(r, xl), randn(r, L), randn(r, xl), randn(r, L), randn(r, n), m, n, k, ld]
end

function cub_unit_checks()
    bad = 0
    check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
    for (name, src, entry, mk, must) in ADJ_CASES
        (must isa Integer && must > 0) || continue
        try
            args = mk(1)
            a = cub_run(src, entry, args, 1); b = cub_run(src, entry, args, 0)
            ng = cub_ngemm(a); nt = cub_ntile(a)
            # a recomputed forward value is one more product: a gemm into the scratch array, or its tile twin
            nst = count(k -> adj_is_stage(string(k.args[1].args[1])), a.plan.kernels); nt += count(adj_is_stage, a.launched)
            must_all = must + nst
            changed = any(x != y for (x, y) in zip(a.outs, args) if x isa AbstractArray)
            err = con_relerr(a, b)
            # the products with a sum of two terms, a batch or a head in the column keep the tile twin: the count of both is the number of statements
            check(ng + nt == must_all && changed && err <= 1e-12 && isempty(a.conflicts), "$name: $ng gemm + $nt tile (want $must_all), error $(round(err, sigdigits = 3)), conflicts $(length(a.conflicts))")
            check(cub_ngemm(b) == 0, "$name: cublas = 0 launches no gemm")
        catch e
            check(false, "$name: error " * first(split(sprint(showerror, e), "\n")))
        end
    end
    # the plain two-statement product is offloaded in both statements
    s2 = ADJ_CASES[1]
    a = cub_run(s2[2], s2[3], s2[4](1), 1)
    check(cub_ngemm(a) == 2 && cub_ntile(a) == 0, "two statements: 2 gemm, no tile twin")
    # the keyword 0 is the plan without the keyword; a keyword above the work keeps the tile twins
    args = s2[4](1)
    t0 = STADE.io_expr_to_source(cub_run(s2[2], s2[3], args, 0).plan.host); tn = STADE.io_expr_to_source(cub_run(s2[2], s2[3], args, nothing).plan.host)
    check(t0 == tn, "cublas = 0 is the plan without the keyword")
    big = cub_run(s2[2], s2[3], args, typemax(Int) ÷ 2)
    check(cub_ngemm(big) == 0 && cub_ntile(big) == 2, "a keyword above the work of every product runs the tile twins (gemm $(cub_ngemm(big)), tile $(cub_ntile(big)))")
    # the work threshold: 67 * 70 * 19 = 89110 products
    lo = cub_run(s2[2], s2[3], args, 89110); hi = cub_run(s2[2], s2[3], args, 89111)
    check(cub_ngemm(lo) == 2 && cub_ngemm(hi) == 0, "the threshold is M * N * K (89110: $(cub_ngemm(lo)) gemm, 89111: $(cub_ngemm(hi)) gemm)")
    # a leading dimension larger than the rows
    src = adj_src(params = ", ld", loop = CUB_LD_LOOP)
    for (cut, wantg) in ((true, 0), (false, 2))
        args = cub_ld_args(3; cut = cut)
        a = cub_run(src, :con_adj, args, 1); b = cub_run(src, :con_adj, args, 0)
        err = con_relerr(a, b)
        check(cub_ngemm(a) == wantg && cub_ngemm(a) + cub_ntile(a) == 2 && err <= 1e-12 && isempty(a.conflicts), "row stride ld > k, array $(cut ? "cut at the last element" : "long enough"): $(cub_ngemm(a)) gemm (want $wantg), $(cub_ntile(a)) tile, error $(round(err, sigdigits = 3))")
    end
    # rows that overlap (a row stride smaller than the inner length): no view of ld rows exists, the tile twin runs
    args = cub_ld_args(4; ld = 2, cut = true)
    a = cub_run(src, :con_adj, args, 1); b = cub_run(src, :con_adj, args, 0)
    check(cub_ngemm(a) == 0 && cub_ntile(a) == 2 && con_relerr(a, b) <= 1e-12, "a row stride smaller than the inner length keeps the tile twins ($(cub_ngemm(a)) gemm, $(cub_ntile(a)) tile)")
    # the plan tests that every size is positive
    ht = STADE.io_expr_to_source(cub_run(s2[2], s2[3], s2[4](1), 1).plan.host)
    check(all(occursin(x, ht) for x in ("m > 0", "n > 0", "k > 0")), "the gemm condition tests m, n and k for positivity")
    # forward products: the sums from one gemm into a zero array, then the plain kernel without its loop (`_fa!`)
    fwd_gemm = Set(["bias product", "residual, tanh and two stores", "operands in the other order", "one row", "same array on both sides", "left operand read across the inner index",
                    "the start value is read from an array of the output index", "head in the column, the operands do not read the head"])
    for (name, src, entry, mk, must) in CON_CASES
        must === true || continue
        try
            args = mk(1); a = cub_run(src, entry, args, 1); b = cub_run(src, entry, args, 0)
            ng = cub_ngemm(a); nfa = count(endswith("_fa!"), a.launched); err = con_relerr(a, b)
            want = name in fwd_gemm ? 1 : 0
            check(ng == want && nfa == ng && err <= 1e-12 && isempty(a.conflicts), "forward, $name: $ng gemm (want $want), $nfa plain kernels without the loop, error $(round(err, sigdigits = 3))")
        catch e
            check(false, "forward, $name: error " * first(split(sprint(showerror, e), "\n")))
        end
    end
    # a loop that starts at a parameter, the array long enough for the view: the gemm runs with the start offset
    lc = first(c for c in CON_CASES if c[1] == "loop starts at a parameter")
    args = lc[4](1); args[1] = vcat(args[1], randn(MersenneTwister(5), args[7]))
    a = cub_run(lc[2], lc[3], args, 1); b = cub_run(lc[2], lc[3], args, 0)
    check(cub_ngemm(a) == 1 && con_relerr(a, b) <= 1e-12, "forward, a loop that starts at a parameter with a long array: $(cub_ngemm(a)) gemm, error $(round(con_relerr(a, b), sigdigits = 3))")
    # a subtracted sum is never offloaded
    sc = first(c for c in CON_CASES if c[1] == "subtracted product")
    a = cub_run(sc[2], sc[3], sc[4](1), 1)
    check(cub_ngemm(a) == 0, "forward, a subtracted sum stays on the tile twin")
    # cublas = 0 gives the text without the keyword for a forward kernel too
    bc = first(c for c in CON_CASES if c[1] == "bias product")
    check(STADE.io_expr_to_source(cub_run(bc[2], bc[3], bc[4](1), 0).plan.host) == STADE.io_expr_to_source(cub_run(bc[2], bc[3], bc[4](1), nothing).plan.host), "forward, cublas = 0 is the plan without the keyword")
    # a recomputed forward value below the threshold: the tile stage fills the scratch array with the sum of the products alone, and the first kernel adds the start value
    for (name, src, entry, mk, must) in ADJ_CASES
        (must isa Integer && must > 0) || continue
        args = mk(1); a = cub_run(src, entry, args, typemax(Int) ÷ 2); b = cub_run(src, entry, args, 0)
        any(k -> adj_is_stage(string(k.args[1].args[1])), a.plan.kernels) || continue
        check(cub_ngemm(a) == 0 && con_relerr(a, b) <= 1e-12 && isempty(a.conflicts), "$name: a threshold above the work, the tile stage keeps the result ($(cub_ngemm(a)) gemm, error $(round(con_relerr(a, b), sigdigits = 3)))")
    end
    # a stride that is not a leading dimension (no unit stride in either index) keeps the tile twin
    src2 = adj_src(params = ", ld", loop = """for c = 1:k
                xb[((t - 1) * ld + c - 1) * 2 + 1] = xb[((t - 1) * ld + c - 1) * 2 + 1] + w[(c - 1) * n + o] * gs
            end""")
    args = cub_ld_args(5; cut = false); args[2] = randn(MersenneTwister(1), 2 * length(args[2])); args[4] = randn(MersenneTwister(2), 2 * length(args[4]))
    a = cub_run(src2, :con_adj, args, 1); b = cub_run(src2, :con_adj, args, 0)
    check(cub_ngemm(a) == 0 && cub_ntile(a) == 1 && con_relerr(a, b) <= 1e-12, "a doubled stride keeps the tile twin ($(cub_ngemm(a)) gemm, $(cub_ntile(a)) tile)")
    return bad
end

# the recognizer on descriptors: the strides, the offset and the swap
function cub_direct_checks()
    bad = 0
    check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
    e = :((((1 - 1) * d + 1) + div(1 - dff, -1)) + 1)
    check(STADE.cgen_cublas_simp(:((1 - 1) * d + 1)) == 1, "folding: (1 - 1) * d + 1 is 1")
    check(STADE.cgen_cublas_simp(:(div(1 - dff, -1) + 1)) == :dff, "folding: div(1 - X, -1) + 1 is X")
    check(STADE.cgen_cublas_simp(:((d * tt + 1) - 1)) == :(d * tt), "folding: (X + 1) - 1 is X")
    check(STADE.cgen_cublas_simp(:(d >= d)) === true, "folding: d >= d is true")
    check(STADE.cgen_cublas_simp(:(a * 1 + 0)) == :a, "folding: a * 1 + 0 is a")
    at = Dict{Symbol,Any}(:t => 1, :c => 1, :o => 5)
    f = STADE.cgen_cublas_affine(:((t - 1) * k + c), Symbol[:t, :c], at)
    check(f !== nothing && f.co[2] == 1 && f.co[1] == :k && STADE.cgen_cublas_simp(f.off) == 1, "affine: (t - 1) * k + c has strides (k, 1), offset 1")
    check(STADE.cgen_cublas_affine(:(t * t + c), Symbol[:t, :c], at) === nothing, "affine: t * t is refused")
    check(STADE.cgen_cublas_affine(:(div(t, 2) + c), Symbol[:t, :c], at) === nothing, "affine: div(t, 2) is refused")
    check(STADE.cgen_cublas_affine(:((t - 1) * c + c), Symbol[:t, :c], at) === nothing, "affine: a stride that reads the other variable is refused")
    check(STADE.cgen_cublas_operand(:w, 1, :m, :k, :k, 1).tr === true && STADE.cgen_cublas_operand(:w, 1, :m, :k, 1, :m).tr === false, "operand: unit stride in the columns is a transpose, in the rows is not")
    check(STADE.cgen_cublas_operand(:w, 1, :m, :k, :a, :b) === nothing, "operand: no unit stride is refused")
    return bad
end

function validate_cublas(; phases = (:units, :sabotage), slice = 1:typemax(Int))
    bad = 0
    if :units in phases
        println("-- units")
        bad += cub_unit_checks()
        bad += cub_direct_checks()
    end
    if :sabotage in phases
        println("-- sabotage")
        check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
        text = read(joinpath(abspath(joinpath(@__DIR__, "..")), "src", "STADE.jl"), String)
        cases = [
            ("the view may end past the array", "Expr(:call, :<=, last, len)", "true"),
            ("the leading dimension may be smaller than the rows", "Expr(:call, :>=, ld, srows)", "true"),
            ("a stride that is not unit is taken as a leading dimension", "    elseif sc == 1\n        ld = sr; srows = cols; scols = rows; tr = true\n    else\n        return nothing\n    end", "    else\n        ld = sr; srows = cols; scols = rows; tr = true\n    end"),
            ("the transpose flag is inverted", "ld = sr; srows = cols; scols = rows; tr = true", "ld = sr; srows = cols; scols = rows; tr = false"),
            ("the output is never swapped", "swap = fc.co[1] != 1", "swap = false"),
            ("the swap keeps the order of the operands", "call = swap ? Expr(:call, gemm, tc(flip(ob.tr)), tc(flip(oa.tr)), 1.0, ob.view, oa.view, 1.0, oc.view) :", "call = swap ? Expr(:call, gemm, tc(flip(ob.tr)), tc(flip(oa.tr)), 1.0, oa.view, ob.view, 1.0, oc.view) :"),
            ("the beta is zero", "tc(oa.tr), tc(ob.tr), 1.0, oa.view, ob.view, 1.0, oc.view)", "tc(oa.tr), tc(ob.tr), 1.0, oa.view, ob.view, 0.0, oc.view)"),
            ("the work threshold is ignored", "Float64(minwork)))", "0.0))"),
            ("the start of the columns is ignored", "Dict{Symbol,Any}(u => 1, v => 1, cv => cbv)", "Dict{Symbol,Any}(u => 1, v => 1, cv => 1)"),
            ("a coefficient that reads a loop variable is accepted", "isempty(intersect(cgen_split_vars(c), Set(vars))) || return nothing", "true"),
            ("an empty product is offloaded", "Expr(:call, :>, M, 0), Expr(:call, :>, N, 0)), Expr(:call, :>, K, 0))", "true, true), true)"),
            ("the guard reads the length of the scratch before it exists", "minwork; lenC = Expr(:call, :*, d.Mh, d.Nh))", "minwork; lenC = Expr(:call, :length, scratch))"),
            ("the forward kernel drops the start value", "repl = Expr(:(=), acc, Expr(:call, :+, acc, Expr(:ref, scratch, first_st.args[1])))", "repl = Expr(:(=), acc, Expr(:ref, scratch, first_st.args[1]))"),
            ("the stage keeps the start value in the scratch", "Expr(:(=), st.args[1], 0.0) : st for (k, st) in enumerate(pre[1:fi+1])]", "st : st for (k, st) in enumerate(pre[1:fi+1])]"),
            ("the first kernel overwrites the sum with the scratch", "fstage.pure ? Expr(:call, :+, fstage.accs[k], Expr(:ref, fstage.scratch[k], idxv)) : Expr(:ref, fstage.scratch[k], idxv)", "Expr(:ref, fstage.scratch[k], idxv)"),
            ("a subtracted sum is offloaded", "(sums !== nothing && length(sums) == 1 && sums[1][2] === :+ && length(sums[1][3]) == 1) || return nothing", "(sums !== nothing && length(sums) == 1 && length(sums[1][3]) == 1) || return nothing"),
            ("the start of the forward loop is ignored", "Dict{Symbol,Any}(u => 1, v => lo, cv => 1)", "Dict{Symbol,Any}(u => 1, v => 1, cv => 1)"),
            ("the keyword is ignored", "cp = cublas > 0 ? cgen_cublas_plan(d, cublas) : nothing", "cp = cgen_cublas_plan(d, max(cublas, 1))"),
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
                    cd($(repr(@__DIR__)))
                    include($(repr(joinpath(@__DIR__, "validate_cublas.jl"))))
                    n = cub_unit_checks() + cub_direct_checks()
                    println("RESULT ", n)
                """
                out = try read(Base.pipeline(`$(Base.julia_cmd()) --startup-file=no -e $script`; stderr = devnull), String) catch e; "ERR" end
                m = match(r"RESULT (\d+)", out)
                caught = out == "ERR" || m === nothing || parse(Int, m.captures[1]) > 0
                check(caught, "sabotage \"$label\": " * (caught ? "a check fails" : "NOT CAUGHT"))
            end
        end
    end
    println("\n", bad == 0 ? "all cuBLAS offload checks passed" : "*** $bad cuBLAS offload check(s) failed ***")
    return bad
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    exit(validate_cublas() == 0 ? 0 : 1)
end
