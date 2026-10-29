# ==================== validate_adjoint_contraction.jl =====================
# Checks the adjoint contraction (`contraction`, v0.4.35): a loop over the elements of a matrix whose inner loop adds products into two arrays by atomic adds
# (the adjoint of a matrix product) becomes a kernel that stores the seed of each element, and one tiled kernel per atomic statement.
#
#   julia validate_adjoint_contraction.jl           units, refusals, direct recognizer cases, sabotage
#   validate_adjoint_contraction(; phases = (:units,))
#
# Every case is a small kernel in the house style, run through the CPU shim (ownership_shim.jl) twice from the same random inputs: with the twin forced on
# (`contraction = 1`) and with `contraction = 0`.
#   - a case that must tile: the kernels `_a!` and `_t1!` (and `_t2!` for two statements) exist and ran, the two runs agree to 1e-12 (atomic adds reorder sums),
#     the shim reports no conflict, and some output changed;
#   - a case that must not tile: no such kernel exists and the two plans are the very same text.
# SABOTAGE edits of the recognizer, each of which one case must catch.

isdefined(Main, :STADE) || include(joinpath(@__DIR__, "..", "src", "STADE.jl"))
isdefined(Main, :con_run) || include(joinpath(@__DIR__, "validate_contraction.jl"))
using Random

# a kernel: `pre` is read between the decode and the loop, `loop` is the loop (text), `post` follows it
function adj_src(; params = "", dec = "o = mod(idx - 1, n) + 1\n            t = div(idx - 1, n) + 1", pre = "gs = g[idx] * 2.0", loop = nothing, post = "", flat = "m * n", head = "for idx = 1:m * n", entry = "con_adj")
    loop === nothing && (loop = """for c = 1:k
                xb[(t - 1) * k + c] = xb[(t - 1) * k + c] + w[(c - 1) * n + o] * gs
                wb[(c - 1) * n + o] = wb[(c - 1) * n + o] + x[(t - 1) * k + c] * gs
            end""")
    return """function $entry(g, x, w, xb, wb, bb, m, n, k$params)
        $head
            $dec
            $pre
            $loop
            $post
        end
        return nothing
    end"""
end
# random arguments for adj_src kernels: n columns, m rows, k inner length
adj_args(sd; n = 70, m = 67, k = 19, extra = Any[]) = (r = MersenneTwister(sd); L = (m + 1) * k + k * n + n; Any[randn(r, m * n + 8), randn(r, L), randn(r, L), randn(r, L), randn(r, L), randn(r, n), m, n, k, extra...])

# batched kernels (the adjoint of a product over heads): nb batches, m rows, n columns, k inner length; the head is the slowest index
const ADJ_BDEC = "o = mod(idx - 1, n) + 1\n            t = mod(div(idx - 1, n), m) + 1\n            h = div(idx - 1, m * n) + 1"
adj_bargs(sd; nb = 3, n = 70, m = 67, k = 19, extra = Any[]) = (r = MersenneTwister(sd); L = (nb + 1) * max(m, n) * k + 8; Any[randn(r, nb * m * n + 8), randn(r, L), randn(r, L), randn(r, L), randn(r, L), randn(r, n), m, n, k, nb, extra...])
const ADJ_BLOOP = """for c = 1:k
                xb[((h - 1) * m + t - 1) * k + c] = xb[((h - 1) * m + t - 1) * k + c] + w[((h - 1) * n + o - 1) * k + c] * gs
                wb[((h - 1) * n + o - 1) * k + c] = wb[((h - 1) * n + o - 1) * k + c] + x[((h - 1) * m + t - 1) * k + c] * gs
            end"""
const ADJ_CASES = [
    ("two statements", adj_src(), :con_adj, adj_args, 2),
    ("first statement only", adj_src(loop = "for c = 1:k\n xb[(t - 1) * k + c] = xb[(t - 1) * k + c] + w[(c - 1) * n + o] * gs\n end"), :con_adj, adj_args, 1),
    ("second statement only", adj_src(loop = "for c = 1:k\n wb[(c - 1) * n + o] = wb[(c - 1) * n + o] + x[(t - 1) * k + c] * gs\n end"), :con_adj, adj_args, 1),
    ("factors in the other order", adj_src(loop = """for c = 1:k
                xb[(t - 1) * k + c] = xb[(t - 1) * k + c] + gs * w[(c - 1) * n + o]
                wb[(c - 1) * n + o] = wb[(c - 1) * n + o] + gs * x[(t - 1) * k + c]
            end"""), :con_adj, adj_args, 2),
    ("a descending loop", adj_src(loop = """for c = k:-1:1
                xb[(t - 1) * k + c] = xb[(t - 1) * k + c] + w[(c - 1) * n + o] * gs
                wb[(c - 1) * n + o] = wb[(c - 1) * n + o] + x[(t - 1) * k + c] * gs
            end"""), :con_adj, adj_args, 2),
    ("the loop starts at a parameter", adj_src(params = ", c0", loop = """for c = c0:k
                xb[(t - 1) * k + c] = xb[(t - 1) * k + c] + w[(c - 1) * n + o] * gs
                wb[(c - 1) * n + o] = wb[(c - 1) * n + o] + x[(t - 1) * k + c] * gs
            end"""), :con_adj, sd -> adj_args(sd; k = 25, extra = Any[4]), 2),
    ("a recomputed forward value, a repeated decode and temporaries", adj_src(
        pre = "s = b0[o]\n for c = 1:k\n s = s + x[(t - 1) * k + c] * w[(c - 1) * n + o]\n end\n o = mod(idx - 1, n) + 1\n t = div(idx - 1, n) + 1\n gs = tanh(s) * g[idx]",
        loop = """for c = 1:k
                a1 = x[(t - 1) * k + c]
                a2 = w[(c - 1) * n + o]
                xb[(t - 1) * k + c] = xb[(t - 1) * k + c] + a2 * gs
                wb[(c - 1) * n + o] = wb[(c - 1) * n + o] + a1 * gs
            end""", params = ", b0"), :con_adj, sd -> (a = adj_args(sd); r = MersenneTwister(99); Any[a..., randn(r, 70)]), 2),
    ("a bias term after the loop", adj_src(post = "bb[o] = bb[o] + gs"), :con_adj, adj_args, 2),
    ("two loops, the second stays", adj_src(loop = """for c = 1:k
                xb[(t - 1) * k + c] = xb[(t - 1) * k + c] + w[(c - 1) * n + o] * gs
                wb[(c - 1) * n + o] = wb[(c - 1) * n + o] + x[(t - 1) * k + c] * gs
            end
            for c = 1:k
                x2b[(t - 1) * k + c] = x2b[(t - 1) * k + c] + w2[(c - 1) * n + o] * gs
                w2b[(c - 1) * n + o] = w2b[(c - 1) * n + o] + x2[(t - 1) * k + c] * gs
            end""", params = ", x2, w2, x2b, w2b"), :con_adj, sd -> (a = adj_args(sd); r = MersenneTwister(98); Any[a..., randn(r, 67 * 19), randn(r, 19 * 70), zeros(67 * 19), zeros(19 * 70)]), 2),
    ("a start value for the loop start", adj_src(params = ", c0", loop = """for c = c0:k
                xb[(t - 1) * k + c] = xb[(t - 1) * k + c] + w[(c - 1) * n + o] * gs
            end"""), :con_adj, sd -> adj_args(sd; k = 25, extra = Any[1]), 1),
    ("a partial last row", adj_src(head = "for idx = 1:m * n + 5"), :con_adj, adj_args, :plain),
    ("a loop with a plain store, then a loop that tiles", adj_src(loop = """for c = 1:k
                xb[(t - 1) * k + c] = xb[(t - 1) * k + c] + w[(c - 1) * n + o] * gs
                g[idx] = gs
            end
            for c = 1:k
                x2b[(t - 1) * k + c] = x2b[(t - 1) * k + c] + w2[(c - 1) * n + o] * gs
                w2b[(c - 1) * n + o] = w2b[(c - 1) * n + o] + x2[(t - 1) * k + c] * gs
            end""", params = ", x2, w2, x2b, w2b"), :con_adj, sd -> (a = adj_args(sd); r = MersenneTwister(97); Any[a..., randn(r, 67 * 19), randn(r, 19 * 70), zeros(67 * 19), zeros(19 * 70)]), 2),
    ("a temporary is assigned twice, used between and after", adj_src(loop = """for c = 1:k
                a1 = x[(t - 1) * k + c]
                wb[(c - 1) * n + o] = wb[(c - 1) * n + o] + a1 * gs
                a1 = w[(c - 1) * n + o]
                xb[(t - 1) * k + c] = xb[(t - 1) * k + c] + a1 * gs
            end"""), :con_adj, adj_args, 2),
    ("a recomputed forward value, subtracted, factors swapped", adj_src(
        pre = "s = 0.5\n for c = 1:k\n s = s - w[(c - 1) * n + o] * x[(t - 1) * k + c]\n end\n gs = s * g[idx]"), :con_adj, adj_args, 2),
    ("two recomputed forward loops, the first is staged", adj_src(
        pre = "s = 0.5\n for c = 1:k\n s = s + x[(t - 1) * k + c] * w[(c - 1) * n + o]\n end\n s2 = 0.0\n for c = 1:k\n s2 = s2 + w[(c - 1) * n + o] * w[(c - 1) * n + o]\n end\n gs = s * s2 * g[idx]"), :con_adj, adj_args, 2),
    # ---- batched over a third index (v0.4.36) ----
    ("batched, two statements", adj_src(params = ", nb", head = "for idx = 1:nb * m * n", dec = ADJ_BDEC, loop = ADJ_BLOOP), :con_adj, adj_bargs, 2),
    ("batched, a recomputed forward value and a repeated decode", adj_src(params = ", nb", head = "for idx = 1:nb * m * n", dec = ADJ_BDEC,
        pre = "s = 0.0\n for c = 1:k\n s = s + x[((h - 1) * m + t - 1) * k + c] * w[((h - 1) * n + o - 1) * k + c]\n end\n o = mod(idx - 1, n) + 1\n t = mod(div(idx - 1, n), m) + 1\n h = div(idx - 1, m * n) + 1\n gs = exp(0.1 * s) * g[idx]",
        loop = ADJ_BLOOP), :con_adj, adj_bargs, 2),
    ("batched, a target that does not read the batch", adj_src(params = ", nb", head = "for idx = 1:nb * m * n", dec = ADJ_BDEC, loop = """for c = 1:k
                xb[(t - 1) * k + c] = xb[(t - 1) * k + c] + w[((h - 1) * n + o - 1) * k + c] * gs
                wb[((h - 1) * n + o - 1) * k + c] = wb[((h - 1) * n + o - 1) * k + c] + x[(t - 1) * k + c] * gs
            end"""), :con_adj, adj_bargs, 2),
    ("batched, the divisor written the other way", adj_src(params = ", nb", head = "for idx = 1:nb * m * n", dec = "o = mod(idx - 1, n) + 1\n            t = mod(div(idx - 1, n), m) + 1\n            h = div(idx - 1, n * m) + 1", loop = ADJ_BLOOP), :con_adj, adj_bargs, 2),
    ("batched, a partial last batch", adj_src(params = ", nb", head = "for idx = 1:nb * m * n + 7", dec = ADJ_BDEC, loop = ADJ_BLOOP), :con_adj, adj_bargs, :plain),
    # ---- refusals ----
    ("batched, the batch size is not rows times columns", adj_src(params = ", nb", head = "for idx = 1:nb * m * n", dec = "o = mod(idx - 1, n) + 1\n            t = mod(div(idx - 1, n), m) + 1\n            h = div(idx - 1, n * n) + 1", loop = ADJ_BLOOP), :con_adj, sd -> adj_bargs(sd; k = 7), 0),
    ("batched, the batch is changed after its decode", adj_src(params = ", nb", head = "for idx = 1:nb * m * n", dec = ADJ_BDEC, pre = "h = h + 0\n gs = g[idx] * 2.0", loop = ADJ_BLOOP), :con_adj, adj_bargs, 0),
    ("batched, the loop variable has the batch's name", adj_src(params = ", nb", head = "for idx = 1:nb * m * n", dec = ADJ_BDEC, loop = """for h = 1:2
                xb[((h - 1) * m + t - 1) * k + h] = xb[((h - 1) * m + t - 1) * k + h] + w[((h - 1) * n + o - 1) * k + h] * gs
            end"""), :con_adj, adj_bargs, 0),
    ("batched, the row decode uses another size than the column", adj_src(params = ", nb", head = "for idx = 1:nb * m * n", dec = "o = mod(idx - 1, n) + 1\n            t = mod(div(idx - 1, n + 1), m) + 1\n            h = div(idx - 1, m * n) + 1", loop = ADJ_BLOOP), :con_adj, adj_bargs, 0),
    ("a target is read elsewhere", adj_src(pre = "gs = g[idx] * 2.0 + xb[o]"), :con_adj, adj_args, 0),
    ("an operand is stored", adj_src(post = "x[o] = gs"), :con_adj, adj_args, 0),
    ("the target index reads neither the row nor the column", adj_src(loop = """for c = 1:k
                xb[c] = xb[c] + w[(c - 1) * n + o] * gs
            end"""), :con_adj, adj_args, 0),
    ("the target index reads the row and the column", adj_src(loop = """for c = 1:k
                xb[(t - 1) * k + c + o] = xb[(t - 1) * k + c + o] + w[(c - 1) * n + o] * gs
            end"""), :con_adj, adj_args, 0),
    ("both indices read the row", adj_src(loop = """for c = 1:k
                xb[(t - 1) * k + c] = xb[(t - 1) * k + c] + w[(t - 1) * k + c] * gs
            end"""), :con_adj, adj_args, 0),
    ("the value is not a product", adj_src(loop = """for c = 1:k
                xb[(t - 1) * k + c] = xb[(t - 1) * k + c] + w[(c - 1) * n + o] + gs
            end"""), :con_adj, adj_args, 0),
    ("a product of three", adj_src(loop = """for c = 1:k
                xb[(t - 1) * k + c] = xb[(t - 1) * k + c] + w[(c - 1) * n + o] * gs * gs
            end"""), :con_adj, adj_args, 0),
    ("the seed is a parameter", adj_src(params = ", sg", loop = """for c = 1:k
                xb[(t - 1) * k + c] = xb[(t - 1) * k + c] + w[(c - 1) * n + o] * sg
            end"""), :con_adj, sd -> adj_args(sd; extra = Any[0.5]), 0),
    ("two seeds", adj_src(pre = "gs = g[idx] * 2.0\n gt = g[idx] * 3.0", loop = """for c = 1:k
                xb[(t - 1) * k + c] = xb[(t - 1) * k + c] + w[(c - 1) * n + o] * gs
                wb[(c - 1) * n + o] = wb[(c - 1) * n + o] + x[(t - 1) * k + c] * gt
            end"""), :con_adj, adj_args, 0),
    ("a temporary is read after the loop", adj_src(pre = "gs = g[idx] * 2.0\n a1 = 0.0", post = "bb[o] = bb[o] + a1", loop = """for c = 1:k
                a1 = x[(t - 1) * k + c]
                wb[(c - 1) * n + o] = wb[(c - 1) * n + o] + a1 * gs
            end"""), :con_adj, adj_args, 0),
    ("a plain store in the loop", adj_src(loop = """for c = 1:k
                xb[(t - 1) * k + c] = xb[(t - 1) * k + c] + w[(c - 1) * n + o] * gs
                bb[o] = gs
            end"""), :con_adj, adj_args, 0),
    ("the row is decoded twice with another variable", adj_src(pre = "t2 = div(idx - 1, n) + 1\n gs = g[idx] * 2.0 + t2"), :con_adj, adj_args, 0),
    ("the row is decoded from another expression", adj_src(dec = "o = mod(idx - 1, n) + 1\n t = div(idx, n) + 1"), :con_adj, adj_args, 0),
    ("the decodes use different sizes", adj_src(params = ", n2", dec = "o = mod(idx - 1, n) + 1\n t = div(idx - 1, n2) + 1"), :con_adj, sd -> adj_args(sd; extra = Any[70]), 0),
    ("the loop variable has the column's name", adj_src(pre = "gs = g[idx] * 2.0 + o", loop = """for o = 1:k
                xb[(t - 1) * k + o] = xb[(t - 1) * k + o] + w[(o - 1) * n + o] * gs
            end"""), :con_adj, adj_args, 0),
    ("the seed has the loop variable's name", adj_src(pre = "c = g[idx] * 2.0", post = "bb[o] = bb[o] + c", loop = """for c = 1:k
                xb[(t - 1) * k + c] = xb[(t - 1) * k + c] + w[(c - 1) * n + o] * c
            end"""), :con_adj, adj_args, 0),
    ("the target index does not read the loop variable", adj_src(loop = """for c = 1:k
                xb[t] = xb[t] + w[(c - 1) * n + o] * gs
            end"""), :con_adj, adj_args, 0),
    ("the operand index does not read the loop variable", adj_src(loop = """for c = 1:k
                xb[(t - 1) * k + c] = xb[(t - 1) * k + c] + w[o] * gs
            end"""), :con_adj, adj_args, 0),
    ("the held index is changed between the decodes", adj_src(dec = "h0 = idx - 1\n o = mod(h0, n) + 1\n h0 = h0 + 1\n t = div(h0, n) + 1"), :con_adj, adj_args, 0),
    ("the loop variable is read after the loop", adj_src(params = ", c", post = "bb[o] = bb[o] + c", loop = """for c = 1:k
                xb[(t - 1) * k + c] = xb[(t - 1) * k + c] + w[(c - 1) * n + o] * gs
            end"""), :con_adj, sd -> adj_args(sd; extra = Any[0.25]), 0),
    ("the loop bound is computed in the kernel", adj_src(pre = "gs = g[idx] * 2.0\n kk = k - 2", loop = """for c = 1:kk
                xb[(t - 1) * k + c] = xb[(t - 1) * k + c] + w[(c - 1) * n + o] * gs
            end"""), :con_adj, adj_args, 0),
]

# The recognizer on hand-written kernel text, for the guards that STADE's own gates make unreachable from source.
adj_kernel(pre, loop, post; first = "idx = 1 + (__tid - 1)", guard = "div(m * n - 1, 1) + 1", params = "", dec = "o = mod(idx - 1, n) + 1\n    t = div(idx - 1, n) + 1") = Meta.parse("""function cuda_kernel_dir!(g, x, w, xb, wb, bb, m, n, k$params)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > $guard
        return nothing
    end
    $first
    $dec
    $pre
    $loop
    $post
    return nothing
end""")
const ADJ_LOOP = "for c = 1:k\n        CUDA.@atomic xb[(t - 1) * k + c] += w[(c - 1) * n + o] * gs\n        CUDA.@atomic wb[(c - 1) * n + o] += x[(t - 1) * k + c] * gs\n    end"
const ADJ_DIRECT = [
    ("control", "gs = g[idx] * 2.0", ADJ_LOOP, "bb[o] = gs", true),
    ("a plain store into an operand array after the loop", "gs = g[idx] * 2.0", ADJ_LOOP, "x[idx] = gs", false),
    ("a plain store into a target array after the loop", "gs = g[idx] * 2.0", ADJ_LOOP, "xb[idx] = gs", false),
    ("the loop variable is a parameter", "gs = g[idx] * 2.0", ADJ_LOOP, "bb[o] = gs", false, ", c"),
    ("the column is changed after its decode", "gs = g[idx] * 2.0\n    o = o + 1", ADJ_LOOP, "bb[o] = gs", false),
    ("the row is changed after its decode", "gs = g[idx] * 2.0\n    t = t + 1", ADJ_LOOP, "bb[o] = gs", false),
    ("the seed is assigned after the loop", "gs = g[idx] * 2.0", ADJ_LOOP, "gs = 0.0", true),
    ("an operand is stored inside a branch", "gs = g[idx] * 2.0", ADJ_LOOP, "if g[idx] > 0.0\n        x[o] = gs\n    end", false),
    ("an operand is also a target", "gs = g[idx] * 2.0", "for c = 1:k\n        CUDA.@atomic xb[(t - 1) * k + c] += xb[(c - 1) * n + o] * gs\n    end", "bb[o] = gs", false),
    ("a temporary has a parameter's name", "gs = g[idx] * 2.0", "for c = 1:k\n        CUDA.@atomic wb[(c - 1) * n + o + q] += x[(t - 1) * k + c] * gs\n        q = 0\n    end", "bb[o] = gs", false, ", q"),
    ("a temporary is read after the loop", "gs = g[idx] * 2.0\n    a1 = 0.0", "for c = 1:k\n        a1 = x[(t - 1) * k + c]\n        CUDA.@atomic wb[(c - 1) * n + o] += a1 * gs\n    end", "bb[o] = a1", false),
]
function adj_direct_checks()
    bad = 0
    for (name, pre, loop, post, must, params...) in ADJ_DIRECT
        kd = adj_kernel(pre, loop, post; params = isempty(params) ? "" : params[1])
        t = try STADE.cgen_adjoint_contraction(kd, STADE.cgen_backend_cuda()) catch e; e end
        ok = must ? (t isa NamedTuple) : t === nothing
        println(rpad(ok ? "ok" : "FAIL", 6), "direct, $name: ", must ? "a twin is built" : "refused")
        ok || (bad += 1)
    end
    # the thread index and the guard of the kernel; the order of the threads reversed
    for (name, first, guard, must) in (("reversed threads", "idx = m * n + (__tid - 1) * -1", "div(1 - m * n, -1) + 1", true),
                                       ("reversed threads, the index offset", "idx = m * n + 1 + (__tid - 1) * -1", "div(1 - m * n, -1) + 1", false))
        kd = adj_kernel("gs = g[idx] * 2.0", ADJ_LOOP, "bb[o] = gs"; first = first, guard = guard)
        t = try STADE.cgen_adjoint_contraction(kd, STADE.cgen_backend_cuda()) catch e; e end
        ok = must ? (t isa NamedTuple) : t === nothing
        println(rpad(ok ? "ok" : "FAIL", 6), "direct, $name: ", must ? "a twin is built" : "refused")
        ok || (bad += 1)
    end
    # the size of the decode is a parameter: a scalar computed in the kernel is not defined where the twin reads it
    for (name, dec) in (("the decode size is computed in the kernel", "nn = n\n    o = mod(idx - 1, nn) + 1\n    t = div(idx - 1, nn) + 1"),)
        kd = adj_kernel("gs = g[idx] * 2.0", ADJ_LOOP, "bb[o] = gs"; dec = dec)
        t = try STADE.cgen_adjoint_contraction(kd, STADE.cgen_backend_cuda()) catch e; e end
        ok = t === nothing
        println(rpad(ok ? "ok" : "FAIL", 6), "direct, $name: refused")
        ok || (bad += 1)
    end
    # a batched kernel whose loop variable has the batch's name: STADE's own passes rename it before the recognizer sees it, so only hand-written text reaches the guard
    bdec = "o = mod(idx - 1, n) + 1\n    t = mod(div(idx - 1, n), m) + 1\n    h = div(idx - 1, m * n) + 1"
    for (name, v, must) in (("batched, control", "c", true), ("batched, the loop variable has the batch's name", "h", false))
        loop = "for $v = 1:k\n        CUDA.@atomic xb[((h - 1) * m + t - 1) * k + $v] += w[((h - 1) * n + o - 1) * k + $v] * gs\n        CUDA.@atomic wb[((h - 1) * n + o - 1) * k + $v] += x[((h - 1) * m + t - 1) * k + $v] * gs\n    end"
        kd = adj_kernel("gs = g[idx] * 2.0", loop, "bb[o] = gs"; dec = bdec, guard = "div(nb * m * n - 1, 1) + 1", params = ", nb")
        t = try STADE.cgen_adjoint_contraction(kd, STADE.cgen_backend_cuda()) catch e; e end
        ok = must ? (t isa NamedTuple) : t === nothing
        println(rpad(ok ? "ok" : "FAIL", 6), "direct, $name: ", must ? "a twin is built" : "refused")
        ok || (bad += 1)
    end
    for (name, first, guard) in (("the flat index is offset", "idx = 2 + (__tid - 1)", "div(m * n - 1, 1) + 1"),
                                 ("the guard is not the product of the sizes", "idx = 1 + (__tid - 1)", "div(q - 1, 1) + 1"))
        kd = adj_kernel("gs = g[idx] * 2.0", ADJ_LOOP, "bb[o] = gs"; first = first, guard = guard)
        t = try STADE.cgen_adjoint_contraction(kd, STADE.cgen_backend_cuda()) catch e; e end
        ok = t === nothing
        println(rpad(ok ? "ok" : "FAIL", 6), "direct, $name: refused")
        ok || (bad += 1)
    end
    return bad
end

adj_is_twin(n) = occursin(r"_t\d+!$", n)
adj_is_stage(n) = endswith(n, "_f_t!")
# the cases whose forward loop becomes a tiled kernel of its own
const ADJ_STAGE = Set(["a recomputed forward value, a repeated decode and temporaries", "a recomputed forward value, subtracted, factors swapped", "two recomputed forward loops, the first is staged", "batched, a recomputed forward value and a repeated decode"])
function adj_unit_checks()
    bad = 0
    check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
    for (name, src, entry, mk, must) in ADJ_CASES
        try
            args = mk(1)
            a = con_run(src, entry, args, 1)
            b = con_run(src, entry, args, 0)
            twins = count(k -> adj_is_twin(string(k.args[1].args[1])), a.plan.kernels)
            firsts = count(k -> endswith(string(k.args[1].args[1]), "_a!"), a.plan.kernels)
            ran = count(adj_is_twin, a.launched)
            nstage = count(k -> adj_is_stage(string(k.args[1].args[1])), a.plan.kernels)
            ranstage = count(adj_is_stage, a.launched)
            ranfirst = any(n -> endswith(n, "_a!"), a.launched)
            changed = any(x != y for (x, y) in zip(a.outs, args) if x isa AbstractArray)
            if must === :plain
                check(twins >= 1 && ran == 0 && !ranfirst && con_relerr(a, b) == 0.0 && changed, "$name: twins exist, the plain kernel runs, outputs equal (twins $twins)")
            elseif must isa Integer && must > 0
                err = con_relerr(a, b)
                want_stage = name in ADJ_STAGE ? 1 : 0
                check(twins == must && firsts == 1 && ran == must && ranfirst && nstage == want_stage && ranstage == want_stage && changed && err <= 1e-12 && isempty(a.conflicts), "$name: twins $twins (want $must), stage $nstage (want $want_stage), ran $ran, error $(round(err, sigdigits = 3)), conflicts $(length(a.conflicts))")
            else
                same_text = STADE.io_expr_to_source(a.plan.host) == STADE.io_expr_to_source(b.plan.host) && twins == 0 && firsts == 0
                check(same_text && ran == 0, "$name: refused, plan unchanged (twins $twins)")
            end
        catch e
            check(false, "$name: error " * first(split(sprint(showerror, e), "\n")))
        end
    end
    bad += adj_direct_checks()
    # thresholds
    try
        name, src, entry, mk, _ = ADJ_CASES[1]
        args = mk(1)                       # n = 70, m = 67, k = 19
        above = con_run(src, entry, args, 19)
        below = con_run(src, entry, args, 20)
        check(count(adj_is_twin, above.launched) == 2 && count(adj_is_twin, below.launched) == 0, "$name: the twins run at the threshold (19) and not below it (20)")
        k0 = Any[args[1:8]..., 0]
        z = con_run(src, entry, k0, 1)
        zb = con_run(src, entry, k0, 0)
        check(count(adj_is_twin, z.launched) == 0 && con_relerr(z, zb) == 0.0, "$name: an inner length of 0 takes the plain kernel")
        lim = con_run(src, entry, args, 1; maxt = nm -> adj_is_twin(nm) ? 128 : 1024)
        base = con_run(src, entry, args, 0)
        check(count(adj_is_twin, lim.launched) == 0 && con_relerr(lim, base) == 0.0, "$name: a register limit below 256 threads takes the plain kernel")
        # the kernel with a forward stage: a limit on the stage alone, and a size below the threshold, send the whole group to the plain kernel
        sname, ssrc, sentry, smk, _ = first(c for c in ADJ_CASES if c[1] == "a recomputed forward value, a repeated decode and temporaries")
        sargs = smk(1)
        sbase = con_run(ssrc, sentry, sargs, 0)
        slim = con_run(ssrc, sentry, sargs, 1; maxt = nm -> adj_is_stage(nm) ? 128 : 1024)
        check(count(adj_is_twin, slim.launched) == 0 && count(adj_is_stage, slim.launched) == 0 && !any(n -> endswith(n, "_a!"), slim.launched) && con_relerr(slim, sbase) == 0.0, "$sname: a register limit on the stage alone takes the plain kernel")
        # a forward loop shorter than the loop of adds: the stage is below the threshold while the twins are not, so the whole group takes the plain kernel
        fsrc = adj_src(params = ", b0, kf", pre = "s = b0[o]\n for c = 1:kf\n s = s + x[(t - 1) * k + c] * w[(c - 1) * n + o]\n end\n o = mod(idx - 1, n) + 1\n t = div(idx - 1, n) + 1\n gs = tanh(s) * g[idx]",
                       loop = "for c = 1:k\n a1 = x[(t - 1) * k + c]\n a2 = w[(c - 1) * n + o]\n xb[(t - 1) * k + c] = xb[(t - 1) * k + c] + a2 * gs\n wb[(c - 1) * n + o] = wb[(c - 1) * n + o] + a1 * gs\n end")
        fargs = Any[adj_args(1)..., randn(MersenneTwister(99), 70), 10]
        fbase = con_run(fsrc, :con_adj, fargs, 0)
        fall = con_run(fsrc, :con_adj, fargs, 1)
        check(count(adj_is_twin, fall.launched) == 2 && count(adj_is_stage, fall.launched) == 1 && con_relerr(fall, fbase) <= 1e-12, "a shorter forward loop: with the threshold 1 the stage and both twins run")
        fcut = con_run(fsrc, :con_adj, fargs, 19)
        check(count(adj_is_twin, fcut.launched) == 0 && count(adj_is_stage, fcut.launched) == 0 && !any(n -> endswith(n, "_a!"), fcut.launched) && con_relerr(fcut, fbase) == 0.0, "a shorter forward loop: the stage is below the threshold (10 < 19), the whole group takes the plain kernel")
        sbelow = con_run(ssrc, sentry, sargs, 20)
        check(count(adj_is_twin, sbelow.launched) == 0 && count(adj_is_stage, sbelow.launched) == 0 && con_relerr(sbelow, sbase) == 0.0, "$sname: a size below the threshold takes the plain kernel")
    catch e
        check(false, "thresholds: error " * first(split(sprint(showerror, e), "\n")))
    end
    return bad
end

function validate_adjoint_contraction(; phases = (:units, :sabotage), slice = 1:typemax(Int))
    bad = 0
    if :units in phases
        println("-- units")
        bad += adj_unit_checks()
    end
    if :sabotage in phases
        println("-- sabotage")
        check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
        text = read(joinpath(abspath(joinpath(@__DIR__, "..")), "src", "STADE.jl"), String)
        cases = [
            ("a loop with a plain store is chosen as the loop of adds", " && all(st -> cgen_atomic_site(st) !== nothing || cgen_split_is_scalar(st), l.body)), rest)", "), rest)"),
            ("the thread index may be offset", " && first_st.args[2] == :(1 + (__tid - 1))\n        X = n_iter", "\n        X = n_iter"),
            ("the reversed thread index may be offset", " && first_st.args[2] == :(\$Xc + (__tid - 1) * -1)", ""),
            ("the thread count may use a local", "    issubset(cgen_split_vars(X), params) || return nothing\n    pre_rest = pre[2:end]", "    pre_rest = pre[2:end]"),
            ("a branch may stand beside the loop", "    all(okst, pre_rest) && all(okst, post) || return nothing\n", ""),
            ("a held index may be changed", "cgen_split_is_scalar(st) && st.args[2] == Expr(:call, :-, idxv, 1) && count(s2 -> s2 isa Expr && s2.head === :(=) && s2.args[1] === st.args[1], Any[pre_rest..., post...]) == 1 && push!(holds, st.args[1])", "cgen_split_is_scalar(st) && st.args[2] == Expr(:call, :-, idxv, 1) && push!(holds, st.args[1])"),
            ("a decode of another value is accepted", "    isX(e) = e == Expr(:call, :-, idxv, 1) || (e isa Symbol && e in holds)\n", "    isX(e) = true\n"),
            ("a decode size computed in the kernel is accepted", "            issubset(cgen_split_vars(a2), params) || return nothing\n            push!(f === :mod ? cols : divs, (st.args[1], a2))", "            push!(f === :mod ? cols : divs, (st.args[1], a2))"),
            ("the two decodes may use different sizes", "        all(d -> d[1] === tv && d[2] == N, divs) || return nothing", "        all(d -> d[1] === tv, divs) || return nothing"),
            ("the column or the row may be changed after its decode", "        count(st -> st isa Expr && st.head === :(=) && st.args[1] === v, leaves(Any[pre_rest..., post...])) == nd || return nothing\n", ""),
            ("the batch may be changed after its decode", "for (v, nd) in ((ov, dec.n_col), (tv, dec.n_row), (hv, dec.n_batch))", "for (v, nd) in ((ov, dec.n_col), (tv, dec.n_row))"),
            ("a loop bound computed in the kernel is accepted", "    issubset(cgen_split_vars(lp.lo), params) && issubset(cgen_split_vars(lp.hi), params) || return nothing\n    cv = lp.var\n    (cv in params", "    cv = lp.var\n    (cv in params"),
            ("the loop variable may be a parameter", "(cv in params || cv in (ov, tv, idxv) || cv === hv) && return nothing", "(cv in (ov, tv, idxv) || cv === hv) && return nothing"),
            ("the loop variable may be the column or the row", "(cv in params || cv in (ov, tv, idxv) || cv === hv) && return nothing", "(cv in params || cv === hv) && return nothing"),
            ("the loop variable may be the batch", "(cv in params || cv in (ov, tv, idxv) || cv === hv) && return nothing", "(cv in params || cv in (ov, tv, idxv)) && return nothing"),
            ("a temporary may have a parameter's name", "            tn in params && return nothing\n", ""),
            ("a temporary may be read outside the loop", "            cgen_count_symbol(outside, tn) == 0 || return nothing\n", ""),
            ("a value that is not a product is accepted", "        (v isa Expr && v.head === :call && v.args[1] === :* && length(v.args) == 3) || return nothing\n        a, b = v.args[2], v.args[3]", "        a, b = (v isa Expr && v.head === :call && length(v.args) == 3) ? (v.args[2], v.args[3]) : (v, v)"),
            ("two seeds are accepted", "        g === nothing || g === gs || return nothing\n", ""),
            ("the seed may be the loop variable", "    g === cv && return nothing\n", ""),
            ("the seed may be a parameter", "    any(st -> cgen_split_is_scalar(st) && st.args[1] === g, pre_rest) || return nothing\n", ""),
            ("an operand that is stored is accepted", "(s.F.arr in stored || s.F.arr in targets) && return nothing", "(s.F.arr in targets) && return nothing"),
            ("an operand that is a target is accepted", "(s.F.arr in stored || s.F.arr in targets) && return nothing", "(s.F.arr in stored) && return nothing"),
            ("a target that is read elsewhere is accepted", "        cgen_count_symbol(others, s.site.arr) == 0 || return nothing\n", ""),
            ("the index roles are not checked", "if issubset(vi, union(params, Set([cu, cv]), bat)) && issubset(vw, union(params, Set([cvv, cv]), bat)) && cu in vi && cvv in vw && cv in vi && cv in vw", "if true"),
            ("the target index may ignore the loop variable", " && cu in vi && cvv in vw && cv in vi && cv in vw", " && cu in vi && cvv in vw"),
            ("the whole-rows test is dropped", "extra = Expr(:call, :(==), Expr(:call, :mod, X, hv === nothing ? N : Expr(:call, :*, N, M)), 0)", "extra = true"),
            ("the seed of a batched product ignores the batch", "Expr(:call, :*, Expr(:call, :+, Expr(:call, :*, Expr(:call, :-, hv, 1), M), Expr(:call, :-, tv, 1)), N), ov)", "Expr(:call, :*, Expr(:call, :+, Expr(:call, :*, 0, M), Expr(:call, :-, tv, 1)), N), ov)"),
            ("a batched twin ignores the batch of the target", "bat = hv === nothing ? Set{Symbol}() : Set{Symbol}([hv])\n    trip = ", "bat = Set{Symbol}()\n    trip = "),
            ("the scratch array is too small", "Expr(:call, Expr(:., :CUDA, QuoteNode(:zeros)), :Float64, A.a.count)", "Expr(:call, Expr(:., :CUDA, QuoteNode(:zeros)), :Float64, 1)"),
            ("the seed is stored before it is computed", "body = Any[parts.head, parts.guard, pre_a..., store, post..., emit_return_nothing()]", "body = Any[parts.head, parts.guard, store, pre_a..., post..., emit_return_nothing()]"),
            ("the probe of the register limit is not built", "probe = Expr(:call, t.name, plain.args[2:end]..., d.target)", "probe = Expr(:call, t.name, plain.args[2:end]..., :__ct_g)"),
            ("the stage store goes to the wrong element", "Expr(:(=), Expr(:ref, :__cf_s, idxv), acf[1])", "Expr(:(=), Expr(:ref, :__cf_s, 1), acf[1])"),
            ("the stage is read from the wrong element", "pre_a[fstage.at] = Expr(:(=), fstage.acc, Expr(:ref, :__cf_s, idxv))", "pre_a[fstage.at] = Expr(:(=), fstage.acc, Expr(:ref, :__cf_s, 1))"),
            ("the stage replaces the wrong statement", "(desc = d0, acc = acf[1], at = fi + 1)", "(desc = d0, acc = acf[1], at = fi)"),
            ("the stage is not in the condition", "                        cond = Expr(:&&, A.fs.cond, fits(A.fs, fprobe))\n", "                        cond = Expr(:&&, A.fs.cond, true)\n"),
            ("the stage size test is dropped", "                        cond = Expr(:&&, A.fs.cond, fits(A.fs, fprobe))\n", "                        cond = fits(A.fs, fprobe)\n"),
            ("the descending loop starts from its start value", "colbase = lp.step == 1 ? lp.lo : lp.hi", "colbase = lp.lo"),
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
                    include($(repr(joinpath(@__DIR__, "validate_adjoint_contraction.jl"))))
                    n = adj_unit_checks()
                    println("RESULT ", n)
                """
                out = try read(Base.pipeline(`$(Base.julia_cmd()) --startup-file=no -e $script`; stderr = devnull), String) catch e; "ERR" end
                m = match(r"RESULT (\d+)", out)
                caught = out == "ERR" || m === nothing || parse(Int, m.captures[1]) > 0
                check(caught, "sabotage \"$label\": " * (caught ? "a check fails" : "NOT CAUGHT"))
            end
        end
    end
    println("\n", bad == 0 ? "all adjoint contraction checks passed" : "*** $bad adjoint contraction check(s) failed ***")
    return bad
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    exit(validate_adjoint_contraction() == 0 ? 0 : 1)
end
