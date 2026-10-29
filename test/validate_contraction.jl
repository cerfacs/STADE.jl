# ==================== validate_contraction.jl =============================
# Checks the contraction twin (`contraction`, v0.4.34): a tiled kernel for a loop over the elements of a matrix whose inner loop sums products.
#
#   julia validate_contraction.jl                  units, refusals, sabotage
#   validate_contraction(; phases = (:units,))
#
# Every case is a small kernel in the house style, run through the CPU shim of validate_ownership (ownership_shim.jl: blocks of tasks, barriers, shared memory,
# a record of who touched what) twice, with the twin forced on (`contraction = 1`) and with `contraction = 0`, from the same random inputs.
#   - a case that must tile: a twin exists, the two runs agree to 1e-12 (the sum order differs), the shim reports no conflict, and the twin ran;
#   - a case that must not tile: no twin exists and the two plans are the very same text.
# The sizes are not multiples of the 64 x 64 tile or of the 16-wide step of the inner loop, and the grid has several blocks in both directions.
# SABOTAGE edits of the contraction code, each of which one case must catch.

isdefined(Main, :STADE) || include(joinpath(@__DIR__, "..", "src", "STADE.jl"))
using Random

# (name, source, entry, args, must_tile)   must_tile = :plain: a twin is built but the host test keeps the plain kernel
# sizes: n columns, m rows, k inner length
function con_args(r; n = 70, m = 67, k = 19)
    (r = r, n = n, m = m, k = k)
end
const CON_CASES = [
    ("bias product", """function con_bias(x, w, b, y, m, n, k)
        for idx = 1:m * n
            o = mod(idx - 1, n) + 1
            t = div(idx - 1, n) + 1
            s = b[o]
            for c = 1:k
                s = s + x[(t - 1) * k + c] * w[(c - 1) * n + o]
            end
            y[idx] = s
        end
        return nothing
    end""", :con_bias, sd -> (r = MersenneTwister(sd); n = 70; m = 67; k = 19; Any[randn(r, m * k), randn(r, k * n), randn(r, n), zeros(m * n), m, n, k]), true),
    ("residual, tanh and two stores", """function con_res(x, w, b, res, y, z, m, n, k)
        for idx = 1:m * n
            o = mod(idx - 1, n) + 1
            t = div(idx - 1, n) + 1
            s = b[o] + res[idx]
            for c = 1:k
                s = s + x[(t - 1) * k + c] * w[(c - 1) * n + o]
            end
            y[idx] = tanh(s)
            z[idx] = s
        end
        return nothing
    end""", :con_res, sd -> (r = MersenneTwister(sd); n = 66; m = 65; k = 33; Any[randn(r, m * k), randn(r, k * n), randn(r, n), randn(r, m * n), zeros(m * n), zeros(m * n), m, n, k]), true),
    ("subtracted product", """function con_sub(x, w, y, m, n, k)
        for idx = 1:m * n
            o = mod(idx - 1, n) + 1
            t = div(idx - 1, n) + 1
            s = 0.5
            for c = 1:k
                s = s - x[(t - 1) * k + c] * w[(c - 1) * n + o]
            end
            y[idx] = s
        end
        return nothing
    end""", :con_sub, sd -> (r = MersenneTwister(sd); n = 71; m = 66; k = 17; Any[randn(r, m * k), randn(r, k * n), zeros(m * n), m, n, k]), true),
    ("operands in the other order", """function con_swap(x, w, y, m, n, k)
        for idx = 1:m * n
            o = mod(idx - 1, n) + 1
            t = div(idx - 1, n) + 1
            s = 0.0
            for c = 1:k
                s = s + w[(c - 1) * n + o] * x[(t - 1) * k + c]
            end
            y[idx] = s
        end
        return nothing
    end""", :con_swap, sd -> (r = MersenneTwister(sd); n = 70; m = 67; k = 21; Any[randn(r, m * k), randn(r, k * n), zeros(m * n), m, n, k]), true),
    ("loop starts at a parameter", """function con_lo(x, w, y, m, n, k, c0)
        for idx = 1:m * n
            o = mod(idx - 1, n) + 1
            t = div(idx - 1, n) + 1
            s = 0.0
            for c = c0:k
                s = s + x[(t - 1) * k + c] * w[(c - 1) * n + o]
            end
            y[idx] = s
        end
        return nothing
    end""", :con_lo, sd -> (r = MersenneTwister(sd); n = 70; m = 67; k = 25; Any[randn(r, m * k), randn(r, k * n), zeros(m * n), m, n, k, 4]), true),
    ("one row", """function con_row(x, w, y, m, n, k)
        for idx = 1:m * n
            o = mod(idx - 1, n) + 1
            t = div(idx - 1, n) + 1
            s = 0.0
            for c = 1:k
                s = s + x[(t - 1) * k + c] * w[(c - 1) * n + o]
            end
            y[idx] = s
        end
        return nothing
    end""", :con_row, sd -> (r = MersenneTwister(sd); n = 70; m = 1; k = 19; Any[randn(r, m * k), randn(r, k * n), zeros(m * n), m, n, k]), true),
    ("same array on both sides", """function con_gram(x, y, m, k)
        for idx = 1:m * m
            o = mod(idx - 1, m) + 1
            t = div(idx - 1, m) + 1
            s = 0.0
            for c = 1:k
                s = s + x[(t - 1) * k + c] * x[(o - 1) * k + c]
            end
            y[idx] = s
        end
        return nothing
    end""", :con_gram, sd -> (r = MersenneTwister(sd); m = 69; k = 19; Any[randn(r, m * k), zeros(m * m), m, k]), true),
    ("left operand read across the inner index", """function con_tn(x, w, y, m, n, k)
        for idx = 1:m * n
            o = mod(idx - 1, n) + 1
            t = div(idx - 1, n) + 1
            s = 0.0
            for c = 1:k
                s = s + x[(c - 1) * m + t] * w[(c - 1) * n + o]
            end
            y[idx] = s
        end
        return nothing
    end""", :con_tn, sd -> (r = MersenneTwister(sd); n = 70; m = 67; k = 19; Any[randn(r, m * k), randn(r, k * n), zeros(m * n), m, n, k]), true),
    ("the start value is read from an array of the output index", """function con_start(x, w, b, y, m, n, k)
        for idx = 1:m * n
            o = mod(idx - 1, n) + 1
            t = div(idx - 1, n) + 1
            s = b[idx] * 2.0
            for c = 1:k
                s = s + x[(t - 1) * k + c] * w[(c - 1) * n + o]
            end
            y[idx] = s
        end
        return nothing
    end""", :con_start, sd -> (r = MersenneTwister(sd); n = 70; m = 67; k = 19; Any[randn(r, m * k), randn(r, k * n), randn(r, m * n), zeros(m * n), m, n, k]), true),
    # ---- refusals ----
    ("left operand depends on the column", """function con_adep(x, w, y, m, n, k)
        for idx = 1:m * n
            o = mod(idx - 1, n) + 1
            t = div(idx - 1, n) + 1
            s = 0.0
            for c = 1:k
                s = s + x[(o - 1) * k + c] * w[(c - 1) * n + o]
            end
            y[idx] = s
        end
        return nothing
    end""", :con_adep, sd -> (r = MersenneTwister(sd); n = 70; m = 67; k = 19; Any[randn(r, max(n, m) * k), randn(r, k * n), zeros(m * n), m, n, k]), false),
    ("right operand depends on the row", """function con_bdep(x, w, y, m, n, k)
        for idx = 1:m * n
            o = mod(idx - 1, n) + 1
            t = div(idx - 1, n) + 1
            s = 0.0
            for c = 1:k
                s = s + x[(t - 1) * k + c] * w[(c - 1) * n + o + t]
            end
            y[idx] = s
        end
        return nothing
    end""", :con_bdep, sd -> (r = MersenneTwister(sd); n = 70; m = 67; k = 19; Any[randn(r, m * k), randn(r, k * n + m), zeros(m * n), m, n, k]), false),
    ("a parameter has the loop variable's name and is read after the loop", """function con_cafter(x, w, y, m, n, k, c)
        for idx = 1:m * n
            o = mod(idx - 1, n) + 1
            t = div(idx - 1, n) + 1
            s = 0.0
            for c = 1:k
                s = s + x[(t - 1) * k + c] * w[(c - 1) * n + o]
            end
            y[idx] = s + c
        end
        return nothing
    end""", :con_cafter, sd -> (r = MersenneTwister(sd); n = 70; m = 67; k = 19; Any[randn(r, m * k), randn(r, k * n), zeros(m * n), m, n, k, 0.25]), false),
    ("a parameter has the loop variable's name and is read before the loop", """function con_cbefore(x, w, y, m, n, k, c)
        for idx = 1:m * n
            o = mod(idx - 1, n) + 1
            t = div(idx - 1, n) + 1
            s = c
            for c = 1:k
                s = s + x[(t - 1) * k + c] * w[(c - 1) * n + o]
            end
            y[idx] = s
        end
        return nothing
    end""", :con_cbefore, sd -> (r = MersenneTwister(sd); n = 70; m = 67; k = 19; Any[randn(r, m * k), randn(r, k * n), zeros(m * n), m, n, k, 0.25]), false),
    ("sum of one array, no product", """function con_sum(x, y, m, n, k)
        for idx = 1:m * n
            o = mod(idx - 1, n) + 1
            t = div(idx - 1, n) + 1
            s = 0.0
            for c = 1:k
                s = s + x[(t - 1) * k + c]
            end
            y[idx] = s
        end
        return nothing
    end""", :con_sum, sd -> (r = MersenneTwister(sd); n = 70; m = 67; k = 19; Any[randn(r, m * k), zeros(m * n), m, n, k]), false),
    ("product of three", """function con_three(x, w, v, y, m, n, k)
        for idx = 1:m * n
            o = mod(idx - 1, n) + 1
            t = div(idx - 1, n) + 1
            s = 0.0
            for c = 1:k
                s = s + x[(t - 1) * k + c] * w[(c - 1) * n + o] * v[c]
            end
            y[idx] = s
        end
        return nothing
    end""", :con_three, sd -> (r = MersenneTwister(sd); n = 70; m = 67; k = 19; Any[randn(r, m * k), randn(r, k * n), randn(r, k), zeros(m * n), m, n, k]), false),
    ("a partial last row", """function con_part(x, w, y, m, n, k)
        for idx = 1:m * n + 5
            o = mod(idx - 1, n) + 1
            t = div(idx - 1, n) + 1
            s = 0.0
            for c = 1:k
                s = s + x[(t - 1) * k + c] * w[(c - 1) * n + o]
            end
            y[idx] = s
        end
        return nothing
    end""", :con_part, sd -> (r = MersenneTwister(sd); n = 70; m = 67; k = 19; Any[randn(r, (m + 1) * k), randn(r, k * n), zeros(m * n + 5), m, n, k]), :plain),
    ("a factor that is a scalar", """function con_scal(x, w, y, m, n, k, a)
        for idx = 1:m * n
            o = mod(idx - 1, n) + 1
            t = div(idx - 1, n) + 1
            s = 0.0
            for c = 1:k
                s = s + x[(t - 1) * k + c] * a
            end
            y[idx] = s
        end
        return nothing
    end""", :con_scal, sd -> (r = MersenneTwister(sd); n = 70; m = 67; k = 19; Any[randn(r, m * k), randn(r, k * n), zeros(m * n), m, n, k, 0.5]), false),
    ("a factor that is a function of an element", """function con_fn(x, w, y, m, n, k)
        for idx = 1:m * n
            o = mod(idx - 1, n) + 1
            t = div(idx - 1, n) + 1
            s = 0.0
            for c = 1:k
                s = s + tanh(x[(t - 1) * k + c]) * w[(c - 1) * n + o]
            end
            y[idx] = s
        end
        return nothing
    end""", :con_fn, sd -> (r = MersenneTwister(sd); n = 70; m = 67; k = 19; Any[randn(r, m * k), randn(r, k * n), zeros(m * n), m, n, k]), false),
    ("two inner loops", """function con_two(x, w, y, m, n, k)
        for idx = 1:m * n
            o = mod(idx - 1, n) + 1
            t = div(idx - 1, n) + 1
            s = 0.0
            for c = 1:k
                s = s + x[(t - 1) * k + c] * w[(c - 1) * n + o]
            end
            for c = 1:k
                s = s + x[(t - 1) * k + c] * w[(c - 1) * n + o]
            end
            y[idx] = s
        end
        return nothing
    end""", :con_two, sd -> (r = MersenneTwister(sd); n = 70; m = 67; k = 19; Any[randn(r, m * k), randn(r, k * n), zeros(m * n), m, n, k]), false),
    ("branch in the inner loop", """function con_branch(x, w, y, m, n, k)
        for idx = 1:m * n
            o = mod(idx - 1, n) + 1
            t = div(idx - 1, n) + 1
            s = 0.0
            for c = 1:k
                if x[(t - 1) * k + c] > 0.0
                    s = s + x[(t - 1) * k + c] * w[(c - 1) * n + o]
                end
            end
            y[idx] = s
        end
        return nothing
    end""", :con_branch, sd -> (r = MersenneTwister(sd); n = 70; m = 67; k = 19; Any[randn(r, m * k), randn(r, k * n), zeros(m * n), m, n, k]), false),
    ("descending inner loop", """function con_down(x, w, y, m, n, k)
        for idx = 1:m * n
            o = mod(idx - 1, n) + 1
            t = div(idx - 1, n) + 1
            s = 0.0
            for c = k:-1:1
                s = s + x[(t - 1) * k + c] * w[(c - 1) * n + o]
            end
            y[idx] = s
        end
        return nothing
    end""", :con_down, sd -> (r = MersenneTwister(sd); n = 70; m = 67; k = 19; Any[randn(r, m * k), randn(r, k * n), zeros(m * n), m, n, k]), false),
    ("operand array stored in the kernel", """function con_stored(x, w, y, m, n, k)
        for idx = 1:m * n
            o = mod(idx - 1, n) + 1
            t = div(idx - 1, n) + 1
            s = 0.0
            for c = 1:k
                s = s + x[(t - 1) * k + c] * w[(c - 1) * n + o]
            end
            x[(t - 1) * k + 1] = s
        end
        return nothing
    end""", :con_stored, sd -> (r = MersenneTwister(sd); n = 70; m = 67; k = 19; Any[randn(r, m * k), randn(r, k * n), zeros(m * n), m, n, k]), false),
    ("matrix-vector loop, no flat index", """function con_mv(w, x, y, n, k)
        for o = 1:n
            s = 0.0
            for c = 1:k
                s = s + w[(c - 1) * n + o] * x[c]
            end
            y[o] = s
        end
        return nothing
    end""", :con_mv, sd -> (r = MersenneTwister(sd); n = 70; k = 19; Any[randn(r, k * n), randn(r, k), zeros(n), n, k]), false),
    ("inner bound computed in the kernel", """function con_kin(x, w, y, m, n, k)
        for idx = 1:m * n
            o = mod(idx - 1, n) + 1
            t = div(idx - 1, n) + 1
            s = 0.0
            kk = k - 2
            for c = 1:kk
                s = s + x[(t - 1) * k + c] * w[(c - 1) * n + o]
            end
            y[idx] = s
        end
        return nothing
    end""", :con_kin, sd -> (r = MersenneTwister(sd); n = 70; m = 67; k = 19; Any[randn(r, m * k), randn(r, k * n), zeros(m * n), m, n, k]), false),
    # products batched over a third index (heads), v0.4.36: the flat index decodes into a column, a row and a batch
    ("batched product", """function con_bat(q, kk, y, nb, m, n, k)
        for idx = 1:nb * m * n
            o = mod(idx - 1, n) + 1
            t = mod(div(idx - 1, n), m) + 1
            h = div(idx - 1, m * n) + 1
            s = 0.0
            for c = 1:k
                s = s + q[((h - 1) * m + t - 1) * k + c] * kk[((h - 1) * n + o - 1) * k + c]
            end
            y[idx] = exp(0.1 * s)
        end
        return nothing
    end""", :con_bat, sd -> (r = MersenneTwister(sd); nb = 3; n = 70; m = 67; k = 19; Any[randn(r, nb * m * k), randn(r, nb * n * k), zeros(nb * m * n), nb, m, n, k]), true),
    ("batched product, one array with strided heads", """function con_bat2(qkv, y, nh, tt, dh, scale)
        d = nh * dh
        d3 = 3 * d
        for idx = 1:nh * tt * tt
            u = mod(idx - 1, tt) + 1
            t = mod(div(idx - 1, tt), tt) + 1
            h = div(idx - 1, tt * tt) + 1
            ss = 0.0
            for c = 1:dh
                ss = ss + qkv[(t - 1) * d3 + (h - 1) * dh + c] * qkv[(u - 1) * d3 + d + (h - 1) * dh + c]
            end
            y[idx] = exp(ss * scale)
        end
        return nothing
    end""", :con_bat2, sd -> (r = MersenneTwister(sd); nh = 3; tt = 70; dh = 17; Any[randn(r, tt * 3 * nh * dh), zeros(nh * tt * tt), nh, tt, dh, 0.1]), true),
    ("batched product, divisor written the other way", """function con_bat3(q, kk, y, nb, m, n, k)
        for idx = 1:nb * m * n
            o = mod(idx - 1, n) + 1
            t = mod(div(idx - 1, n), m) + 1
            h = div(idx - 1, n * m) + 1
            s = 1.0
            for c = 1:k
                s = s - q[((h - 1) * m + t - 1) * k + c] * kk[((h - 1) * n + o - 1) * k + c]
            end
            y[idx] = s
        end
        return nothing
    end""", :con_bat3, sd -> (r = MersenneTwister(sd); nb = 2; n = 66; m = 65; k = 33; Any[randn(r, nb * m * k), randn(r, nb * n * k), zeros(nb * m * n), nb, m, n, k]), true),
    ("batched, the batch size is not rows times columns", """function con_batn(q, kk, y, nb, m, n, k)
        for idx = 1:nb * m * n
            o = mod(idx - 1, n) + 1
            t = mod(div(idx - 1, n), m) + 1
            h = div(idx - 1, n * n) + 1
            s = 0.0
            for c = 1:k
                s = s + q[((h - 1) * m + t - 1) * k + c] * kk[((h - 1) * n + o - 1) * k + c]
            end
            y[idx] = s
        end
        return nothing
    end""", :con_batn, sd -> (r = MersenneTwister(sd); nb = 3; n = 70; m = 67; k = 19; Any[randn(r, nb * m * k + 4000), randn(r, nb * n * k + 4000), zeros(nb * m * n), nb, m, n, k]), false),
    ("batched, the row decode uses another size than the column", """function con_batr(q, kk, y, nb, m, n, k)
        for idx = 1:nb * m * n
            o = mod(idx - 1, n) + 1
            t = mod(div(idx - 1, n + 1), m) + 1
            h = div(idx - 1, m * n) + 1
            s = 0.0
            for c = 1:k
                s = s + q[((h - 1) * m + t - 1) * k + c] * kk[((h - 1) * n + o - 1) * k + c]
            end
            y[idx] = s
        end
        return nothing
    end""", :con_batr, sd -> (r = MersenneTwister(sd); nb = 3; n = 70; m = 67; k = 19; Any[randn(r, nb * m * k + 4000), randn(r, nb * n * k + 4000), zeros(nb * m * n), nb, m, n, k]), false),
    ("batched, the batch is changed after its decode", """function con_batc(q, kk, y, nb, m, n, k)
        for idx = 1:nb * m * n
            o = mod(idx - 1, n) + 1
            t = mod(div(idx - 1, n), m) + 1
            h = div(idx - 1, m * n) + 1
            s = 0.0
            h = h + 0
            for c = 1:k
                s = s + q[((h - 1) * m + t - 1) * k + c] * kk[((h - 1) * n + o - 1) * k + c]
            end
            y[idx] = s
        end
        return nothing
    end""", :con_batc, sd -> (r = MersenneTwister(sd); nb = 3; n = 70; m = 67; k = 19; Any[randn(r, nb * m * k), randn(r, nb * n * k), zeros(nb * m * n), nb, m, n, k]), false),
    ("batched, a partial last batch", """function con_batp(q, kk, y, nb, m, n, k)
        for idx = 1:nb * m * n + 7
            o = mod(idx - 1, n) + 1
            t = mod(div(idx - 1, n), m) + 1
            h = div(idx - 1, m * n) + 1
            s = 0.0
            for c = 1:k
                s = s + q[((h - 1) * m + t - 1) * k + c] * kk[((h - 1) * n + o - 1) * k + c]
            end
            y[idx] = s
        end
        return nothing
    end""", :con_batp, sd -> (r = MersenneTwister(sd); nb = 3; n = 70; m = 67; k = 19; Any[randn(r, (nb + 1) * m * k), randn(r, (nb + 1) * n * k), zeros(nb * m * n + 7), nb, m, n, k]), :plain),
    ("batched, a product of three factors", """function con_batx(q, kk, hh, y, nb, m, n, k)
        for idx = 1:nb * m * n
            o = mod(idx - 1, n) + 1
            t = mod(div(idx - 1, n), m) + 1
            h = div(idx - 1, m * n) + 1
            s = 0.0
            for c = 1:k
                s = s + q[((h - 1) * m + t - 1) * k + c] * kk[((h - 1) * n + o - 1) * k + c] * hh[1]
            end
            y[idx] = s
        end
        return nothing
    end""", :con_batx, sd -> (r = MersenneTwister(sd); nb = 3; n = 70; m = 67; k = 19; Any[randn(r, nb * m * k), randn(r, nb * n * k), randn(r, 1), zeros(nb * m * n), nb, m, n, k]), false),
    # products whose head is a part of the column index (v0.4.37): the flat index decodes into a column and a row, the head is `div(column - 1, dh) + 1`
    ("head in the column", """function con_hc(sc, v, den, y, d, dh, tt)
        for idx = 1:tt * d
            c = mod(idx - 1, d) + 1
            t = div(idx - 1, d) + 1
            h = div(c - 1, dh) + 1
            s = 0.0
            for u = 1:tt
                s = s + sc[((h - 1) * tt + t - 1) * tt + u] * v[(u - 1) * d + c]
            end
            y[idx] = s / den[(h - 1) * tt + t]
        end
        return nothing
    end""", :con_hc, sd -> (r = MersenneTwister(sd); nh = 3; tt = 70; dh = 17; d = nh * dh; Any[randn(r, nh * tt * tt), randn(r, tt * d), rand(r, nh * tt) .+ 1.0, zeros(tt * d), d, dh, tt]), true),
    ("head in the column, one array with strided columns", """function con_hc2(sc, qkv, den, y, d, dh, tt)
        d3 = 3 * d
        for idx = 1:tt * d
            c = mod(idx - 1, d) + 1
            t = div(idx - 1, d) + 1
            h = div(c - 1, dh) + 1
            s = 0.0
            for u = 1:tt
                s = s + sc[((h - 1) * tt + t - 1) * tt + u] * qkv[(u - 1) * d3 + 2d + c]
            end
            y[idx] = s / den[(h - 1) * tt + t]
        end
        return nothing
    end""", :con_hc2, sd -> (r = MersenneTwister(sd); nh = 3; tt = 66; dh = 25; d = nh * dh; Any[randn(r, nh * tt * tt), randn(r, tt * 3 * d), rand(r, nh * tt) .+ 1.0, zeros(tt * d), d, dh, tt]), true),
    ("head in the column, an epilogue that reads the head and the column", """function con_hc3(sc, v, w2, den, y, d, dh, tt)
        for idx = 1:tt * d
            c = mod(idx - 1, d) + 1
            t = div(idx - 1, d) + 1
            h = div(c - 1, dh) + 1
            s = 1.0
            for u = 1:tt
                s = s - sc[((h - 1) * tt + t - 1) * tt + u] * v[(u - 1) * d + c]
            end
            y[idx] = s * w2[c] + den[(h - 1) * tt + t]
        end
        return nothing
    end""", :con_hc3, sd -> (r = MersenneTwister(sd); nh = 4; tt = 33; dh = 20; d = nh * dh; Any[randn(r, nh * tt * tt), randn(r, tt * d), randn(r, d), rand(r, nh * tt) .+ 1.0, zeros(tt * d), d, dh, tt]), true),
    ("head in the column, the operands do not read the head", """function con_hc4(sc, v, den, y, d, dh, tt)
        for idx = 1:tt * d
            c = mod(idx - 1, d) + 1
            t = div(idx - 1, d) + 1
            h = div(c - 1, dh) + 1
            s = 0.0
            for u = 1:tt
                s = s + sc[(t - 1) * tt + u] * v[(u - 1) * d + c]
            end
            y[idx] = s / den[(h - 1) * tt + t]
        end
        return nothing
    end""", :con_hc4, sd -> (r = MersenneTwister(sd); nh = 3; tt = 70; dh = 17; d = nh * dh; Any[randn(r, tt * tt), randn(r, tt * d), rand(r, nh * tt) .+ 1.0, zeros(tt * d), d, dh, tt]), true),
    ("head in the column, the head width does not divide the width", """function con_hc5(sc, v, den, y, d, dh, tt)
        for idx = 1:tt * d
            c = mod(idx - 1, d) + 1
            t = div(idx - 1, d) + 1
            h = div(c - 1, dh) + 1
            s = 0.0
            for u = 1:tt
                s = s + sc[((h - 1) * tt + t - 1) * tt + u] * v[(u - 1) * d + c]
            end
            y[idx] = s / den[(h - 1) * tt + t]
        end
        return nothing
    end""", :con_hc5, sd -> (r = MersenneTwister(sd); nh = 3; tt = 70; dh = 17; d = 50; Any[randn(r, nh * tt * tt), randn(r, tt * d), rand(r, nh * tt) .+ 1.0, zeros(tt * d), d, dh, tt]), :plain),
    ("head in the column, the head is changed after its decode", """function con_hc6(sc, v, den, y, d, dh, tt)
        for idx = 1:tt * d
            c = mod(idx - 1, d) + 1
            t = div(idx - 1, d) + 1
            h = div(c - 1, dh) + 1
            s = 0.0
            h = h + 0
            for u = 1:tt
                s = s + sc[((h - 1) * tt + t - 1) * tt + u] * v[(u - 1) * d + c]
            end
            y[idx] = s / den[(h - 1) * tt + t]
        end
        return nothing
    end""", :con_hc6, sd -> (r = MersenneTwister(sd); nh = 3; tt = 70; dh = 17; d = nh * dh; Any[randn(r, nh * tt * tt), randn(r, tt * d), rand(r, nh * tt) .+ 1.0, zeros(tt * d), d, dh, tt]), false),
    ("head in the column, the head is derived from the row", """function con_hc8(sc, v, den, y, d, dh, tt)
        for idx = 1:tt * d
            c = mod(idx - 1, d) + 1
            t = div(idx - 1, d) + 1
            h = div(t - 1, dh) + 1
            s = 0.0
            for u = 1:tt
                s = s + sc[((h - 1) * tt + t - 1) * tt + u] * v[(u - 1) * d + c]
            end
            y[idx] = s / den[(h - 1) * tt + t]
        end
        return nothing
    end""", :con_hc8, sd -> (r = MersenneTwister(sd); nh = 5; tt = 70; dh = 17; d = 51; Any[randn(r, nh * tt * tt), randn(r, tt * d), rand(r, nh * tt) .+ 1.0, zeros(tt * d), d, dh, tt]), false),
    ("head in the column, the head width is assigned in the kernel", """function con_hc9(sc, v, den, y, d, nh, tt)
        for idx = 1:tt * d
            dw = div(d, nh)
            c = mod(idx - 1, d) + 1
            t = div(idx - 1, d) + 1
            h = div(c - 1, dw) + 1
            s = 0.0
            for u = 1:tt
                s = s + sc[((h - 1) * tt + t - 1) * tt + u] * v[(u - 1) * d + c]
            end
            y[idx] = s / den[(h - 1) * tt + t]
        end
        return nothing
    end""", :con_hc9, sd -> (r = MersenneTwister(sd); nh = 3; tt = 70; dh = 17; d = nh * dh; Any[randn(r, nh * tt * tt), randn(r, tt * d), rand(r, nh * tt) .+ 1.0, zeros(tt * d), d, nh, tt]), false),
    ("head in the column, two different head decodes", """function con_hc10(sc, v, den, y, d, dh, tt)
        for idx = 1:tt * d
            c = mod(idx - 1, d) + 1
            t = div(idx - 1, d) + 1
            h = div(c - 1, dh) + 1
            g = div(c - 1, tt) + 1
            s = 0.0
            for u = 1:tt
                s = s + sc[((h - 1) * tt + t - 1) * tt + u] * v[(u - 1) * d + c]
            end
            y[idx] = s / den[(h - 1) * tt + t] + 0.0 * g
        end
        return nothing
    end""", :con_hc10, sd -> (r = MersenneTwister(sd); nh = 3; tt = 70; dh = 17; d = nh * dh; Any[randn(r, nh * tt * tt), randn(r, tt * d), rand(r, nh * tt) .+ 1.0, zeros(tt * d), d, dh, tt]), false),
    ("head in the column, the loop variable has the head's name", """function con_hc11(sc, v, den, y, d, dh, tt)
        for idx = 1:tt * d
            c = mod(idx - 1, d) + 1
            t = div(idx - 1, d) + 1
            h = div(c - 1, dh) + 1
            s = 0.0 * den[(h - 1) * tt + t]
            for h = 1:tt
                s = s + sc[(t - 1) * tt + h] * v[(h - 1) * d + c]
            end
            y[idx] = s
        end
        return nothing
    end""", :con_hc11, sd -> (r = MersenneTwister(sd); nh = 3; tt = 70; dh = 17; d = nh * dh; Any[randn(r, tt * tt), randn(r, tt * d), rand(r, nh * tt) .+ 1.0, zeros(tt * d), d, dh, tt]), false),
    ("head in the column, a head width computed from parameters", """function con_hc7(sc, v, den, y, d, nh, tt)
        for idx = 1:tt * d
            c = mod(idx - 1, d) + 1
            t = div(idx - 1, d) + 1
            h = div(c - 1, div(d, nh)) + 1
            s = 0.0
            for u = 1:tt
                s = s + sc[((h - 1) * tt + t - 1) * tt + u] * v[(u - 1) * d + c]
            end
            y[idx] = s / den[(h - 1) * tt + t]
        end
        return nothing
    end""", :con_hc7, sd -> (r = MersenneTwister(sd); nh = 3; tt = 70; dh = 17; d = nh * dh; Any[randn(r, nh * tt * tt), randn(r, tt * d), rand(r, nh * tt) .+ 1.0, zeros(tt * d), d, nh, tt]), true),
]

# The recognizer on hand-written kernel text, for the guards that STADE's own gates make unreachable from source: a store into an operand array, and a parameter that has the
# loop variable's name. Each variant must be refused, the control must not.
con_kernel(pre, post; extra = "", first = "idx = 1 + (__tid - 1)", dec = "o = mod(idx - 1, n) + 1\n    t = div(idx - 1, n) + 1") = Meta.parse("""function cuda_kernel_dir!(x, w, y, m, n, k, c$extra)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(m * n - 1, 1) + 1
        return nothing
    end
    $first
    $dec
    $pre
    for c = 1:k
        s = s + x[(t - 1) * k + c] * w[(c - 1) * n + o]
    end
    $post
    return nothing
end""")
const CON_DIRECT = [
    ("control", "s = 0.0", "y[idx] = s", true),
    ("a store into an operand array", "s = 0.0", "x[idx] = s", false),
    ("a store into the other operand array", "s = 0.0", "w[idx] = s", false),
    ("the loop variable's name read after the loop", "s = 0.0", "y[idx] = s + c", false),
    ("the loop variable's name read before the loop", "s = c", "y[idx] = s", false),
    ("the column is changed after its decode", "s = 0.0\n    o = o + 1", "y[idx] = s", false),
    ("the row is changed in the epilogue", "s = 0.0", "t = t + 1\n    y[idx] = s", false),
    ("an atomic store in the epilogue", "s = 0.0", "CUDA.@atomic y[idx] += s", false),
    ("the flat index is not the thread index", "s = 0.0", "y[idx] = s", false),
    ("a decode through a held index", "s = 0.0", "y[idx] = s", true, "h0 = idx - 1\n    o = mod(h0, n) + 1\n    t = div(h0, n) + 1"),
    ("the held index is changed between the decodes", "s = 0.0", "y[idx] = s", false, "h0 = idx - 1\n    o = mod(h0, n) + 1\n    h0 = h0 + 1\n    t = div(h0, n) + 1"),
]
# The same for a head in the column: the head is not the loop variable and no parameter has its name. STADE's own gates leave neither reachable from source
# (a head that is not read before the loop is dropped as dead, a parameter that is assigned is renamed), so the recognizer gets the kernel text.
con_head_kernel(; loopvar = "u", head = "h", extra = "") = Meta.parse("""function cuda_kernel_dirh!(sc, v, y, d, dh, tt$extra)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(tt * d - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
    c = mod(idx - 1, d) + 1
    t = div(idx - 1, d) + 1
    $head = div(c - 1, dh) + 1
    s = 0.0
    for $loopvar = 1:tt
        s = s + sc[(($head - 1) * tt + t - 1) * tt + $loopvar] * v[($loopvar - 1) * d + c]
    end
    y[idx] = s
    return nothing
end""")
const CON_HEAD_DIRECT = [
    ("head in the column, control", con_head_kernel(), true),
    # refused by two guards, the head test and the test that the loop variable is not read before the loop (the head's own decode mentions it); only the second can be sabotaged alone, so the head test has no sabotage case
    ("head in the column, the loop variable has the head's name", con_head_kernel(loopvar = "h"), false),
    ("head in the column, a parameter has the head's name", con_head_kernel(extra = ", h"), false),
]
function con_direct_checks()
    bad = 0
    for (name, kd, must) in CON_HEAD_DIRECT
        t = try STADE.cgen_contraction_kernel(kd) catch e; e end
        ok = must ? (t isa NamedTuple) : t === nothing
        println(rpad(ok ? "ok" : "FAIL", 6), "direct, $name: ", must ? "a twin is built" : "refused")
        ok || (bad += 1)
    end
    for (name, pre, post, must, dec...) in CON_DIRECT
        kd = con_kernel(pre, post; first = startswith(name, "the flat index") ? "idx = 2 + (__tid - 1)" : "idx = 1 + (__tid - 1)", dec = isempty(dec) ? "o = mod(idx - 1, n) + 1\n    t = div(idx - 1, n) + 1" : dec[1])
        t = try STADE.cgen_contraction_kernel(kd) catch e; e end
        ok = must ? (t isa NamedTuple) : t === nothing
        println(rpad(ok ? "ok" : "FAIL", 6), "direct, $name: ", must ? "a twin is built" : "refused")
        ok || (bad += 1)
    end
    return bad
end

function con_init!()
    isdefined(Main, :CUDA) || include(joinpath(@__DIR__, "ownership_shim.jl"))
    isdefined(Main, :CuArray) || Core.eval(Main, :(const CuArray = CUDA.TArray))
    Core.eval(Main, :(using .CUDA))
end

# builds the plan, evaluates its functions into Main, runs the host function; returns (outputs, kernels launched, conflicts, plan)
function con_run(src::String, entry::Symbol, args, contraction::Int; maxt = nothing)
    con_init!()
    return Base.invokelatest(con_run_body, src, entry, args, contraction, maxt)
end
function con_run_body(src::String, entry::Symbol, args, contraction::Int, maxt)
    Main.CUDA.MAXT[] = maxt === nothing ? (name -> 1024) : maxt
    cuda = STADE.cgen_backend_cuda()
    plan = STADE.stade_gpu(Meta.parse(src), cuda; warp_atomics = false, shared_private = false, inbounds = false, split_reduce = 0, contraction = contraction)
    for k in plan.kernels; Core.eval(Main, k); end
    Core.eval(Main, plan.host)
    host = plan.host.args[1].args[1]
    dev = Any[a isa AbstractArray ? Main.CUDA.TArray(copy(a)) : a for a in args]
    empty!(Main.CUDA.LAUNCHED); empty!(Main.CUDA.VIOLATIONS)
    Base.invokelatest(getfield(Main, host), dev...)
    outs = Any[a isa Main.CUDA.TArray ? copy(a.data) : a for a in dev]
    return (outs = outs, launched = copy(Main.CUDA.LAUNCHED), conflicts = copy(Main.CUDA.VIOLATIONS), plan = plan)
end

con_relerr(a, b) = maximum(maximum(abs.(x .- y) ./ max.(abs.(x), abs.(y), 1.0)) for (x, y) in zip(a.outs, b.outs) if x isa AbstractArray; init = 0.0)

function con_unit_checks()
    bad = 0
    check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
    for (name, src, entry, mk, must) in CON_CASES
        try
            args = mk(1)
            a = con_run(src, entry, args, 1)
            b = con_run(src, entry, args, 0)
            twins = count(k -> endswith(string(k.args[1].args[1]), "_t!"), a.plan.kernels)
            ran = any(n -> endswith(n, "_t!"), a.launched)
            if must === :plain
                # a twin exists but the host test sends the launch to the plain kernel (a loop that does not make whole rows)
                check(twins >= 1 && !ran && con_relerr(a, b) == 0.0 && any(x != y for (x, y) in zip(a.outs, args) if x isa AbstractArray), "$name: a twin exists, the plain kernel runs, outputs equal (twins $twins)")
            elseif must
                err = con_relerr(a, b)
                changed = any(x != y for (x, y) in zip(a.outs, args) if x isa AbstractArray)
                check(twins >= 1 && ran && changed && err <= 1e-12 && isempty(a.conflicts), "$name: twins $twins, ran $ran, error $(round(err, sigdigits = 3)), conflicts $(length(a.conflicts))")
            else
                same_text = STADE.io_expr_to_source(a.plan.host) == STADE.io_expr_to_source(b.plan.host) && twins == 0
                check(same_text && !ran, "$name: refused, plan unchanged (twins $twins)")
            end
        catch e
            check(false, "$name: error " * first(split(sprint(showerror, e), "\n")))
        end
    end
    bad += con_direct_checks()
    # thresholds. A product of 67 x 70 outputs with an inner length of 19 is 4 tiles. The number of tiles that the twin needs depends on the layout of the operands:
    # cld(contraction, 8) for "nn", cld(contraction, 32) for "nt" and cld(contraction, 4) for "tn" (4 tiles: 32, 128 and 16). A side or the inner length below
    # min(contraction, 16) takes the plain kernel. An inner length of 0 does too; the answer is the start value.
    try
        ran(a) = any(n -> endswith(n, "_t!"), a.launched)
        name, src, entry, mk, _ = CON_CASES[1]
        args = mk(1)                       # n = 70, m = 67, k = 19
        check(ran(con_run(src, entry, args, 32)) && !ran(con_run(src, entry, args, 33)), "$name (nn): 4 tiles run the twin at 32 and not at 33")
        argk = Any[args[1:end-1]..., 15]
        check(ran(con_run(src, entry, argk, 15)) && !ran(con_run(src, entry, argk, 16)), "$name: an inner length of 15 runs the twin at 15 and not at 16")
        argm = Any[args[1:end-3]..., 15, args[end-1:end]...]
        check(ran(con_run(src, entry, argm, 15)) && !ran(con_run(src, entry, argm, 16)), "$name: 15 rows run the twin at 15 and not at 16")
        argn = Any[args[1:end-2]..., 15, args[end]]
        check(ran(con_run(src, entry, argn, 15)) && !ran(con_run(src, entry, argn, 16)), "$name: 15 columns run the twin at 15 and not at 16")
        gname, gsrc, gentry, gmk, _ = first(c for c in CON_CASES if c[1] == "same array on both sides")
        gargs = gmk(1)                     # 69 x 69 outputs, k = 19
        check(ran(con_run(gsrc, gentry, gargs, 128)) && !ran(con_run(gsrc, gentry, gargs, 129)), "$gname (nt): 4 tiles run the twin at 128 and not at 129")
        tname, tsrc, tentry, tmk, _ = first(c for c in CON_CASES if c[1] == "left operand read across the inner index")
        targs = tmk(1)
        check(ran(con_run(tsrc, tentry, targs, 16)) && !ran(con_run(tsrc, tentry, targs, 17)), "$tname (tn): 4 tiles run the twin at 16 and not at 17")
        k0 = Any[args[1:end-1]..., 0]
        z = con_run(src, entry, k0, 1)
        zb = con_run(src, entry, k0, 0)
        check(!ran(z) && con_relerr(z, zb) == 0.0, "$name: an inner length of 0 takes the plain kernel")
        lim = con_run(src, entry, args, 1; maxt = nm -> endswith(nm, "_t!") ? 128 : 1024)
        base = con_run(src, entry, args, 0)
        check(!ran(lim) && con_relerr(lim, base) == 0.0, "$name: a register limit below 256 threads takes the plain kernel")
        # the layout of an operand and the count of tiles, on hand-made descriptors
        u(e, v) = STADE.cgen_contraction_unit(e, v)
        check(u(:((t - 1) * k + c), :c) && u(:c, :c) && u(:(c - 1 + (t - 1) * k), :c) && u(:(1 + (t - 1) * k + (c - 1)), :c), "the layout test: a summand that is the inner index moves by 1")
        check(!u(:((c - 1) * n + o), :c) && !u(:(2 * c), :c) && !u(:(c + c), :c) && !u(:(c - c), :c) && !u(:(k - c), :c) && !u(:(-c), :c) && !u(:(t * k + o), :c), "the layout test: anything else does not")
        dsc(a, b) = (A = (idx = a,), B = (idx = b,), kvar = :c)
        nn_d = dsc(:((t - 1) * k + c), :((c - 1) * n + o)); nt_d = dsc(:((t - 1) * k + c), :((o - 1) * k + c))
        tn_d = dsc(:((c - 1) * m + t), :((c - 1) * n + o)); tt_d = dsc(:((c - 1) * m + t), :((o - 1) * k + c))
        tiles(d, c) = STADE.cgen_contraction_tiles(d, c)
        check((tiles(nn_d, 256), tiles(nt_d, 256), tiles(tn_d, 256), tiles(tt_d, 256)) == (32, 8, 64, 64), "the tiles at 256: nn 32, nt 8, tn 64, tt 64 ($((tiles(nn_d, 256), tiles(nt_d, 256), tiles(tn_d, 256), tiles(tt_d, 256))))")
        check((tiles(nn_d, 1), tiles(nt_d, 1), tiles(tn_d, 1), tiles(tt_d, 1)) == (1, 1, 1, 1) && tiles(nn_d, 512) == 64 && tiles(tn_d, 1000) == 250 && tiles(nt_d, 257) == 9, "the tiles scale with contraction, at least 1")
    catch e
        check(false, "thresholds: error " * first(split(sprint(showerror, e), "\n")))
    end
    # a product batched over heads: the rows must reach `contraction`, the columns and the inner length only min(contraction, 8); the grid has at most 65535 batches
    try
        name, src, entry, mk, _ = first(c for c in CON_CASES if c[1] == "batched product")
        args = mk(1)                       # nb = 3, m = 67 rows, n = 70 columns, k = 19
        ran(a) = any(n -> endswith(n, "_t!"), a.launched)
        check(ran(con_run(src, entry, args, 67)) && !ran(con_run(src, entry, args, 68)), "$name: the twin runs at the row threshold (67) and not below it (68)")
        argk = Any[args[1:end-1]..., 7]
        check(ran(con_run(src, entry, argk, 1)) && !ran(con_run(src, entry, argk, 8)), "$name: an inner length of 7 runs the twin at contraction 1 and not at 8")
        argn = Any[args[1:end-2]..., 7, args[end]]
        check(ran(con_run(src, entry, argn, 1)) && !ran(con_run(src, entry, argn, 8)), "$name: 7 columns run the twin at contraction 1 and not at 8")
        # an inner length of 9 is below one tile step of 16 and above the batched minimum of 8: the twin runs and agrees with the plain kernel
        argk9 = Any[args[1:end-1]..., 9]
        r9 = con_run(src, entry, argk9, 8); b9 = con_run(src, entry, argk9, 0)
        check(ran(r9) && con_relerr(r9, b9) <= 1e-12, "$name: an inner length of 9 runs the twin at contraction 8 and agrees (error $(round(con_relerr(r9, b9), sigdigits = 3)))")
        hostsrc = STADE.io_expr_to_source(con_run(src, entry, args, 1).plan.host)
        plainsrc = STADE.io_expr_to_source(con_run(CON_CASES[1][2], CON_CASES[1][3], CON_CASES[1][4](1), 1).plan.host)
        check(occursin("<= 65535", hostsrc) && !occursin("65535", plainsrc), "$name: the host test has the grid limit of 65535 batches, and the unbatched twin does not")
        zb = Any[args[1:end-4]..., 0, args[end-2:end]...]
        z = con_run(src, entry, zb, 1); zbase = con_run(src, entry, zb, 0)
        check(!ran(z) && con_relerr(z, zbase) == 0.0, "$name: zero batches take the plain kernel")
    catch e
        check(false, "batched thresholds: error " * first(split(sprint(showerror, e), "\n")))
    end
    # a head in the column: the tiles are batched over the heads, `dh` columns each; the rows or the columns of one head must reach `contraction`
    try
        name4, src4, entry4, mk4, _ = first(c for c in CON_CASES if c[1] == "head in the column, the operands do not read the head")
        host4 = STADE.io_expr_to_source(con_run(src4, entry4, mk4(1), 1).plan.host)
        check(!occursin("div(d, dh)", host4) && !occursin("mod(d, dh)", host4), "$name4: the tiles are not batched over heads")
    catch e
        check(false, "head in the column, unread head: error " * first(split(sprint(showerror, e), "\n")))
    end
    try
        name, src, entry, mk, _ = first(c for c in CON_CASES if c[1] == "head in the column")
        args = mk(1)                       # nh = 3 heads of dh = 17 columns, tt = 70 rows
        ran(a) = any(n -> endswith(n, "_t!"), a.launched)
        hostsrc = STADE.io_expr_to_source(con_run(src, entry, args, 1).plan.host)
        check(occursin("div(d, dh)", hostsrc) && occursin("mod(d, dh) == 0", hostsrc) && occursin("dh > 0 &&", hostsrc), "$name: the host test and the grid count heads, and the head width is tested before it divides")
        check(ran(con_run(src, entry, args, 70)) && !ran(con_run(src, entry, args, 71)), "$name: the twin runs at 70 rows (the rows reach contraction) and not at 71")
        r = MersenneTwister(5); tt = 9; nh = 3; dh = 17; d = nh * dh
        argw = Any[randn(r, nh * tt * tt), randn(r, tt * d), rand(r, nh * tt) .+ 1.0, zeros(tt * d), d, dh, tt]
        check(ran(con_run(src, entry, argw, 17)) && !ran(con_run(src, entry, argw, 18)), "$name: with 9 rows and 17 columns per head the twin runs at contraction 17 (the columns reach it) and not at 18")
        argh = Any[randn(r, 5 * 70 * 70), randn(r, 70 * 51), rand(r, 5 * 70) .+ 1.0, zeros(70 * 51), 51, 12, 70]       # 51 columns are not a whole number of heads of 12
        check(!ran(con_run(src, entry, argh, 1)), "$name: a head width that does not divide the width takes the plain kernel")
        w = con_run(src, entry, argw, 17); wb = con_run(src, entry, argw, 0)
        check(con_relerr(w, wb) <= 1e-12 && any(!iszero, w.outs[4]), "$name: the twin agrees with the plain kernel on 9 rows (error $(round(con_relerr(w, wb), sigdigits = 3)))")
    catch e
        check(false, "head in the column thresholds: error " * first(split(sprint(showerror, e), "\n")))
    end
    return bad
end

function validate_contraction(; phases = (:units, :sabotage), slice = 1:typemax(Int))
    bad = 0
    if :units in phases
        println("-- units")
        bad += con_unit_checks()
    end
    if :sabotage in phases
        println("-- sabotage")
        check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
        text = read(joinpath(abspath(joinpath(@__DIR__, "..")), "src", "STADE.jl"), String)
        cases = [
            ("no barrier after the loads", "Expr(:block, loadA..., loadB..., call(:sync_threads),", "Expr(:block, loadA..., loadB...,"),
            ("no barrier after the products", "Expr(:for, asg(:__ct_kk, call(:(:), 0, TK - 1)), Expr(:block, mac...)),\n                      call(:sync_threads)))", "Expr(:for, asg(:__ct_kk, call(:(:), 0, TK - 1)), Expr(:block, mac...))))"),
            ("the tile is padded with ones", "zfill = asg(:__ct_v, 0.0)", "zfill = asg(:__ct_v, 1.0)"),
            ("the row of a load is not bounded", "Expr(:&&, call(:<, cadd(:__ct_r0, :__ct_rr), :__ct_M), call(:<, cadd(:__ct_k0, :__ct_kk), :__ct_K))", "call(:<, cadd(:__ct_k0, :__ct_kk), :__ct_K)"),
            ("the column of a load is not bounded", "Expr(:&&, call(:<, cadd(:__ct_c0, :__ct_cc), :__ct_N), call(:<, cadd(:__ct_k0, :__ct_kk), :__ct_K))", "call(:<, cadd(:__ct_k0, :__ct_kk), :__ct_K)"),
            ("the inner index of a load is not bounded", "Expr(:&&, call(:<, cadd(:__ct_r0, :__ct_rr), :__ct_M), call(:<, cadd(:__ct_k0, :__ct_kk), :__ct_K))", "call(:<, cadd(:__ct_r0, :__ct_rr), :__ct_M)"),
            ("the outputs are not bounded", "Expr(:if, Expr(:&&, call(:<, :__ct_row, :__ct_M), call(:<, :__ct_col, :__ct_N)), Expr(:block, inner...))", "Expr(:if, true, Expr(:block, inner...))"),
            ("the loop start is ignored", "push!(loadA, asg(d.kvar, cadd(cadd(d.kbase, :__ct_k0), :__ct_kk)))", "push!(loadA, asg(d.kvar, cadd(cadd(1, :__ct_k0), :__ct_kk)))"),
            ("a descending loop is accepted", "(lp !== nothing && lp.step == 1 && length(lp.body) == 1) || return nothing", "(lp !== nothing && length(lp.body) == 1) || return nothing"),
            ("operands that depend on the wrong index are accepted", "issubset(va, union(params, Set([tv, cv]), bat)) && issubset(vb, union(params, Set([ov, cv]), bat)) || return nothing", "true"),
            ("an operand of the product that is stored is accepted", "(fa.arr in stored || fb.arr in stored) && return nothing", "false && return nothing"),
            ("the loop variable may be read after the loop", "    for st in post\n        cv in cgen_split_vars(st) && return nothing\n    end\n", ""),
            ("the loop variable may be read before the loop", "    any(cv in cgen_split_vars(st) for st in pre_rest) && return nothing\n", ""),
            ("an inner bound computed in the kernel is accepted", "issubset(cgen_split_vars(lp.lo), params) && issubset(cgen_split_vars(lp.hi), params) || return nothing\n    ac = ", "true\n    ac = "),
            ("a loop after the first is allowed in the epilogue", "        (cgen_split_is_scalar(st) || (cgen_split_is_store(st) && cgen_atomic_site(st) === nothing)) || return nothing", "        true"),
            ("a product test is dropped", "(term isa Expr && term.head === :call && term.args[1] === :* && length(term.args) == 3) || return nothing", "true"),
            ("a factor that is not an array element is accepted", "(f isa Expr && f.head === :ref && length(f.args) == 2 && f.args[1] isa Symbol) || return nothing", "true"),
            ("the column or row may be reassigned", "        count(st -> st isa Expr && st.head === :(=) && st.args[1] === v, Any[pre_rest..., post...]) == 1 || return nothing\n", ""),
            ("a held index may be changed", "        st.args[2] == Expr(:call, :-, idxv, 1) && count(s2 -> s2 isa Expr && s2.head === :(=) && s2.args[1] === st.args[1], Any[pre_rest..., post...]) == 1 && push!(holds, st.args[1])", "        st.args[2] == Expr(:call, :-, idxv, 1) && push!(holds, st.args[1])"),
            ("the flat index may be offset", "(first_st isa Expr && first_st.head === :(=) && first_st.args[1] isa Symbol && first_st.args[2] == :(1 + (__tid - 1))) || return nothing", "(first_st isa Expr && first_st.head === :(=) && first_st.args[1] isa Symbol) || return nothing"),
            ("the register limit is not checked", "    fits(t, call) = Expr(:call, :>=, Expr(:call, Expr(:., :CUDA, QuoteNode(:maxthreads)), Expr(:macrocall, backend.launch_macro, nothing, Expr(:(=), :launch, false), call)), 256)", "    fits(t, call) = true"),
            ("the whole-rows test is dropped", "extra = Expr(:call, :(==), Expr(:call, :mod, n_iter, hv === nothing ? N : Expr(:call, :*, N, M)), 0))", "extra = nothing)"),
            ("the columns are not tested against the floor", "call(:>=, d.Nh, min(cmin, 16))", "call(:>, d.Nh, 0)"),
            ("the rows are not tested against the floor", "call(:>=, d.Mh, min(cmin, 16))", "call(:>, d.Mh, 0)"),
            ("the inner length is not tested against the floor", "call(:>=, d.Kh, min(cmin, 16))", "call(:>, d.Kh, 0)"),
            ("the number of tiles is not tested", "want = call(:>=, tiles, cgen_contraction_tiles(d, cmin))", "want = nothing"),
            ("the tiles are counted by the rows only", "call(Expr(:., :Base, QuoteNode(:cld)), d.Mh, TM), call(Expr(:., :Base, QuoteNode(:cld)), d.Nh, TM))\n        cond", "call(Expr(:., :Base, QuoteNode(:cld)), d.Mh, TM), 1)\n        cond"),
            ("every layout needs the count of nn", "    base = au && bu ? 8 : au ? 32 : 64\n", "    base = 32\n"),
            ("a layout with the right operand along the inner index needs the count of nt", "    base = au && bu ? 8 : au ? 32 : 64\n", "    base = bu ? 8 : au ? 32 : 64\n"),
            ("the count of tiles does not scale with contraction", "    return cld(base * cmin, 256)", "    return base"),
            ("two summands that move with the inner index count as one", "        return n == 1 && any(a -> cgen_contraction_unit(a, v), ex.args[2:end])", "        return any(a -> cgen_contraction_unit(a, v), ex.args[2:end])"),
            ("a difference that subtracts the inner index counts as moving by 1", "        return cgen_contraction_unit(ex.args[2], v) && !has(ex.args[3])", "        return cgen_contraction_unit(ex.args[2], v)"),
            ("the launch asks for too little shared memory", "shmem = 8 * nt * TS)", "shmem = 8 * nt * TS - 8)"),
            ("a batched twin ignores the row threshold", ": call(:>=, d.Mh, cmin)\n", ": call(:>, d.Mh, 0)\n"),
            ("a head-derived twin needs the rows to reach the threshold, not the columns", "Expr(:||, call(:>=, d.Mh, cmin), call(:>=, d.Nh, cmin))", "call(:>=, d.Mh, cmin)"),
            ("a head-derived twin needs both the rows and the columns", "Expr(:||, call(:>=, d.Mh, cmin), call(:>=, d.Nh, cmin))", "Expr(:&&, call(:>=, d.Mh, cmin), call(:>=, d.Nh, cmin))"),
            ("the head width is not tested before it divides", "get(d, :dhexpr, nothing) === nothing || (cond = Expr(:&&, call(:>, d.dhexpr, 0), cond))", "nothing"),
            ("the heads must divide the width", "extra = Expr(:&&, Expr(:call, :(==), Expr(:call, :mod, n_iter, N), 0), Expr(:call, :(==), Expr(:call, :mod, N, dbat.dh), 0)))", "extra = Expr(:call, :(==), Expr(:call, :mod, n_iter, N), 0))"),
            ("the column of a head starts at the first column", "colbase = Expr(:call, :+, 1, Expr(:call, :*, :__ct_bz, :__ct_dh))", "colbase = 1"),
            ("the flat index of a head ignores the head", "Expr(:call, :*, row, :__ct_D), Expr(:call, :*, :__ct_bz, :__ct_dh)), col), 1)", "Expr(:call, :*, row, :__ct_D), 0), col), 1)"),
            ("the number of heads is the width", "Bh = Expr(:call, :div, N, dbat.dh)\n        flat =", "Bh = N\n        flat ="),
            ("a head is used when no operand reads it", "derived = dbat !== nothing && hv === nothing && dbat.var in union(cgen_split_vars(fa.idx), cgen_split_vars(fb.idx))", "derived = dbat !== nothing && hv === nothing"),
            ("the head may be a kernel parameter", "(dbat.var === lp.var || dbat.var in params) && return nothing", "dbat.var === lp.var && return nothing"),
            ("the head is not assigned once", "        count(st -> st isa Expr && st.head === :(=) && st.args[1] === dbat.var, Any[pre_rest..., post...]) == 1 || return nothing\n", ""),
            ("a head of another variable is accepted", "length(r.args[2].args) == 3 && r.args[2].args[2] == Expr(:call, :-, ov, 1)) || continue", "length(r.args[2].args) == 3) || continue"),
            ("a head width computed in the kernel is accepted", "            issubset(cgen_split_vars(r.args[2].args[3]), params) || continue\n", ""),
            ("two different heads are accepted", "if !isempty(heads) && all(x -> x[1] === heads[1][1] && x[2] == heads[1][2], heads)", "if !isempty(heads)"),
            ("a batched twin ignores the column threshold", "call(:>=, d.Nh, min(cmin, 8))", "call(:>, d.Nh, 0)"),
            ("a batched twin ignores the inner length threshold", "call(:>=, d.Kh, min(cmin, 8))", "call(:>, d.Kh, 0)"),
            ("the grid limit of a batched twin is dropped", "call(:<=, d.Bh, 65535))", "true)"),
            ("the batch variable is not set in the tile", "    bv === nothing || push!(body, asg(bv, cadd(:__ct_bz, 1)))\n", ""),
            ("the batch index of the output is dropped", "Expr(:call, :*, :__ct_bz, :__ct_M)", "0"),
            ("the batch divisor is not checked", "all(d -> d[1] === hv && (d[2] == Expr(:call, :*, N, M) || d[2] == Expr(:call, :*, M, N)), divs) || return nothing", "all(d -> d[1] === hv, divs) || return nothing"),
            ("the size of the row decode may differ from the column's", "    N2 == N || return nothing\n    all(m -> m[1] === tv && m[2] == N && m[3] == M, rowmods) || return nothing", "    all(m -> m[1] === tv && m[3] == M, rowmods) || return nothing"),
            ("the batch may be reassigned", "for v in (hv === nothing ? (ov, tv) : (ov, tv, hv))", "for v in (ov, tv)"),
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
                    include($(repr(joinpath(@__DIR__, "validate_contraction.jl"))))
                    n = con_unit_checks()
                    println("RESULT ", n)
                """
                out = try read(Base.pipeline(`$(Base.julia_cmd()) --startup-file=no -e $script`; stderr = devnull), String) catch e; "ERR" end
                m = match(r"RESULT (\d+)", out)
                caught = out == "ERR" || m === nothing || parse(Int, m.captures[1]) > 0
                check(caught, "sabotage \"$label\": " * (caught ? "a check fails" : "NOT CAUGHT"))
            end
        end
    end
    println("\n", bad == 0 ? "all contraction checks passed" : "*** $bad contraction check(s) failed ***")
    return bad
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    exit(validate_contraction() == 0 ? 0 : 1)
end
