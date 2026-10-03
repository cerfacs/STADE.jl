# red_nested_inner(out, u, v, i_n, i_m)
#
# A reduction nested inside an outer loop. The whole nest offloads as one
# kernel with the inner sum running sequentially inside a thread, which is
# the right answer: lowering the inner loop to a mapreduce per outer
# iteration would launch i_m reductions instead of one kernel.
#
# out: per-row results, length i_m
# u: vector of length i_n
# v: matrix stored row-major, length i_m * i_n
# i_n: row length
# i_m: row count
function red_nested_inner(out, u, v, i_n, i_m)
    for i_j = 1:i_m
        s = 0.0
        for i_k = 1:i_n
            s = s + v[(i_j - 1) * i_n + i_k] * u[i_k]
        end
        out[i_j] = s * s
    end
    return nothing
end
