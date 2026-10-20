# if_in_outer_loop(x, y, i_n)
#
# The `if` that gives `s` its value sits in an outer loop body, and an inner loop in the same body reads `s` nonlinearly. See
# if_top_scalar.
#
# x: input, length >= max(3, i_n)
# y: output accumulator, length i_n, updated in place
# i_n: inner loop length
function if_in_outer_loop(x, y, i_n)
    for i_k = 1:2
        s = 0.0
        if i_n > 2
            s = x[i_k] * x[i_k + 1]
        end
        for i = 1:i_n
            y[i] = y[i] + s * s * x[i]
        end
    end
    return nothing
end
