# retire_in_if(x, y, n, levels, out)
#
# A loop that reads `cur`, then a retiring assignment to `cur`, both inside an `if` that is always true. No top-level statement of the
# level body is the retiring assignment. Through 0.4.9 the sizing pass for keep_push_pop = false looked only at top-level statements,
# wrote each level's table from the value after the retirement, and every block overran its successor (wrong gradient, relative
# error about 0.02 in the adjoint and 0.13 in the dot-product check). The split now comes from the program-order events of the whole body.
#
# x: input field, length >= n
# y: accumulator, length >= n, updated in place
# n: finest level width
# levels: number of coarsening passes
# out: single-element output
function retire_in_if(x, y, n, levels, out)
    cur = n
    for i_l = 1:levels
        if n >= 0
            for i = 1:cur
                if n >= 0
                    t = x[i] * x[i]
                    y[i] = y[i] + t * t
                end
            end
            cur = div(cur + 1, 2)
        end
    end
    out[1] = y[1]
    return nothing
end

