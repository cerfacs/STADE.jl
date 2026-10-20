# retire_if(x, y, i_n, i_nlev, i_nrep, out)
#
# A ragged (Tier B) kernel whose width is retired AFTER the loops that use it, where those loops are inside an `if` that
# is inside a repeat loop. The only loop that reads `cur` is two statement levels below the level body. See retire_nested.
#
# x: input field, length >= i_n
# y: accumulator, length >= i_n, updated in place
# i_n: finest width
# i_nlev: number of passes
# i_nrep: repeats per pass
# out: single-element output
function retire_if(x, y, i_n, i_nlev, i_nrep, out)
    cur = i_n
    for i_l = 1:i_nlev
        for i_r = 1:i_nrep
            if i_r > 1
                for i = 1:cur
                    t = x[i] * x[i]
                    y[i] = y[i] + t * t
                end
            else
                for i = 1:cur
                    y[i] = y[i] + x[i]
                end
            end
        end
        cur = div(cur + 1, 2)
    end
    out[1] = y[1]
    return nothing
end
