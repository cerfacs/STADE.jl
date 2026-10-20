# if_top_scalar(x, y, i_n)
#
# An `if` outside any loop gives the scalar `s` its value; a later loop reads `s` nonlinearly. A scalar assigned in an `if` and read
# elsewhere is a shape the backward sweep handles apart from a plain assignment. This kernel places it outside any loop body; v0.4.9 passed
# it, and it stays as a witness because the handling of that shape changed in 0.4.10.
#
# x: input, length >= max(2, i_n)
# y: output accumulator, length i_n, updated in place
# i_n: loop length
function if_top_scalar(x, y, i_n)
    if x[1] > 0.0
        s = x[1] * x[2]
    else
        s = x[2] * x[2]
    end
    for i = 1:i_n
        y[i] = y[i] + s * s * x[i]
    end
    return nothing
end
