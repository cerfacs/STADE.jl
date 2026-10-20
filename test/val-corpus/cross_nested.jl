# cross_nested(x, w, z, loss, i_n, i_nc)
#
# A nested loop writes the scalar between two top-level writes of the enclosing body, with reads before it, inside it and after it. The
# nested loop is itself loop-local and could drop its first-write save, but that save is what brings back the enclosing body's first value
# for the read between them. A scalar written inside a loop that also writes it at its top level is refused.
#
# x: length i_n + i_nc + 2, w: length i_nc + 2, z: length i_n, loss: length 1
function cross_nested(x, w, z, loss, i_n, i_nc)
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            xv = x[t + ci - 1]
            a = a + xv * xv * w[ci]
            for cj = 1:2
                xv = x[t + ci + cj - 1] * w[cj]
                a = a + xv * xv
            end
            xv = x[t + ci - 1] * 0.5
            a = a + xv * xv * w[ci + 1]
        end
        z[t] = a
    end
    for m = 1:i_n
        loss[1] = loss[1] + z[m] * z[m]
    end
    return nothing
end
