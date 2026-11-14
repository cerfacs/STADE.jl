# pv_a10(x, y, tot, n)
#
# The accumulated scalar is read in the condition of a `while` that runs in each thread.
#
# x: input of length n
# y: output of length n
# tot: length-1 output
# n: number of points
function pv_a10(x, y, tot, n)
    acc = 0.0
    for i = 1:n
        y[i] = x[i] * x[i]
        acc += y[i]
    end
    for i = 1:n
        r = y[i]
        while r > 0.001 * abs(acc)
            r = 0.5 * r
        end
        y[i] = r
    end
    tot[1] = acc
    return nothing
end
