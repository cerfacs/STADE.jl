# pv_a1(acc, x, y, z, out, n)
#
# The same with a scalar ARGUMENT. Defect of v0.5.17 and v0.5.18: the argument is boxed for the first loop and the second loop's kernel receives the box where it reads a plain value.
#
# acc: scalar argument, start value of the accumulation
# x: input of length n
# y: output of length n
# z: output of length n
# out: length-1 output
# n: number of points
function pv_a1(acc, x, y, z, out, n)
    for i = 1:n
        y[i] = 3.0 * x[i]
        acc += y[i]
    end
    for i = 1:n
        z[i] = y[i] * acc
    end
    out[1] = acc
    return nothing
end
