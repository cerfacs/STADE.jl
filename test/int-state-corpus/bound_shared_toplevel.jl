# bound_shared_toplevel(x, n, m, out)
#
# One scalar holds the bound of two loops, and is assigned between them at the top level of the kernel. The reverse sweep
# must reverse the first loop with the first value.
#
# x: input array of length max(n, m)
# n: bound of the first loop
# m: bound of the second loop
# out: length-1 output array; out[1] receives the sum
function bound_shared_toplevel(x, n, m, out)
    s = 0.0
    c = n
    for i = 1:c
        s += x[i] * x[i]
    end
    c = m
    for i = 1:c
        s += x[i] * x[i] * x[i]
    end
    out[1] = s
    return nothing
end
