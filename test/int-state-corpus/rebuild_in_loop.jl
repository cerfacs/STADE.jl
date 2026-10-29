# rebuild_in_loop(x, idx, n, steps, out)
#
# An index array rebuilt at the top of every outer pass, then used through a scalar copy in the same pass.
# The first pass must see the array as it was built for the first pass.
#
# x: input array of length n
# idx: integer array of length n; rebuilt each pass
# n: length of x and idx
# steps: number of outer passes
# out: length-1 output array; out[1] receives the sum
function rebuild_in_loop(x, idx, n, steps, out)
    s = 0.0
    for t = 1:steps
        for i = 1:n
            idx[i] = mod(i + t, n) + 1
        end
        for i = 1:n
            j = idx[i]
            s += x[j] * x[i]
        end
    end
    out[1] = s
    return nothing
end
