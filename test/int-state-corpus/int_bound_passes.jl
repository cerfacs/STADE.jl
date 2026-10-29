# int_bound_passes(x, cnt, steps, out)
#
# The same inside an outer loop, so every pass enters the inner loop with a bound that the previous pass left behind.
#
# x: input array
# cnt: integer array of length 3; cnt[2] is the bound
# steps: number of passes
# out: length-1 output array; out[1] receives the sum
function int_bound_passes(x, cnt, steps, out)
    s = 0.0
    for t = 1:steps
        for i = 1:cnt[2]
            s += x[i] * x[i]
            cnt[2] = mod(cnt[2] + 1, 4) + 1
        end
    end
    out[1] = s
    return nothing
end
