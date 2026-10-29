# flag_two_phase(x, flag, n, out)
#
# A branch on an integer array element, then a rewrite of the array, then the same branch again.
#
# x: input array of length n
# flag: integer array of length n; overwritten with its complement
# n: number of entries
# out: length-1 output array; out[1] receives the sum
function flag_two_phase(x, flag, n, out)
    s = 0.0
    for i = 1:n
        if flag[i] > 0
            s += x[i] * x[i] * x[i]
        end
    end
    for i = 1:n
        flag[i] = 1 - flag[i]
    end
    for i = 1:n
        if flag[i] > 0
            s += x[i] * x[i]
        end
    end
    out[1] = s
    return nothing
end
