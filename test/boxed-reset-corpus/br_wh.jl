# br_wh(p, rs, s, out, n)
#
# A while loop whose body resets a boxed reduction and reads it nonlinearly. The loop halves rs at least once per pass, so it ends.
#
# p: vector of length n
# rs: length-1 array, numerator and loop variable
# s: length-1 array for the dot product
# out: length-1 output
# n: length
function br_wh(p, rs, s, out, n)
    while rs[1] > 0.01
        s[1] = 0.0
        for i = 1:n
            s[1] += p[i] * p[i]
        end
        out[1] = rs[1] / (s[1] + 1.0)
        rs[1] = 0.5 * out[1]
    end
    return nothing
end
