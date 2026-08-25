# raggedii(out, u, x, n, m0)
#
# Per-row sum whose inner segment length is reassigned inside an
# iteration-independent middle loop, two levels under the outer loop.
#
# out: output array of length n, filled in place
# u: input array of length n
# x: input array of length m0 + 2
# n: number of rows
# m0: base segment length
function raggedii(out, u, x, n, m0)
    # outer loop is iteration-independent: row i_x writes only out[i_x]
    for i_x = 1:n
        s = 0.0
        # middle loop is iteration-independent, yet it retires w each pass
        for i_y = 1:2
            w = m0 + i_y
            # running sum over this segment's own length
            for i_j = 1:w
                s = s + x[i_j] * u[i_x]
            end
        end
        out[i_x] = s * s
    end
    return nothing
end
