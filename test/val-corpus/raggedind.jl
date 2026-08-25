# raggedind(out, u, x, n, m0)
#
# Per-row weighted sum whose row length grows with the row index, squared.
# The inner loop's bound is reassigned inside an iteration-independent outer
# loop, so the inner trip count varies per outer iteration with no ancestor
# sequential loop anywhere.
#
# out: output array of length n, filled in place
# u: input array of length n
# x: input array of length m0 + n
# n: number of rows
# m0: base row length
function raggedind(out, u, x, n, m0)
    # outer loop is iteration-independent: row i_x writes only out[i_x]
    for i_x = 1:n
        w = m0 + i_x
        s = 0.0
        # running sum over this row's own length
        for i_j = 1:w
            s = s + x[i_j] * u[i_x]
        end
        out[i_x] = s * s
    end
    return nothing
end
