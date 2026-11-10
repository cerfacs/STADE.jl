# f4_seq_inner_ind
# Outer S: array u is read by one inner loop and accumulated by the other.
# LOOP t1 S H
# LOOP i1 I K
# LOOP i2 I K
function f4_seq_inner_ind(u, v, nt, n)
    for t1 = 1:nt
        for i1 = 1:n
            v[i1] = v[i1] + 0.1 * u[i1]
        end
        for i2 = 1:n
            u[i2] = u[i2] + v[i2]
        end
    end
    return nothing
end
