# f1_gather
# Gather read, own-index write.
# LOOP i1 I K
function f1_gather(x, y, idx, n)
    for i1 = 1:n
        j1 = idx[i1]
        y[i1] = x[j1] * 2.0
    end
    return nothing
end
