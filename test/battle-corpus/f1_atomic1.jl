# f1_atomic1
# Accumulation at a fixed index.
# LOOP i1 I K
function f1_atomic1(x, loss, n)
    for i1 = 1:n
        loss[1] = loss[1] + x[i1] * x[i1]
    end
    return nothing
end
