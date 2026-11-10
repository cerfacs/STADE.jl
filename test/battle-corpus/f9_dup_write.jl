# f9_dup_write
# Every iteration writes element 1.
# LOOP i1 S H
function f9_dup_write(x, y, n)
    for i1 = 1:n
        y[1] = x[i1]
    end
    return nothing
end
