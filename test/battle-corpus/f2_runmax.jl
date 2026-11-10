# f2_runmax
# Running maximum (max is not a plus-accumulation).
# LOOP i1 S H
function f2_runmax(x, y, n)
    m1 = x[1]
    for i1 = 1:n
        m1 = max(m1, x[i1])
        y[i1] = m1
    end
    return nothing
end
