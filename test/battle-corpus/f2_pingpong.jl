# f2_pingpong
# Two arrays read each other one step apart.
# LOOP i1 S H
function f2_pingpong(a, b, n)
    for i1 = 2:n
        a[i1] = b[i1 - 1]
        b[i1] = a[i1] * 2.0
    end
    return nothing
end
