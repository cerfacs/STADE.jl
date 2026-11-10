# f2_chain
# All iterations read element 1 and write the next.
# LOOP i1 S H
function f2_chain(u, n)
    for i1 = 2:n
        u[i1] = u[i1 - 1]
    end
    return nothing
end
