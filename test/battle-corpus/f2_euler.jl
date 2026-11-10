# f2_euler
# Explicit time stepping.
# LOOP i1 S H
function f2_euler(x, dt, nt)
    for i1 = 1:nt
        x[i1 + 1] = x[i1] + dt * x[i1] * (1.0 - x[i1])
    end
    return nothing
end
