# f6_early_exit
# Counted inner loop with an early exit through the while condition.
# LOOP i1 I K
# WHILE j1 T
function f6_early_exit(a, lim, y, n, m)
    for i1 = 1:n
        s1 = 0.0
        j1 = 1
        while j1 <= m
            s1 = s1 + a[i1, j1]
            if s1 > lim[1]
                j1 = m + 1
            else
                j1 = j1 + 1
            end
        end
        y[i1] = s1
    end
    return nothing
end
