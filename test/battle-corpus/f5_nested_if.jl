# f5_nested_if
# Nested branches inside an independent loop.
# LOOP i1 I K
function f5_nested_if(x, y, n)
    for i1 = 1:n
        if x[i1] > 0.0
            if x[i1] > 1.0
                y[i1] = 1.0
            else
                y[i1] = x[i1]
            end
        else
            y[i1] = 0.0
        end
    end
    return nothing
end
