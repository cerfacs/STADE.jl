# f5_in_body
# Branch inside an independent loop.
# LOOP i1 I K
function f5_in_body(x, y, n)
    for i1 = 1:n
        if x[i1] > 0.5
            y[i1] = x[i1] * x[i1]
        else
            y[i1] = 0.0
        end
    end
    return nothing
end
