function dotprod(loss, u, v, i_n)
    for i_x = 1:i_n
        loss[1] = loss[1] + u[i_x] * v[i_x]
    end
end
