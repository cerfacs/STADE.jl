function nest_if_scalar_d(u, ud, rhs, rhsd, i_nu, i_n)
    for i_k = 1:i_nu
        if i_nu >= 0
            for i_j = 1:i_n
                if i_nu >= 0
                    leftd = 0.0
                    left = 0.0
                    if i_j > 1
                        leftd = ud[i_j - 1]
                        left = u[i_j - 1]
                    end
                    rightd = 0.0
                    right = 0.0
                    if i_j < i_n
                        rightd = ud[i_j + 1]
                        right = u[i_j + 1]
                    end
                    ud[i_j] = 0.5 * ((rhsd[i_j] + (right * leftd + left * rightd)) + leftd)
                    u[i_j] = 0.5 * (rhs[i_j] + left * right + left)
                end
            end
        end
    end
    return nothing
end

function nest_if_scalar(u, rhs, i_nu, i_n)
    for i_k = 1:i_nu
        if i_nu >= 0
            for i_j = 1:i_n
                if i_nu >= 0
                    left = 0.0
                    if i_j > 1
                        left = u[i_j - 1]
                    end
                    right = 0.0
                    if i_j < i_n
                        right = u[i_j + 1]
                    end
                    u[i_j] = 0.5 * (rhs[i_j] + left * right + left)
                end
            end
        end
    end
end
