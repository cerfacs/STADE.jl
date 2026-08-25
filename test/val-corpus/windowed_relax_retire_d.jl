function windowed_relax_retire_d(u, ud, f, fd, w0, num_passes, dx, dxd, n, nd)
    wd = 0.0
    w = w0
    dx2d = dx * dxd + dx * dxd
    dx2 = dx * dx
    for i_pass = 1:num_passes
        if mod(i_pass, 2) == 0
            wd = 0.0
            w = w - 1
        end
        for i_j = 1:w
            leftd = 0.0
            left = 0.0
            if i_j > 1
                leftd = ud[i_j - 1]
                left = u[i_j - 1]
            end
            rightd = 0.0
            right = 0.0
            if i_j < n
                rightd = ud[i_j + 1]
                right = u[i_j + 1]
            end
            ud[i_j] = 0.5 * (((f[i_j] * dx2d + dx2 * fd[i_j]) + leftd) + rightd)
            u[i_j] = 0.5 * (dx2 * f[i_j] + left + right)
        end
    end
    return nothing
end

function windowed_relax_retire(u, f, w0, num_passes, dx, n)
    w = w0
    dx2 = dx * dx
    for i_pass = 1:num_passes
        if mod(i_pass, 2) == 0
            w = w - 1
        end
        for i_j = 1:w
            left = 0.0
            if i_j > 1
                left = u[i_j - 1]
            end
            right = 0.0
            if i_j < n
                right = u[i_j + 1]
            end
            u[i_j] = 0.5 * (dx2 * f[i_j] + left + right)
        end
    end
end
