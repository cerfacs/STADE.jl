import Pkg
haskey(Pkg.project().dependencies, "CUDA") || Pkg.add("CUDA")
using CUDA
using LinearAlgebra
CUDA.allowscalar(false)

function windowed_relax_retire_d_cuda(u, ud, f, fd, w0, num_passes, dx, dxd, n, nd)
    wd = 0.0
    w = w0
    dx2d = dx * dxd + dx * dxd
    dx2 = dx * dx
    for i_seq_pass = 1:num_passes
        if mod(i_seq_pass, 2) == 0
            wd = 0.0
            w = w - 1
        end
        for i_seq_j = 1:w
            leftd = 0.0
            left = 0.0
            if i_seq_j > 1
                CUDA.@allowscalar begin
                        leftd = ud[i_seq_j - 1]
                        left = u[i_seq_j - 1]
                    end
            end
            rightd = 0.0
            right = 0.0
            if i_seq_j < n
                CUDA.@allowscalar begin
                        rightd = ud[i_seq_j + 1]
                        right = u[i_seq_j + 1]
                    end
            end
            CUDA.@allowscalar begin
                    ud[i_seq_j] = 0.5 * (((f[i_seq_j] * dx2d + dx2 * fd[i_seq_j]) + leftd) + rightd)
                    u[i_seq_j] = 0.5 * (dx2 * f[i_seq_j] + left + right)
                end
        end
    end
    return nothing
end

function windowed_relax_retire_cuda(u, f, w0, num_passes, dx, n)
    w = w0
    dx2 = dx * dx
    for i_seq_pass = 1:num_passes
        if mod(i_seq_pass, 2) == 0
            w = w - 1
        end
        for i_seq_j = 1:w
            left = 0.0
            if i_seq_j > 1
                CUDA.@allowscalar begin
                        left = u[i_seq_j - 1]
                    end
            end
            right = 0.0
            if i_seq_j < n
                CUDA.@allowscalar begin
                        right = u[i_seq_j + 1]
                    end
            end
            CUDA.@allowscalar begin
                    u[i_seq_j] = 0.5 * (dx2 * f[i_seq_j] + left + right)
                end
        end
    end
    return nothing
end
