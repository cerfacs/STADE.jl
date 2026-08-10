import Pkg
haskey(Pkg.project().dependencies, "JACC") || Pkg.add("JACC")
haskey(Pkg.project().dependencies, "Atomix") || Pkg.add("Atomix")
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_pipeline_d_1!(__jacc_i, i_n, u, ud, v, vd)
    i_x = 1 + (__jacc_i - 1)
    vd[i_x] = (2 * u[i_x]) * ud[i_x]
    v[i_x] = u[i_x] ^ 2 + 1.0
    return nothing
end

function jacc_kernel_pipeline_d_2!(__jacc_i, i_n, u, ud, v, vd, w, wd)
    i_x = 1 + (__jacc_i - 1)
    wd[i_x] = u[i_x] * vd[i_x] + v[i_x] * ud[i_x]
    w[i_x] = v[i_x] * u[i_x]
    return nothing
end

function jacc_kernel_pipeline_1!(__jacc_i, i_n, u, v)
    i_x = 1 + (__jacc_i - 1)
    v[i_x] = u[i_x] ^ 2 + 1.0
    return nothing
end

function jacc_kernel_pipeline_2!(__jacc_i, i_n, u, v, w)
    i_x = 1 + (__jacc_i - 1)
    w[i_x] = v[i_x] * u[i_x]
    return nothing
end

function pipeline_d_jacc(loss, lossd, u, ud, v, vd, w, wd, i_n)
    JACC.parallel_for(div(i_n - 1, 1) + 1, jacc_kernel_pipeline_d_1!, i_n, u, ud, v, vd)
    JACC.parallel_for(div(i_n - 1, 1) + 1, jacc_kernel_pipeline_d_2!, i_n, u, ud, v, vd, w, wd)
    for i_seq_x = 1:i_n
        lossd[1] = lossd[1] + wd[i_seq_x]
        loss[1] = loss[1] + w[i_seq_x]
    end
    return nothing
end

function pipeline_jacc(loss, u, v, w, i_n)
    JACC.parallel_for(div(i_n - 1, 1) + 1, jacc_kernel_pipeline_1!, i_n, u, v)
    JACC.parallel_for(div(i_n - 1, 1) + 1, jacc_kernel_pipeline_2!, i_n, u, v, w)
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + w[i_seq_x]
    end
    return nothing
end
