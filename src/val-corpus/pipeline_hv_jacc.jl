import Pkg
haskey(Pkg.project().dependencies, "JACC") || Pkg.add("JACC")
haskey(Pkg.project().dependencies, "Atomix") || Pkg.add("Atomix")
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_pipeline_hv_1!(__jacc_i, i_n, u, ud, v, vd)
    i_x = 1 + (__jacc_i - 1)
    vd[i_x] = (2 * u[i_x]) * ud[i_x]
    v[i_x] = u[i_x] ^ 2 + 1.0
    return nothing
end

function jacc_kernel_pipeline_hv_2!(__jacc_i, i_n, u, ud, v, vd, w, wd)
    i_x = 1 + (__jacc_i - 1)
    wd[i_x] = u[i_x] * vd[i_x] + v[i_x] * ud[i_x]
    w[i_x] = v[i_x] * u[i_x]
    return nothing
end

function jacc_kernel_pipeline_hv_3!(__jacc_i, i_n, u, ub, ubd, ud, v, vb, vbd, vd, wb, wbd)
    i_x = 1 + (__jacc_i - 1)
    vbd[i_x] = vbd[i_x] + (wb[i_x] * ud[i_x] + u[i_x] * wbd[i_x])
    vb[i_x] = vb[i_x] + u[i_x] * wb[i_x]
    ubd[i_x] = ubd[i_x] + (wb[i_x] * vd[i_x] + v[i_x] * wbd[i_x])
    ub[i_x] = ub[i_x] + v[i_x] * wb[i_x]
    wbd[i_x] = 0.0
    wb[i_x] = 0.0
    return nothing
end

function jacc_kernel_pipeline_hv_4!(__jacc_i, i_n, u, ub, ubd, ud, vb, vbd)
    i_x = 1 + (__jacc_i - 1)
    ubd[i_x] = ubd[i_x] + (vb[i_x] * (2 * ud[i_x]) + (2 * u[i_x]) * vbd[i_x])
    ub[i_x] = ub[i_x] + (2 * u[i_x]) * vb[i_x]
    vbd[i_x] = 0.0
    vb[i_x] = 0.0
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

function initstacks_pipeline_b_jacc()
    return nothing
end

function pipeline_hv_jacc(loss, lossb, u, ub, v, vb, w, wb, i_n, lossd, lossbd, ud, ubd, vd, vbd, wd, wbd)
    JACC.parallel_for(div(i_n - 1, 1) + 1, jacc_kernel_pipeline_hv_1!, i_n, u, ud, v, vd)
    JACC.parallel_for(div(i_n - 1, 1) + 1, jacc_kernel_pipeline_hv_2!, i_n, u, ud, v, vd, w, wd)
    for i_seq_x = 1:i_n
        lossd[1] = lossd[1] + wd[i_seq_x]
        loss[1] = loss[1] + w[i_seq_x]
    end
    for i_seq_x = i_n:-1:1
        wbd[i_seq_x] = wbd[i_seq_x] + lossbd[1]
        wb[i_seq_x] = wb[i_seq_x] + lossb[1]
    end
    JACC.parallel_for(div(i_n - 1, 1) + 1, jacc_kernel_pipeline_hv_3!, i_n, u, ub, ubd, ud, v, vb, vbd, vd, wb, wbd)
    JACC.parallel_for(div(i_n - 1, 1) + 1, jacc_kernel_pipeline_hv_4!, i_n, u, ub, ubd, ud, vb, vbd)
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
