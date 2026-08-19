import Pkg
haskey(Pkg.project().dependencies, "JACC") || Pkg.add(name = "JACC", version = "1")
haskey(Pkg.project().dependencies, "Atomix") || Pkg.add("Atomix")
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_richardson_substep_d_1!(__jacc_i, a_coef, a_coefd, h, hd, nsub, y, yd)
    i_seq_sub = 1 + (__jacc_i - 1)
    Atomix.@atomic yd[1] += -((a_coef * y) * hd) + -((h * y) * a_coefd) + -((h * a_coef) * yd)
    Atomix.@atomic y[1] += -(h * a_coef * y)
    return nothing
end

function jacc_kernel_richardson_substep_1!(__jacc_i, a_coef, h, nsub, y)
    i_seq_sub = 1 + (__jacc_i - 1)
    Atomix.@atomic y[1] += -(h * a_coef * y)
    return nothing
end

function richardson_substep_d_jacc(y_init, y_initd, out, outd, a_coef, a_coefd, dt_stage, dt_staged, num_stages)
    y = JACC.array([y])
    yd = JACC.array([yd])
    nsubd = 0.0
    nsub = 1
    for i_seq_stage = 1:num_stages
        hd = (1.0 / nsub) * dt_staged
        h = dt_stage / nsub
        yd = y_initd
        y = y_init
        JACC.@parallel_for range = div(nsub - 1, 1) + 1 jacc_kernel_richardson_substep_d_1!(a_coef, a_coefd, h, hd, nsub, y, yd)
        outd[i_seq_stage] = yd
        out[i_seq_stage] = y
        nsubd = 0.0
        nsub = nsub * 2
    end
    return nothing
end

function richardson_substep_jacc(y_init, out, a_coef, dt_stage, num_stages)
    y = JACC.array([y])
    nsub = 1
    for i_seq_stage = 1:num_stages
        h = dt_stage / nsub
        y = y_init
        JACC.@parallel_for range = div(nsub - 1, 1) + 1 jacc_kernel_richardson_substep_1!(a_coef, h, nsub, y)
        out[i_seq_stage] = y
        nsub = nsub * 2
    end
    return nothing
end
