import Pkg
haskey(Pkg.project().dependencies, "JACC") || Pkg.add("JACC")
haskey(Pkg.project().dependencies, "Atomix") || Pkg.add("Atomix")
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_normcomp_1!(__jacc_i, i_n, u, v, w)
    i_x = 1 + (__jacc_i - 1)
    w[i_x] = u[i_x] - v[i_x]
    return nothing
end

function normcomp_jacc(loss, u, v, w, i_n)
    JACC.parallel_for(div(i_n - 1, 1) + 1, jacc_kernel_normcomp_1!, i_n, u, v, w)
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + w[i_seq_x] ^ 2
    end
    return nothing
end
