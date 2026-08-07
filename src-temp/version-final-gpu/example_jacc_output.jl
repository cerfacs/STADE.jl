import Pkg
haskey(Pkg.project().dependencies, "JACC") || Pkg.add("JACC")
haskey(Pkg.project().dependencies, "Atomix") || Pkg.add("Atomix")
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_bump_1!(__jacc_i, n, u, v)
    i_x = 1 + (__jacc_i - 1)
    v[i_x] = v[i_x] + 2.0 * u[i_x]
    Atomix.@atomic v[1] += u[i_x]
    return nothing
end

function bump_jacc(u, v, n)
    JACC.parallel_for(div(n - 1, 1) + 1, jacc_kernel_bump_1!, n, u, v)
    acc = 0.0
    for i_seq_t = 1:n
        acc = acc + u[i_seq_t]
    end
    v[2] = acc
    return nothing
end
