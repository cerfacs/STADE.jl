import Pkg
haskey(Pkg.project().dependencies, "JACC") || Pkg.add("JACC")
haskey(Pkg.project().dependencies, "Atomix") || Pkg.add("Atomix")
import JACC
import Atomix
JACC.@init_backend

function bilinear_jacc(loss, x, a, y, i_m, i_n)
    for i_seq_i = 1:i_m
        for i_seq_j = 1:i_n
            loss[1] = loss[1] + x[i_seq_i] * a[i_seq_i, i_seq_j] * y[i_seq_j]
        end
    end
    return nothing
end
