import Pkg
haskey(Pkg.project().dependencies, "JACC") || Pkg.add("JACC")
haskey(Pkg.project().dependencies, "Atomix") || Pkg.add("Atomix")
import JACC
import Atomix
JACC.@init_backend

function clamped_sumsq_d_jacc(loss, lossd, u, ud, i_n)
    for i_seq_x = 1:i_n
        if u[i_seq_x] > 0.0
            wd = (2 * u[i_seq_x]) * ud[i_seq_x]
            w = u[i_seq_x] ^ 2
        else
            wd = 0.0
            w = 0.0
        end
        lossd[1] = lossd[1] + wd
        loss[1] = loss[1] + w
    end
    return nothing
end

function clamped_sumsq_jacc(loss, u, i_n)
    for i_seq_x = 1:i_n
        if u[i_seq_x] > 0.0
            w = u[i_seq_x] ^ 2
        else
            w = 0.0
        end
        loss[1] = loss[1] + w
    end
    return nothing
end
