import Pkg
haskey(Pkg.project().dependencies, "JACC") || Pkg.add("JACC")
haskey(Pkg.project().dependencies, "Atomix") || Pkg.add("Atomix")
import JACC
import Atomix
JACC.@init_backend

function geomrecur_d_jacc(loss, lossd, u, ud, c, cd, i_n)
    for i_seq_x = 2:i_n
        ud[i_seq_x] = u[i_seq_x - 1] * cd + c * ud[i_seq_x - 1]
        u[i_seq_x] = c * u[i_seq_x - 1]
    end
    for i_seq_x = 1:i_n
        lossd[1] = lossd[1] + (2 * u[i_seq_x]) * ud[i_seq_x]
        loss[1] = loss[1] + u[i_seq_x] ^ 2
    end
    return nothing
end

function geomrecur_jacc(loss, u, c, i_n)
    for i_seq_x = 2:i_n
        u[i_seq_x] = c * u[i_seq_x - 1]
    end
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + u[i_seq_x] ^ 2
    end
    return nothing
end
