import Pkg
haskey(Pkg.project().dependencies, "JACC") || Pkg.add(name = "JACC", version = "1")
haskey(Pkg.project().dependencies, "Atomix") || Pkg.add("Atomix")
import JACC
import Atomix
JACC.@init_backend

function branchsel_d_jacc(loss, lossd, x, xd, y, yd)
    if x > y
        lossd[1] = (2x) * xd + -yd
        loss[1] = x ^ 2 - y
    else
        lossd[1] = (2y) * yd + -xd
        loss[1] = y ^ 2 - x
    end
    return nothing
end

function branchsel_jacc(loss, x, y)
    if x > y
        loss[1] = x ^ 2 - y
    else
        loss[1] = y ^ 2 - x
    end
    return nothing
end
