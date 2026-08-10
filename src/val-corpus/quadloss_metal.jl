import Pkg
haskey(Pkg.project().dependencies, "Metal") || Pkg.add("Metal")
using Metal
Metal.allowscalar(false)

function quadloss_metal(loss, x, y, z)
    loss[1] = (x ^ 2 * y - 3.0f0 * y * z) + z ^ 3
    return nothing
end
