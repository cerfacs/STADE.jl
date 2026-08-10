import Pkg
haskey(Pkg.project().dependencies, "Metal") || Pkg.add("Metal")
using Metal
Metal.allowscalar(false)

function cond_loop_choice_metal(loss, u, v, i_branch, i_n)
    if i_branch == 1
        for i_seq_x = 1:i_n
            loss[1] = loss[1] + u[i_seq_x] ^ 2
        end
    else
        for i_seq_x = 1:i_n
            loss[1] = loss[1] + v[i_seq_x] ^ 2
        end
    end
    return nothing
end
