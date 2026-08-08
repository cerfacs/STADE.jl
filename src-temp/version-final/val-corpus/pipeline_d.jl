function pipeline_d(loss, lossd, u, ud, v, vd, w, wd, i_n)
    for i_x = 1:i_n
        vd[i_x] = (2 * u[i_x]) * ud[i_x]
        v[i_x] = u[i_x] ^ 2 + 1.0
    end
    for i_x = 1:i_n
        wd[i_x] = u[i_x] * vd[i_x] + v[i_x] * ud[i_x]
        w[i_x] = v[i_x] * u[i_x]
    end
    for i_seq_x = 1:i_n
        lossd[1] = lossd[1] + wd[i_seq_x]
        loss[1] = loss[1] + w[i_seq_x]
    end
    return nothing
end

function pipeline(loss, u, v, w, i_n)
    #= none:1 =#
    #= none:2 =#
    for i_x = 1:i_n
        #= none:3 =#
        v[i_x] = u[i_x] ^ 2 + 1.0
        #= none:4 =#
    end
    #= none:5 =#
    for i_x = 1:i_n
        #= none:6 =#
        w[i_x] = v[i_x] * u[i_x]
        #= none:7 =#
    end
    #= none:8 =#
    for i_seq_x = 1:i_n
        #= none:9 =#
        loss[1] = loss[1] + w[i_seq_x]
        #= none:10 =#
    end
end
