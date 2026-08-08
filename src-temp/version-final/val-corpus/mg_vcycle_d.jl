function mg_vcycle_d(u, ud, f, fd, r, rd, nfine, num_levels, h1, h1d, nu1, nu2, n)
    nd = 0.0
    n = n * 2
    nld = 0.0
    nl = nfine
    hld = h1d
    hl = h1
    for i_seq_level = 1:num_levels - 1
        nd = 0.0
        n = nl - 1
        hl2d = hl * hld + hl * hld
        hl2 = hl * hl
        for i_seq_k = 1:nu1
            for i_seq_j = 1:n
                leftd = 0.0
                left = 0.0
                if i_seq_j > 1
                    leftd = ud[i_seq_j - 1, i_seq_level]
                    left = u[i_seq_j - 1, i_seq_level]
                end
                rightd = 0.0
                right = 0.0
                if i_seq_j < n
                    rightd = ud[i_seq_j + 1, i_seq_level]
                    right = u[i_seq_j + 1, i_seq_level]
                end
                ud[i_seq_j, i_seq_level] = 0.5 * (((f[i_seq_j, i_seq_level] * hl2d + hl2 * fd[i_seq_j, i_seq_level]) + leftd) + rightd)
                u[i_seq_j, i_seq_level] = 0.5 * (hl2 * f[i_seq_j, i_seq_level] + left + right)
            end
        end
        for j = 1:n
            leftd = 0.0
            left = 0.0
            if j > 1
                leftd = ud[j - 1, i_seq_level]
                left = u[j - 1, i_seq_level]
            end
            rightd = 0.0
            right = 0.0
            if j < n
                rightd = ud[j + 1, i_seq_level]
                right = u[j + 1, i_seq_level]
            end
            rd[j, i_seq_level] = fd[j, i_seq_level] + -(((1.0 / hl2) * ((2.0 * ud[j, i_seq_level] + -leftd) + -rightd) + -(((2.0 * u[j, i_seq_level] - left) - right) / hl2 ^ 2) * hl2d))
            r[j, i_seq_level] = f[j, i_seq_level] - ((2.0 * u[j, i_seq_level] - left) - right) / hl2
        end
        ncgd = 0.0
        ncg = div(nl, 2)
        ncd = 0.0
        nc = ncg - 1
        for j = 1:nc
            jfd = 0.0
            jf = j * 2
            fd[j, i_seq_level + 1] = (0.25 * rd[jf - 1, i_seq_level] + 0.5 * rd[jf, i_seq_level]) + 0.25 * rd[jf + 1, i_seq_level]
            f[j, i_seq_level + 1] = 0.25 * r[jf - 1, i_seq_level] + 0.5 * r[jf, i_seq_level] + 0.25 * r[jf + 1, i_seq_level]
        end
        for j = 1:nc
            ud[j, i_seq_level + 1] = 0.0
            u[j, i_seq_level + 1] = 0.0
        end
        nld = 0.0
        nl = ncg
        hld = 2.0hld
        hl = hl * 2.0
    end
    hl2d = hl * hld + hl * hld
    hl2 = hl * hl
    ud[1, num_levels] = (0.5 * f[1, num_levels]) * hl2d + (0.5hl2) * fd[1, num_levels]
    u[1, num_levels] = 0.5 * hl2 * f[1, num_levels]
    for i_seq_level = num_levels - 1:-1:1
        nld = 0.0
        nl = nl * 2
        hld = 0.5hld
        hl = hl / 2.0
        nd = 0.0
        n = nl - 1
        ncgd = 0.0
        ncg = div(nl, 2)
        ncd = 0.0
        nc = ncg - 1
        hl2d = hl * hld + hl * hld
        hl2 = hl * hl
        for j = 1:nc
            jfd = 0.0
            jf = j * 2
            ud[jf, i_seq_level] = ud[jf, i_seq_level] + ud[j, i_seq_level + 1]
            u[jf, i_seq_level] = u[jf, i_seq_level] + u[j, i_seq_level + 1]
        end
        for j = 1:nc + 1
            jfd = 0.0
            jf = j * 2 - 1
            cld = 0.0
            cl = 0.0
            if j > 1
                cld = ud[j - 1, i_seq_level + 1]
                cl = u[j - 1, i_seq_level + 1]
            end
            crd = 0.0
            cr = 0.0
            if j <= nc
                crd = ud[j, i_seq_level + 1]
                cr = u[j, i_seq_level + 1]
            end
            ud[jf, i_seq_level] = ud[jf, i_seq_level] + 0.5 * (cld + crd)
            u[jf, i_seq_level] = u[jf, i_seq_level] + 0.5 * (cl + cr)
        end
        for i_seq_k = 1:nu2
            for i_seq_j = 1:n
                leftd = 0.0
                left = 0.0
                if i_seq_j > 1
                    leftd = ud[i_seq_j - 1, i_seq_level]
                    left = u[i_seq_j - 1, i_seq_level]
                end
                rightd = 0.0
                right = 0.0
                if i_seq_j < n
                    rightd = ud[i_seq_j + 1, i_seq_level]
                    right = u[i_seq_j + 1, i_seq_level]
                end
                ud[i_seq_j, i_seq_level] = 0.5 * (((f[i_seq_j, i_seq_level] * hl2d + hl2 * fd[i_seq_j, i_seq_level]) + leftd) + rightd)
                u[i_seq_j, i_seq_level] = 0.5 * (hl2 * f[i_seq_j, i_seq_level] + left + right)
            end
        end
    end
    return nothing
end

function mg_vcycle(u, f, r, nfine, num_levels, h1, nu1, nu2, n)
    #= none:1 =#
    #= none:2 =#
    n = n * 2
    #= none:3 =#
    nl = nfine
    #= none:4 =#
    hl = h1
    #= none:5 =#
    for i_seq_level = 1:num_levels - 1
        #= none:6 =#
        n = nl - 1
        #= none:7 =#
        hl2 = hl * hl
        #= none:8 =#
        for i_seq_k = 1:nu1
            #= none:9 =#
            for i_seq_j = 1:n
                #= none:10 =#
                left = 0.0
                #= none:11 =#
                if i_seq_j > 1
                    #= none:12 =#
                    left = u[i_seq_j - 1, i_seq_level]
                end
                #= none:14 =#
                right = 0.0
                #= none:15 =#
                if i_seq_j < n
                    #= none:16 =#
                    right = u[i_seq_j + 1, i_seq_level]
                end
                #= none:18 =#
                u[i_seq_j, i_seq_level] = 0.5 * (hl2 * f[i_seq_j, i_seq_level] + left + right)
                #= none:19 =#
            end
            #= none:20 =#
        end
        #= none:21 =#
        for j = 1:n
            #= none:22 =#
            left = 0.0
            #= none:23 =#
            if j > 1
                #= none:24 =#
                left = u[j - 1, i_seq_level]
            end
            #= none:26 =#
            right = 0.0
            #= none:27 =#
            if j < n
                #= none:28 =#
                right = u[j + 1, i_seq_level]
            end
            #= none:30 =#
            r[j, i_seq_level] = f[j, i_seq_level] - ((2.0 * u[j, i_seq_level] - left) - right) / hl2
            #= none:31 =#
        end
        #= none:32 =#
        ncg = div(nl, 2)
        #= none:33 =#
        nc = ncg - 1
        #= none:34 =#
        for j = 1:nc
            #= none:35 =#
            jf = j * 2
            #= none:36 =#
            f[j, i_seq_level + 1] = 0.25 * r[jf - 1, i_seq_level] + 0.5 * r[jf, i_seq_level] + 0.25 * r[jf + 1, i_seq_level]
            #= none:37 =#
        end
        #= none:38 =#
        for j = 1:nc
            #= none:39 =#
            u[j, i_seq_level + 1] = 0.0
            #= none:40 =#
        end
        #= none:41 =#
        nl = ncg
        #= none:42 =#
        hl = hl * 2.0
        #= none:43 =#
    end
    #= none:44 =#
    hl2 = hl * hl
    #= none:45 =#
    u[1, num_levels] = 0.5 * hl2 * f[1, num_levels]
    #= none:46 =#
    for i_seq_level = num_levels - 1:-1:1
        #= none:47 =#
        nl = nl * 2
        #= none:48 =#
        hl = hl / 2.0
        #= none:49 =#
        n = nl - 1
        #= none:50 =#
        ncg = div(nl, 2)
        #= none:51 =#
        nc = ncg - 1
        #= none:52 =#
        hl2 = hl * hl
        #= none:53 =#
        for j = 1:nc
            #= none:54 =#
            jf = j * 2
            #= none:55 =#
            u[jf, i_seq_level] = u[jf, i_seq_level] + u[j, i_seq_level + 1]
            #= none:56 =#
        end
        #= none:57 =#
        for j = 1:nc + 1
            #= none:58 =#
            jf = j * 2 - 1
            #= none:59 =#
            cl = 0.0
            #= none:60 =#
            if j > 1
                #= none:61 =#
                cl = u[j - 1, i_seq_level + 1]
            end
            #= none:63 =#
            cr = 0.0
            #= none:64 =#
            if j <= nc
                #= none:65 =#
                cr = u[j, i_seq_level + 1]
            end
            #= none:67 =#
            u[jf, i_seq_level] = u[jf, i_seq_level] + 0.5 * (cl + cr)
            #= none:68 =#
        end
        #= none:69 =#
        for i_seq_k = 1:nu2
            #= none:70 =#
            for i_seq_j = 1:n
                #= none:71 =#
                left = 0.0
                #= none:72 =#
                if i_seq_j > 1
                    #= none:73 =#
                    left = u[i_seq_j - 1, i_seq_level]
                end
                #= none:75 =#
                right = 0.0
                #= none:76 =#
                if i_seq_j < n
                    #= none:77 =#
                    right = u[i_seq_j + 1, i_seq_level]
                end
                #= none:79 =#
                u[i_seq_j, i_seq_level] = 0.5 * (hl2 * f[i_seq_j, i_seq_level] + left + right)
                #= none:80 =#
            end
            #= none:81 =#
        end
        #= none:82 =#
    end
end
