# --- from advection_b.jl ---
function initstacks_func_b(du)
    du_stack = Vector{typeof(du)}()
    sizehint!(du_stack, 100)
    return du_stack
end
function func_b(u, ub, du, dub, c, cb, dx, dxb, dt, dtb, i_nstep, i_nnode, du_stack)
    for i_seq_ = 1:i_nstep
        push!(du_stack, copy(du))
        for i_x = 2:i_nnode
            du[i_x] = u[i_x] - u[i_x - 1]
        end
        push!(du_stack, copy(du))
        for i_x = 2:i_nnode
            u[i_x] = u[i_x] - (c * dt * du[i_x]) / dx
        end
    end
    for i_seq_ = i_nstep:-1:1
        du .= pop!(du_stack)
        for i_x = 2:i_nnode
            temp = du[i_x] / dx
            cb = cb - dt * temp * ub[i_x]
            dtb = dtb - c * temp * ub[i_x]
            tempb = -((c * dt * ub[i_x]) / dx)
            dub[i_x] = dub[i_x] + tempb
            dxb = dxb - temp * tempb
        end
        du .= pop!(du_stack)
        for i_x = 2:i_nnode
            ub[i_x] = ub[i_x] + dub[i_x]
            ub[i_x - 1] = ub[i_x - 1] - dub[i_x]
            dub[i_x] = 0.0
        end
    end
    return cb,dxb,dtb
end
function func(u, du, c, dx, dt, i_nstep, i_nnode)
    for i_seq_ = 1:i_nstep
        for i_x = 2:i_nnode
            du[i_x] = u[i_x] - u[i_x - 1]
        end
        for i_x = 2:i_nnode
            u[i_x] = u[i_x] - (c * dt * du[i_x]) / dx
        end
    end
end


# --- from affine_loss_b.jl ---
function initstacks_affine_loss_b()
    return
end
function affine_loss_b(loss, lossb, u, ub, a, ab, b, bb, v, vb, i_n)
    for i_x = 1:i_n
        v[i_x] = a[i_x] * u[i_x] + b[i_x]
    end
    for i_seq_x = i_n:-1:1
        vb[i_seq_x] = vb[i_seq_x] + 2 * v[i_seq_x] * lossb[1]
    end
    for i_x = 1:i_n
        ab[i_x] = ab[i_x] + u[i_x] * vb[i_x]
        ub[i_x] = ub[i_x] + a[i_x] * vb[i_x]
        bb[i_x] = bb[i_x] + vb[i_x]
        vb[i_x] = 0.0
    end
    return 
end
function affine_loss(loss, u, a, b, v, i_n)
    for i_x = 1:i_n
        v[i_x] = a[i_x] * u[i_x] + b[i_x]
    end
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + v[i_seq_x] ^ 2
    end
end


# --- from bilinear_b.jl ---
function initstacks_bilinear_b()
    return
end
function bilinear_b(loss, lossb, x, xb, a, ab, y, yb, i_m, i_n)
    for i_seq_i = i_m:-1:1
        for i_seq_j = i_n:-1:1
            ab[i_seq_i, i_seq_j] = ab[i_seq_i, i_seq_j] + x[i_seq_i] * y[i_seq_j] * lossb[1]
            tempb = a[i_seq_i, i_seq_j] * lossb[1]
            xb[i_seq_i] = xb[i_seq_i] + y[i_seq_j] * tempb
            yb[i_seq_j] = yb[i_seq_j] + x[i_seq_i] * tempb
        end
    end
    return 
end
function bilinear(loss, x, a, y, i_m, i_n)
    for i_seq_i = 1:i_m
        for i_seq_j = 1:i_n
            loss[1] = loss[1] + x[i_seq_i] * a[i_seq_i, i_seq_j] * y[i_seq_j]
        end
    end
end


# --- from branchsel_b.jl ---
function initstacks_branchsel_b()
    return
end
function branchsel_b(loss, lossb, x, xb, y, yb)
    if x > y
        xb = xb + 2 * x * lossb[1]
        yb = yb - lossb[1]
        lossb[1] = 0.0
    else
        yb = yb + 2 * y * lossb[1]
        xb = xb - lossb[1]
        lossb[1] = 0.0
    end
    return xb,yb
end
function branchsel(loss, x, y)
    if x > y
        loss[1] = x ^ 2 - y
    else
        loss[1] = y ^ 2 - x
    end
end


# --- from clamped_sumsq_b.jl ---
function initstacks_clamped_sumsq_b()
    branch_stack = Int[]
    sizehint!(branch_stack, 100)
    return branch_stack
end
function clamped_sumsq_b(loss, lossb, u, ub, i_n, branch_stack)
    for i_seq_x = 1:i_n
        if u[i_seq_x] > 0.0
            push!(branch_stack, 0)
        else
            push!(branch_stack, 1)
        end
    end
    for i_seq_x = i_n:-1:1
        wb = lossb[1]
        branch = pop!(branch_stack)
        if branch == 0
            ub[i_seq_x] = ub[i_seq_x] + 2 * u[i_seq_x] * wb
        end
    end
    return 
end
function clamped_sumsq(loss, u, i_n)
    for i_seq_x = 1:i_n
        if u[i_seq_x] > 0.0
            w = u[i_seq_x] ^ 2
        else
            w = 0.0
        end
        loss[1] = loss[1] + w
    end
end


# --- from cond_field_choice_b.jl ---
function initstacks_cond_field_choice_b()
    branch_stack = Int[]
    sizehint!(branch_stack, 100)
    return branch_stack
end
function cond_field_choice_b(loss, lossb, u, ub, v, vb, w, wb, i_branch, i_n, branch_stack)
    if i_branch == 1
        push!(branch_stack, 1)
    else
        push!(branch_stack, 0)
    end
    for i_seq_x = i_n:-1:1
        wb[i_seq_x] = wb[i_seq_x] + lossb[1]
    end
    branch = pop!(branch_stack)
    if branch == 0
        for i_x = 1:i_n
            vb[i_x] = vb[i_x] + 2 * v[i_x] * wb[i_x]
            wb[i_x] = 0.0
        end
    else
        for i_x = 1:i_n
            ub[i_x] = ub[i_x] + 2 * u[i_x] * wb[i_x]
            wb[i_x] = 0.0
        end
    end
    return 
end
function cond_field_choice(loss, u, v, w, i_branch, i_n)
    if i_branch == 1
        for i_x = 1:i_n
            w[i_x] = u[i_x] ^ 2
        end
    else
        for i_x = 1:i_n
            w[i_x] = v[i_x] ^ 2
        end
    end
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + w[i_seq_x]
    end
end


# --- from cond_loop_choice_b.jl ---
function initstacks_cond_loop_choice_b()
    return
end
function cond_loop_choice_b(loss, lossb, u, ub, v, vb, i_branch, i_n)
    if i_branch == 1
        for i_seq_x = i_n:-1:1
            ub[i_seq_x] = ub[i_seq_x] + 2 * u[i_seq_x] * lossb[1]
        end
    else
        for i_seq_x = i_n:-1:1
            vb[i_seq_x] = vb[i_seq_x] + 2 * v[i_seq_x] * lossb[1]
        end
    end
    return 
end
function cond_loop_choice(loss, u, v, i_branch, i_n)
    if i_branch == 1
        for i_seq_x = 1:i_n
            loss[1] = loss[1] + u[i_seq_x] ^ 2
        end
    else
        for i_seq_x = 1:i_n
            loss[1] = loss[1] + v[i_seq_x] ^ 2
        end
    end
end


# --- from dotprod_b.jl ---
function initstacks_dotprod_b()
    return
end
function dotprod_b(loss, lossb, u, ub, v, vb, i_n)
    for i_seq_x = i_n:-1:1
        ub[i_seq_x] = ub[i_seq_x] + v[i_seq_x] * lossb[1]
        vb[i_seq_x] = vb[i_seq_x] + u[i_seq_x] * lossb[1]
    end
    return 
end
function dotprod(loss, u, v, i_n)
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + u[i_seq_x] * v[i_seq_x]
    end
end


# --- from geomrecur_b.jl ---
function initstacks_geomrecur_b(u)
    u_stack = Vector{eltype(u)}()
    sizehint!(u_stack, 100)
    return u_stack
end
function geomrecur_b(loss, lossb, u, ub, c, cb, i_n, u_stack)
    for i_seq_x = 2:i_n
        push!(u_stack, u[i_seq_x])
        u[i_seq_x] = c * u[i_seq_x - 1]
    end
    for i_seq_x = i_n:-1:1
        ub[i_seq_x] = ub[i_seq_x] + 2 * u[i_seq_x] * lossb[1]
    end
    for i_seq_x = i_n:-1:2
        u[i_seq_x] = pop!(u_stack)
        cb = cb + u[i_seq_x - 1] * ub[i_seq_x]
        ub[i_seq_x - 1] = ub[i_seq_x - 1] + c * ub[i_seq_x]
        ub[i_seq_x] = 0.0
    end
    return cb
end
function geomrecur(loss, u, c, i_n)
    for i_seq_x = 2:i_n
        u[i_seq_x] = c * u[i_seq_x - 1]
    end
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + u[i_seq_x] ^ 2
    end
end


# --- from matvec_loss_b.jl ---
function initstacks_matvec_loss_b()
    return
end
function matvec_loss_b(loss, lossb, a, ab, u, ub, v, vb, i_m, i_n)
    for i_i = 1:i_m
        for i_seq_j = 1:i_n
            v[i_i] = v[i_i] + a[i_i, i_seq_j] * u[i_seq_j]
        end
    end
    for i_seq_i = i_m:-1:1
        vb[i_seq_i] = vb[i_seq_i] + 2 * v[i_seq_i] * lossb[1]
    end
    for i_i = 1:i_m
        for i_seq_j = i_n:-1:1
            ab[i_i, i_seq_j] = ab[i_i, i_seq_j] + u[i_seq_j] * vb[i_i]
            ub[i_seq_j] = ub[i_seq_j] + a[i_i, i_seq_j] * vb[i_i]
        end
    end
    return 
end
function matvec_loss(loss, a, u, v, i_m, i_n)
    for i_i = 1:i_m
        for i_seq_j = 1:i_n
            v[i_i] = v[i_i] + a[i_i, i_seq_j] * u[i_seq_j]
        end
    end
    for i_seq_i = 1:i_m
        loss[1] = loss[1] + v[i_seq_i] ^ 2
    end
end


# --- from mg_vcycle_b.jl ---
function initstacks_mg_vcycle_b(f, u)
    n_stack = Int[]
    sizehint!(n_stack, 100)
    f_stack = Vector{typeof(f)}()
    sizehint!(f_stack, 100)
    branch_stack = Int[]
    sizehint!(branch_stack, 100)
    hl_stack = Float64[]
    sizehint!(hl_stack, 100)
    hl2_stack = Float64[]
    sizehint!(hl2_stack, 100)
    u_stack = Vector{typeof(u)}()
    sizehint!(u_stack, 100)
    integer8_stack = Int[]
    sizehint!(integer8_stack, 100)
    return (n_stack, f_stack, branch_stack, hl_stack, hl2_stack, u_stack, integer8_stack)
end
function mg_vcycle_b(u, ub, f, fb, r, rb, nfine, num_levels, h1, h1b, nu1, nu2, n, n_stack, f_stack, branch_stack, hl_stack, hl2_stack, u_stack, integer8_stack)
    nl = nfine
    hl = h1
    for i_seq_level = 1:num_levels - 1
        n = nl - 1
        hl2 = hl * hl
        for i_seq_k = 1:nu1
            for i_seq_j = 1:n
                left = 0.0
                if i_seq_j > 1
                    left = u[i_seq_j - 1, i_seq_level]
                    push!(branch_stack, 0)
                else
                    push!(branch_stack, 1)
                end
                right = 0.0
                if i_seq_j < n
                    right = u[i_seq_j + 1, i_seq_level]
                    push!(branch_stack, 0)
                else
                    push!(branch_stack, 1)
                end
                u[i_seq_j, i_seq_level] = 0.5 * (hl2 * f[i_seq_j, i_seq_level] + left + right)
            end
            # push!(integer8_stack, i_seq_j - 1)
            push!(integer8_stack, n)
        end
        push!(n_stack, n)
        push!(u_stack, copy(u))
        for j = 1:n
            left = 0.0
            if j > 1
                left = u[j - 1, i_seq_level]
            end
            right = 0.0
            if j < n
                right = u[j + 1, i_seq_level]
            end
            r[j, i_seq_level] = f[j, i_seq_level] - ((2.0 * u[j, i_seq_level] - left) - right) / hl2
        end
        ncg = div(nl, 2)
        nc = ncg - 1
        push!(f_stack, copy(f))
        for j = 1:nc
            jf = j * 2
            f[j, i_seq_level + 1] = 0.25 * r[jf - 1, i_seq_level] + 0.5 * r[jf, i_seq_level] + 0.25 * r[jf + 1, i_seq_level]
        end
        push!(u_stack, copy(u))
        for j = 1:nc
            u[j, i_seq_level + 1] = 0.0
        end
        nl = ncg
        push!(hl_stack, hl)
        hl = hl * 2.0
    end
    # Add initialization of 'nc' 
    # to be able to register its latest value
    # for later use in backward pass
    nc = 1
    for i_seq_level = num_levels - 1:-1:1
        nl = nl * 2
        push!(hl_stack, hl)
        hl = hl / 2.0
        n = nl - 1
        ncg = div(nl, 2)
        nc = ncg - 1
        push!(hl2_stack, hl2)
        hl2 = hl * hl
        for i_seq_k = 1:nu2
            for i_seq_j = 1:n
                if i_seq_j > 1
                    push!(branch_stack, 0)
                else
                    push!(branch_stack, 1)
                end
                if i_seq_j < n
                    push!(branch_stack, 0)
                else
                    push!(branch_stack, 1)
                end
            end
            # push!(integer8_stack, i_seq_j - 1)
            push!(integer8_stack, n)
        end
    end
    hlb = 0.0
    for i_seq_level = 1:1:num_levels - 1
        hl2b = 0.0
        for i_seq_k = nu2:-1:1
            ad_to0 = pop!(integer8_stack)
            for i_seq_j = ad_to0:-1:1
                tempb = 0.5 * ub[i_seq_j, i_seq_level]
                ub[i_seq_j, i_seq_level] = 0.0
                hl2b = hl2b + f[i_seq_j, i_seq_level] * tempb
                fb[i_seq_j, i_seq_level] = fb[i_seq_j, i_seq_level] + hl2 * tempb
                leftb = tempb
                rightb = tempb
                branch = pop!(branch_stack)
                if branch == 0
                    ub[i_seq_j + 1, i_seq_level] = ub[i_seq_j + 1, i_seq_level] + rightb
                end
                branch = pop!(branch_stack)
                if branch == 0
                    ub[i_seq_j - 1, i_seq_level] = ub[i_seq_j - 1, i_seq_level] + leftb
                end
            end
        end
        for j = 1:nc + 1
            jf = j * 2 - 1
            if j > 1
                push!(branch_stack, 0)
            else
                push!(branch_stack, 1)
            end
            if j <= nc
                push!(branch_stack, 0)
            else
                push!(branch_stack, 1)
            end
            clb = 0.5 * ub[jf, i_seq_level]
            crb = 0.5 * ub[jf, i_seq_level]
            branch = pop!(branch_stack)
            if branch == 0
                ub[j, i_seq_level + 1] = ub[j, i_seq_level + 1] + crb
            end
            branch = pop!(branch_stack)
            if branch == 0
                ub[j - 1, i_seq_level + 1] = ub[j - 1, i_seq_level + 1] + clb
            end
        end
        for j = 1:nc
            jf = j * 2
            ub[j, i_seq_level + 1] = ub[j, i_seq_level + 1] + ub[jf, i_seq_level]
        end
        hl2 = pop!(hl2_stack)
        hlb = hlb + 2 * hl * hl2b
        hl = pop!(hl_stack)
        hlb = hlb / 2.0
    end
    hl2 = hl * hl
    hl2b = f[1, num_levels] * 0.5 * ub[1, num_levels]
    fb[1, num_levels] = fb[1, num_levels] + hl2 * 0.5 * ub[1, num_levels]
    ub[1, num_levels] = 0.0
    hlb = hlb + 2 * hl * hl2b
    for i_seq_level = num_levels - 1:-1:1
        hl = pop!(hl_stack)
        hlb = 2.0hlb
        u .= pop!(u_stack)
        for j = 1:nc
            ub[j, i_seq_level + 1] = 0.0
        end
        f .= pop!(f_stack)
        for j = 1:nc
            jf = j * 2
            rb[jf - 1, i_seq_level] = rb[jf - 1, i_seq_level] + 0.25 * fb[j, i_seq_level + 1]
            rb[jf, i_seq_level] = rb[jf, i_seq_level] + 0.5 * fb[j, i_seq_level + 1]
            rb[jf + 1, i_seq_level] = rb[jf + 1, i_seq_level] + 0.25 * fb[j, i_seq_level + 1]
            fb[j, i_seq_level + 1] = 0.0
        end
        hl2 = hl * hl
        hl2b = 0.0
        u .= pop!(u_stack)
        n = pop!(n_stack)
        for j = 1:n
            left = 0.0
            if j > 1
                left = u[j - 1, i_seq_level]
                push!(branch_stack, 0)
            else
                push!(branch_stack, 1)
            end
            right = 0.0
            if j < n
                right = u[j + 1, i_seq_level]
                push!(branch_stack, 0)
            else
                push!(branch_stack, 1)
            end
            fb[j, i_seq_level] = fb[j, i_seq_level] + rb[j, i_seq_level]
            tempb = -(rb[j, i_seq_level] / hl2)
            rb[j, i_seq_level] = 0.0
            ub[j, i_seq_level] = ub[j, i_seq_level] + 2.0tempb
            leftb = -tempb
            rightb = -tempb
            hl2b = hl2b - (((2.0 * u[j, i_seq_level] - left) - right) * tempb) / hl2
            branch = pop!(branch_stack)
            if branch == 0
                ub[j + 1, i_seq_level] = ub[j + 1, i_seq_level] + rightb
            end
            branch = pop!(branch_stack)
            if branch == 0
                ub[j - 1, i_seq_level] = ub[j - 1, i_seq_level] + leftb
            end
        end
        for i_seq_k = nu1:-1:1
            ad_to = pop!(integer8_stack)
            for i_seq_j = ad_to:-1:1
                tempb = 0.5 * ub[i_seq_j, i_seq_level]
                ub[i_seq_j, i_seq_level] = 0.0
                hl2b = hl2b + f[i_seq_j, i_seq_level] * tempb
                fb[i_seq_j, i_seq_level] = fb[i_seq_j, i_seq_level] + hl2 * tempb
                leftb = tempb
                rightb = tempb
                branch = pop!(branch_stack)
                if branch == 0
                    ub[i_seq_j + 1, i_seq_level] = ub[i_seq_j + 1, i_seq_level] + rightb
                end
                branch = pop!(branch_stack)
                if branch == 0
                    ub[i_seq_j - 1, i_seq_level] = ub[i_seq_j - 1, i_seq_level] + leftb
                end
            end
        end
        hlb = hlb + 2 * hl * hl2b
    end
    h1b = h1b + hlb
    return h1b
end
function mg_vcycle(u, f, r, nfine, num_levels, h1, nu1, nu2, n)
    n = n * 2
    nl = nfine
    hl = h1
    for i_seq_level = 1:num_levels - 1
        n = nl - 1
        hl2 = hl * hl
        for i_seq_k = 1:nu1
            for i_seq_j = 1:n
                left = 0.0
                if i_seq_j > 1
                    left = u[i_seq_j - 1, i_seq_level]
                end
                right = 0.0
                if i_seq_j < n
                    right = u[i_seq_j + 1, i_seq_level]
                end
                u[i_seq_j, i_seq_level] = 0.5 * (hl2 * f[i_seq_j, i_seq_level] + left + right)
            end
        end
        for j = 1:n
            left = 0.0
            if j > 1
                left = u[j - 1, i_seq_level]
            end
            right = 0.0
            if j < n
                right = u[j + 1, i_seq_level]
            end
            r[j, i_seq_level] = f[j, i_seq_level] - ((2.0 * u[j, i_seq_level] - left) - right) / hl2
        end
        ncg = div(nl, 2)
        nc = ncg - 1
        for j = 1:nc
            jf = j * 2
            f[j, i_seq_level + 1] = 0.25 * r[jf - 1, i_seq_level] + 0.5 * r[jf, i_seq_level] + 0.25 * r[jf + 1, i_seq_level]
        end
        for j = 1:nc
            u[j, i_seq_level + 1] = 0.0
        end
        nl = ncg
        hl = hl * 2.0
    end
    hl2 = hl * hl
    u[1, num_levels] = 0.5 * hl2 * f[1, num_levels]
    for i_seq_level = num_levels - 1:-1:1
        nl = nl * 2
        hl = hl / 2.0
        n = nl - 1
        ncg = div(nl, 2)
        nc = ncg - 1
        hl2 = hl * hl
        for j = 1:nc
            jf = j * 2
            u[jf, i_seq_level] = u[jf, i_seq_level] + u[j, i_seq_level + 1]
        end
        for j = 1:nc + 1
            jf = j * 2 - 1
            cl = 0.0
            if j > 1
                cl = u[j - 1, i_seq_level + 1]
            end
            cr = 0.0
            if j <= nc
                cr = u[j, i_seq_level + 1]
            end
            u[jf, i_seq_level] = u[jf, i_seq_level] + 0.5 * (cl + cr)
        end
        for i_seq_k = 1:nu2
            for i_seq_j = 1:n
                left = 0.0
                if i_seq_j > 1
                    left = u[i_seq_j - 1, i_seq_level]
                end
                right = 0.0
                if i_seq_j < n
                    right = u[i_seq_j + 1, i_seq_level]
                end
                u[i_seq_j, i_seq_level] = 0.5 * (hl2 * f[i_seq_j, i_seq_level] + left + right)
            end
        end
    end
end


# --- from normcomp_b.jl ---
function initstacks_normcomp_b()
    return
end
function normcomp_b(loss, lossb, u, ub, v, vb, w, wb, i_n)
    for i_x = 1:i_n
        w[i_x] = u[i_x] - v[i_x]
    end
    for i_seq_x = i_n:-1:1
        wb[i_seq_x] = wb[i_seq_x] + 2 * w[i_seq_x] * lossb[1]
    end
    for i_x = 1:i_n
        ub[i_x] = ub[i_x] + wb[i_x]
        vb[i_x] = vb[i_x] - wb[i_x]
        wb[i_x] = 0.0
    end
    return 
end
function normcomp(loss, u, v, w, i_n)
    for i_x = 1:i_n
        w[i_x] = u[i_x] - v[i_x]
    end
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + w[i_seq_x] ^ 2
    end
end


# --- from pipeline_b.jl ---
function initstacks_pipeline_b()
    return
end
function pipeline_b(loss, lossb, u, ub, v, vb, w, wb, i_n)
    for i_x = 1:i_n
        v[i_x] = u[i_x] ^ 2 + 1.0
    end
    for i_seq_x = i_n:-1:1
        wb[i_seq_x] = wb[i_seq_x] + lossb[1]
    end
    for i_x = 1:i_n
        vb[i_x] = vb[i_x] + u[i_x] * wb[i_x]
        ub[i_x] = ub[i_x] + v[i_x] * wb[i_x]
        wb[i_x] = 0.0
    end
    for i_x = 1:i_n
        ub[i_x] = ub[i_x] + 2 * u[i_x] * vb[i_x]
        vb[i_x] = 0.0
    end
    return 
end
function pipeline(loss, u, v, w, i_n)
    for i_x = 1:i_n
        v[i_x] = u[i_x] ^ 2 + 1.0
    end
    for i_x = 1:i_n
        w[i_x] = v[i_x] * u[i_x]
    end
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + w[i_seq_x]
    end
end


# --- from quadloss_b.jl ---
function initstacks_quadloss_b()
    return
end
function quadloss_b(loss, lossb, x, xb, y, yb, z, zb)
    xb = xb + 2 * x * y * lossb[1]
    yb = yb + (x ^ 2 - z * 3.0) * lossb[1]
    zb = zb + (3 * z ^ 2 - y * 3.0) * lossb[1]
    lossb[1] = 0.0
    return xb,yb,zb
end
function quadloss(loss, x, y, z)
    loss[1] = (x ^ 2 * y - 3.0 * y * z) + z ^ 3
end


# --- from relu_field_b.jl ---
function initstacks_relu_field_b()
    return
end
function relu_field_b(loss, lossb, u, ub, v, vb, i_n)
    for i_seq_x = i_n:-1:1
        vb[i_seq_x] = vb[i_seq_x] + lossb[1]
    end
    for i_x = 1:i_n
        if u[i_x] > 0.0
            ub[i_x] = ub[i_x] + 2 * u[i_x] * vb[i_x]
            vb[i_x] = 0.0
        else
            vb[i_x] = 0.0
        end
    end
    return 
end
function relu_field(loss, u, v, i_n)
    for i_x = 1:i_n
        if u[i_x] > 0.0
            v[i_x] = u[i_x] ^ 2
        else
            v[i_x] = 0.0
        end
    end
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + v[i_seq_x]
    end
end


# --- from stencil_loss_b.jl ---
function initstacks_stencil_loss_b()
    return
end
function stencil_loss_b(loss, lossb, u, ub, w, wb, i_n)
    for i_x = 2:i_n - 1
        w[i_x] = (u[i_x - 1] - 2.0 * u[i_x]) + u[i_x + 1]
    end
    for i_seq_x = i_n - 1:-1:2
        wb[i_seq_x] = wb[i_seq_x] + 2 * w[i_seq_x] * lossb[1]
    end
    for i_x = 2:i_n - 1
        ub[i_x - 1] = ub[i_x - 1] + wb[i_x]
        ub[i_x] = ub[i_x] - 2.0 * wb[i_x]
        ub[i_x + 1] = ub[i_x + 1] + wb[i_x]
        wb[i_x] = 0.0
    end
    return 
end
function stencil_loss(loss, u, w, i_n)
    for i_x = 2:i_n - 1
        w[i_x] = (u[i_x - 1] - 2.0 * u[i_x]) + u[i_x + 1]
    end
    for i_seq_x = 2:i_n - 1
        loss[1] = loss[1] + w[i_seq_x] ^ 2
    end
end


# --- from sumsq_shifted_b.jl ---
function initstacks_sumsq_shifted_b()
    return
end
function sumsq_shifted_b(loss, lossb, u, ub, alpha, alphab, beta, betab, i_n)
    for i_seq_x = i_n:-1:1
        tempb = 2 * (alpha * u[i_seq_x] + beta) * lossb[1]
        alphab = alphab + u[i_seq_x] * tempb
        ub[i_seq_x] = ub[i_seq_x] + alpha * tempb
        betab = betab + tempb
    end
    return alphab,betab
end
function sumsq_shifted(loss, u, alpha, beta, i_n)
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + (alpha * u[i_seq_x] + beta) ^ 2
    end
end


# --- from two_field_loss_b.jl ---
function initstacks_two_field_loss_b()
    return
end
function two_field_loss_b(loss, lossb, u, ub, v, vb, p, pb, q, qb, i_n)
    for i_seq_x = i_n:-1:1
        pb[i_seq_x] = pb[i_seq_x] + lossb[1]
        qb[i_seq_x] = qb[i_seq_x] + lossb[1]
    end
    for i_x = 1:i_n
        vb[i_x] = vb[i_x] + 3 * v[i_x] ^ 2 * qb[i_x]
        qb[i_x] = 0.0
    end
    for i_x = 1:i_n
        ub[i_x] = ub[i_x] + 2 * u[i_x] * pb[i_x]
        pb[i_x] = 0.0
    end
    return 
end
function two_field_loss(loss, u, v, p, q, i_n)
    for i_x = 1:i_n
        p[i_x] = u[i_x] ^ 2
    end
    for i_x = 1:i_n
        q[i_x] = v[i_x] ^ 3
    end
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + p[i_seq_x] + q[i_seq_x]
    end
end


# --- from weightedsumsq_b.jl ---
function initstacks_weightedsumsq_b()
    return
end
function weightedsumsq_b(loss, lossb, u, ub, w, wb, i_n)
    for i_seq_x = i_n:-1:1
        wb[i_seq_x] = wb[i_seq_x] + u[i_seq_x] ^ 2 * lossb[1]
        ub[i_seq_x] = ub[i_seq_x] + 2 * u[i_seq_x] * w[i_seq_x] * lossb[1]
    end
    return 
end
function weightedsumsq(loss, u, w, i_n)
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + w[i_seq_x] * u[i_seq_x] ^ 2
    end
end


