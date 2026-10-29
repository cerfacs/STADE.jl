isdefined(Main, :STADE) || include(joinpath(@__DIR__, "..", "src", "STADE.jl"))

"""
    validate_shared_private(dir="val-corpus"; only=nothing, sabotage=true)

Structural gate for block-private accumulators in the CUDA backend. No GPU needed; `validate_corpus_gpu.jl` checks the numbers on a device.

Three groups of checks:

  1. UNITS: `cgen_private_arrays`, `cgen_private_kernel` and `cgen_launch_stmts` on hand-written kernels. An array qualifies only when the kernel
     solely accumulates into it, every index is thread-invariant or depends on the thread through `mod` by an invariant, and some index varies in
     the kernel. A gather, the thread index itself, a mod by an array read, a bare constant index, a read of the array, a local array and a kernel
     of another shape all stay as they are. The private variant has both barriers, no early return, a predicate in place of the guard, one offset
     argument per array and no site left on the global array. The host statements pack offsets that are multiples of 8 and choose the private variant
     only when the total fits the budget and every array pays off (a threads-per-element ratio, and four threads per element for an array with one add
     per thread); they are evaluated on CPU arrays.
  2. ROUND TRIP: for every corpus kernel, adjoint and HVP, dropping the `_p` variants and keeping the plain launch of each choice on the default CUDA
     output gives exactly the output with `shared_private = false`; and every variant is launched.
  3. SABOTAGE: each guard removed in a copy of STADE.jl must make a unit check fail.
"""
function private_unit_checks(S)
    bad = 0
    check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
    wa = S.cgen_warp_add_name()
    cuda = S.cgen_backend_cuda()
    src(e) = S.io_expr_to_source(e)
    add(a, i, v) = Expr(:call, wa, a, i, v)
    function kern(body...; args = [:A, :B, :x, :src, :rp, :n, :m])
        Expr(:function, Expr(:call, :kern!, args...), Expr(:block,
            :(__tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x),
            :(if __tid > n
                  return nothing
              end),
            :(i = 1 + (__tid - 1)), body..., :(return nothing)))
    end
    arrs(k) = S.cgen_private_arrays(k)
    binned = [:(k = mod(i - 1, m) + 1), Expr(:for, :(e = 1:3), Expr(:block, add(:A, :((e - 1) * m + k), :(x[e]))))]

    check(arrs(kern(binned...)) == [:A], "an index through mod by an invariant, with a loop variable, qualifies")
    check(isempty(arrs(kern(add(:A, :i, :(x[i]))))), "the thread index as the index stays global")
    check(isempty(arrs(kern(:(j = src[i]), add(:A, :j, :(x[i]))))), "an index from an array read stays global")
    check(isempty(arrs(kern(add(:A, 1, :(x[i]))))), "a bare constant index stays global (the warp helper's case)")
    check(isempty(arrs(kern(binned..., :(t = A[2])))), "a read of the array in the kernel blocks it")
    check(isempty(arrs(kern(binned..., add(:A, :i, :(x[i]))))), "one wide site among qualifying sites blocks the array")
    check(arrs(kern(Expr(:for, :(e = 1:m), Expr(:block, add(:A, :e, :(x[e])))))) == [:A], "a loop variable with invariant bounds is an index that varies")
    check(isempty(arrs(kern(:(lo = rp[i]), Expr(:for, :(e = lo:m), Expr(:block, add(:A, :e, :(x[e]))))))), "a loop variable with bounds read from an array is wide")
    check(isempty(arrs(kern(:(k = mod(i, rp[i])), add(:A, :k, :(x[i]))))), "a mod by a value read from an array is wide")
    check(isempty(arrs(kern(:(k = mod(i - 1, m) + 1), add(:C, :k, :(x[i]))))), "an array that is not a kernel argument stays")
    check(arrs(kern(binned..., add(:B, :k, :(x[i])))) == [:A, :B], "two qualifying arrays are both listed, in first-use order")
    many = Any[add(Symbol("A", t), :k, :(x[i])) for t in 1:12]
    check(length(arrs(kern(:(k = mod(i - 1, m) + 1), many...; args = [Symbol("A", t) for t in 1:12]))) == S.cgen_private_max_arrays(), "at most $(S.cgen_private_max_arrays()) arrays are privatized per kernel")
    other = Expr(:function, Expr(:call, :kern!, :A, :n), Expr(:block, :(k = 1), add(:A, :k, 1.0), :(return nothing)))
    check(isempty(arrs(other)) && S.cgen_private_kernel(other, [:A], cuda) === other, "a kernel without the standard guard is left alone")

    k = kern(binned...)
    p = S.cgen_private_kernel(k, [:A], cuda)
    ps = src(p)
    nsync = length(findall("sync_threads()", ps))
    check(nsync == 2, "the rewritten kernel has exactly two barriers ($nsync)")
    body = [x for x in p.args[2].args if !(x isa LineNumberNode)]
    guard_ok = any(x -> x isa Expr && x.head == :if && x.args[1] == :(__tid <= n), body)
    check(guard_ok && !any(x -> x isa Expr && x.head == :if && x.args[1] == :(__tid > n), body), "the thread guard became a predicate around the body")
    check(count(x -> x isa Expr && x.head == :return, body) == 1 && body[end] == :(return nothing), "the only return is the last statement")
    check(p.args[1].args[1] == :kern_p! && p.args[1].args[end] == :__po_A && length(p.args[1].args) == length(k.args[1].args) + 1, "the variant is named kern_p! and takes one offset argument per array")
    check(!occursin("__po_A >= 0", ps) && occursin("__sh_A", ps) && occursin("CuDynamicSharedArray", ps), "the variant has no branch on the offset and uses the shared copy")
    check(!occursin("A[", replace(ps, "__sh_A[" => "")) || occursin("A[__q] +=", ps), "the only global access to A is the flush")
    check(occursin(string(wa) * "(__sh_A", ps), "a warp-helper site keeps the helper on the private array (the helper sums a warp's values before the shared atomic)")
    check(S.cgen_private_kernel(k, Symbol[], cuda) === k, "no arrays: the kernel is returned unchanged")
    check(!S.cgen_private_straight(k, :A), "an accumulation under a loop is not straight-line")
    check(S.cgen_private_straight(kern(:(k = mod(i - 1, m) + 1), add(:A, :k, :(x[i]))), :A), "an accumulation under no loop is straight-line")
    check(S.cgen_private_straight(kern(:(k = mod(i - 1, m) + 1), :(if x[i] > 0.0
              $(add(:A, :k, :(x[i])))
          end)), :A), "an accumulation under an if only is straight-line")

    # host statements, evaluated on CPU arrays: returns (total bytes, offsets..., the condition that picks the private variant)
    stmt = (kind = :for, var = :i, lo = 1, hi = :n, step = 1, body = NamedTuple[])
    function offsets(lens...; names = [:A, :B, :C][1:length(lens)], T = Float64, n = 10^6, ntb = 256, straight = Symbol[], be = cuda)
        stmts = S.cgen_launch_stmts(stmt, :kern, 3, Symbol[names..., :n], be, Symbol[names...], Set{Symbol}(straight))
        ifs = stmts[end]
        fn = Expr(:function, Expr(:call, :f__, names..., :n, :nthread_per_block), Expr(:block, stmts[1:end-1]..., Expr(:tuple, Symbol("__pn_3"), [Symbol("__po_", a, "_3") for a in names]..., ifs.args[1])))
        return Base.invokelatest(eval(fn), [zeros(T, l) for l in lens]..., n, ntb), ifs
    end
    (r, ifs) = offsets(100, 200)
    check(r == (2400, 0, 800, true), "offsets pack the arrays in order: A at 0, B at 800, total $(r[1]) bytes; the private variant is chosen")
    check(length(ifs.args) == 3 && occursin("shmem = __pn_3", src(ifs.args[2])) && occursin("kern_3_p!", src(ifs.args[2])) && occursin("__po_A_3, __po_B_3", src(ifs.args[2])),
          "the then branch launches the variant with the shared size and the offsets")
    check(!occursin("shmem", src(ifs.args[3])) && !occursin("_p!", src(ifs.args[3])) && occursin("kern_3!(", src(ifs.args[3])), "the else branch launches the original kernel unchanged")
    (r, _) = offsets(100, 2000)
    check(r[end] == false, "an array that does not fit the budget sends the launch to the original kernel")
    (r, _) = offsets(600, 600)
    check(r[1] == 9600 && r[end] == false, "a total above the budget ($(r[1]) of $(S.cgen_private_max_bytes()) bytes) is refused")
    (r, _) = offsets(3, 3; T = Float32, n = 1000)
    check(r[1:3] == (32, 0, 16) && r[end] == true, "three 4-byte elements occupy 16 bytes: offsets stay multiples of 8")
    (r, _) = offsets(100, 50; n = 3199)
    check(r[end] == false, "fewer than $(S.cgen_private_min_ratio(cuda)) threads per element of the larger array is refused (3199 < 3200)")
    (r, _) = offsets(100, 50; n = 3200)
    check(r[end] == true, "at $(S.cgen_private_min_ratio(cuda)) threads per element the array is kept (3200)")
    (r, _) = offsets(100; straight = [:A])
    check(r[end] == false, "a straight-line array of 100 elements in blocks of 256 is refused (4 * 100 > 256)")
    (r, _) = offsets(64; straight = [:A])
    check(r[end] == true, "a straight-line array of 64 elements in blocks of 256 is kept (4 * 64 = 256)")
    (r, _) = offsets(100; straight = Symbol[])
    check(r[end] == true, "an array accumulated in a loop does not need the block test")
    (r, _) = offsets(100; be = merge(cuda, (private_min_ratio = 7,)), n = 699)
    check(r[end] == false, "a backend ratio of 7 refuses 100 elements at 699 threads")
    (r, _) = offsets(100; be = merge(cuda, (private_min_ratio = 7,)), n = 700)
    check(r[end] == true, "a backend ratio of 7 keeps 100 elements at 700 threads")
    (r, _) = offsets(100; be = merge(cuda, (private_min_ratio = 0,)), n = 1)
    check(r[end] == true, "a ratio of 0 privatizes whenever the array fits")
    (r, _) = offsets(100; be = merge(cuda, (private_min_ratio = 0,)), n = 1, straight = [:A])
    check(r[end] == true, "a ratio of 0 also turns the block test off")
    plan7 = S.stade_gpu(:(function f__(A, n)
        for i = 1:n
            A[mod(i, 4) + 1] += 1.0
        end
        return nothing
    end), cuda; shared_private = 7)
    check(occursin("7 * length(A)", src(plan7.host)), "shared_private = 7 reaches the launch test")
    check(!occursin("shmem", src(S.cgen_launch_stmts(stmt, :kern, 3, Symbol[:A, :n], cuda)[end])) && length(S.cgen_launch_stmts(stmt, :kern, 3, Symbol[:A, :n], cuda)) == 1, "no private arrays: the plain launch alone")
    return bad
end

# Inverse of the rewrite on generated code. Kernels: drop the `_p` variants. Host: drop the offset statements and keep the plain launch of each choice.
private_is_variant(k) = k isa Expr && k.head == :function && endswith(string(k.args[1].args[1]), "_p!")

function private_invert_host(e)
    e isa Expr || return e
    out = Any[]
    for a in e.args
        a isa LineNumberNode && continue
        if a isa Expr && a.head == :(=) && a.args[1] isa Symbol && (startswith(string(a.args[1]), "__pn_") || startswith(string(a.args[1]), "__po_"))
            continue
        elseif a isa Expr && a.head == :if && length(a.args) == 3 && a.args[2] isa Expr && any(x -> x isa Expr && x.head == :macrocall && any(y -> y isa Expr && y.head == :(=) && y.args[1] === :shmem, x.args), a.args[2].args)
            for x in a.args[3].args
                x isa LineNumberNode || push!(out, x)
            end
        else
            push!(out, private_invert_host(a))
        end
    end
    return Expr(e.head, out...)
end

function validate_shared_private(dir::String = joinpath(@__DIR__, "val-corpus"); only::Union{Vector{String},Nothing} = nothing, sabotage::Bool = true)
    bad = 0
    check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
    println("-- unit checks")
    bad += private_unit_checks(STADE)
    println("-- round trip over the corpus")
    cuda = STADE.cgen_backend_cuda()
    src(e) = STADE.io_expr_to_source(e)
    names = sort([splitext(f)[1] for f in readdir(dir) if endswith(f, ".jl") && !any(endswith(f, s) for s in ("_b.jl", "_d.jl", "_hv.jl", "_cuda.jl", "_jacc.jl"))])
    only !== nothing && (names = filter(n -> n in only, names))
    plans = 0; changed = 0; arrays = 0
    for name in names
        pe = STADE.io_read_corpus_entry(joinpath(dir, name * ".jl"))
        for (mode, gen) in (("adjoint", STADE.stade_adjoint), ("hvp", STADE.stade_hvp)), fuse in (false, true)
            out = try gen(pe; keep_push_pop = false, fuse_ii_loops = fuse) catch; continue end
            e = mode == "hvp" ? out.hvp : out.adjoint
            on = STADE.stade_gpu(e, cuda); off = STADE.stade_gpu(e, cuda; shared_private = false)
            plans += 1
            variants = [k for k in on.kernels if private_is_variant(k)]
            plain = [k for k in on.kernels if !private_is_variant(k)]
            n = sum(count(x -> startswith(string(x), "__po_"), k.args[1].args[2:end]) for k in variants; init = 0)
            arrays += n; n > 0 && (changed += 1)
            same = length(plain) == length(off.kernels) && all(src(a) == src(b) for (a, b) in zip(plain, off.kernels)) && src(private_invert_host(on.host)) == src(off.host)
            check(same, "$name [$mode f=$(Int(fuse))]: $n private array(s) in $(length(variants)) variant(s), dropping the variants and keeping the plain launches gives the output of shared_private = false")
            host_text = src(on.host)
            check(all(k -> occursin(string(k.args[1].args[1]), host_text), variants), "$name [$mode f=$(Int(fuse))]: every private variant is launched")
            amd = STADE.stade_gpu(e, STADE.cgen_backend_amdgpu())
            check(!any(k -> occursin("CuDynamicSharedArray", src(k)), amd.kernels), "$name [$mode f=$(Int(fuse))]: AMDGPU output has no shared memory")
        end
    end
    println("\n$arrays private array(s) in $changed of $plans plan(s)")
    only === nothing && check(arrays > 0, "at least one corpus kernel privatizes an array")
    if sabotage
        repo = abspath(joinpath(@__DIR__, ".."))
        text = read(joinpath(repo, "src", "STADE.jl"), String)
        cases = [
            ("the kernel only accumulates into the array", "        cgen_count_symbol(whole, a) == length(mine) || continue\n", ""),
            ("no wide index", "        all(st -> cgen_expr_class(st.idx, env) !== :wide, mine) || continue\n", ""),
            ("the index must vary", "        any(st -> !isempty(intersect(cgen_expr_symbols(st.idx), assigned)), mine) || continue\n", ""),
            ("an array read is wide", "    e.head == :ref && return :wide\n", "    e.head == :ref && return :inv\n"),
            ("a mod by a variant is wide", "            cb === :inv || return :wide\n", ""),
            ("a kernel argument", "        a in fargs || continue\n", ""),
            ("cap on arrays", "        length(out) >= cgen_private_max_arrays() && break\n", ""),
            ("predicate around the body", "Expr(:if, Expr(:call, :<=, :__tid, n_iter), Expr(:block, rewrite(parts.rest)...))", "Expr(:if, Expr(:call, :<, :__tid, n_iter), Expr(:block, rewrite(parts.rest)...))"),
            ("barrier before the flush", "    push!(body, Expr(:call, :sync_threads))\n    append!(body, post)", "    append!(body, post)"),
            ("a warp site keeps the helper on the private array", "push!(out, e.head == :call ? Expr(:call, e.args[1], sh_arr, site.idx, site.val) :", "push!(out, false ? Expr(:call, e.args[1], sh_arr, site.idx, site.val) :"),
            ("worth test in the launch", "push!(conds, Expr(:call, :>=, n_iter, Expr(:call, :*, ratio, Expr(:call, :length, a))))", "push!(conds, true)"),
            ("block test for a straight-line array", "a in straight && push!(conds,", "false && push!(conds,"),
            ("budget in the launch", "fits = Expr(:call, :<=, pn, cgen_private_max_bytes())", "fits = true"),
            ("the else branch is the original kernel", "emit_if(cond, Any[private], plain_stmts)", "emit_if(cond, Any[private], Any[private])"),
            ("offsets rounded to 8 bytes", "Expr(:call, :*, 8, Expr(:call, Expr(:., :Base, QuoteNode(:cld)), Expr(:call, :*, Expr(:call, :length, a), Expr(:call, :sizeof, Expr(:call, :eltype, a))), 8))", "Expr(:call, :*, Expr(:call, :length, a), Expr(:call, :sizeof, Expr(:call, :eltype, a)))"),
            ("a loop counts as under-loop", "                under_loop(e.args[2].args, true) && return true", "                under_loop(e.args[2].args, false) && return true"),
        ]
        mktempdir() do tmp
            for (label, old, new) in cases
                count_old = length(findall(old, text))
                if count_old != 1
                    check(false, "sabotage \"$label\": anchor found $count_old times")
                    continue
                end
                path = joinpath(tmp, "STADE_sab.jl")
                write(path, replace(text, old => new; count = 1))
                script = """
                    include($(repr(path)))
                    include($(repr(joinpath(@__DIR__, "validate_shared_private.jl"))))
                    println("RESULT ", private_unit_checks(STADE))
                """
                out = try read(pipeline(`$(Base.julia_cmd()) --startup-file=no -e $script`; stderr = devnull), String) catch e; "ERR" end
                m = match(r"RESULT (\d+)", out)
                check(m !== nothing && parse(Int, m.captures[1]) > 0, "sabotage \"$label\": a unit check fails")
            end
        end
    end
    println("\n", bad == 0 ? "all shared private checks passed" : "*** $bad shared private check(s) failed ***")
    return bad
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(validate_shared_private(; only = isempty(ARGS) ? nothing : ARGS) == 0 ? 0 : 1)
end
