# ============================================================
# check_codegen_call_tree.jl -- end-to-end correctness check for STADE's
# agen_* codegen: generates _b/initstacks_ (adjoint) 
# Julia code from the real corpus primals, `eval`s it, and
# checks the ADJOINT against `val_finite_diff_check`
#
# Run with:
#     julia check_codegen_call_tree.jl
# from a directory containing STADE.jl and all_b.jl.
# ============================================================


using Core.Compiler: MethodInstance

"""
    call_tree(f, argtypes::Tuple; seen=Dict())

Recursively build the call tree of `f(argtypes...)`, skipping Base/Core.
Returns a nested Dict you can print or serialize.
"""
function call_tree(f, argtypes::Tuple; seen=Dict{Tuple{Symbol,Module,Tuple},Bool}())
    node = Dict{String,Any}(
        "name"     => string(f),
        "argtypes" => string(argtypes),
        "children" => Any[],
    )

    cis = try
        code_typed(f, argtypes; optimize=true)   # optimize=true is required for :invoke
    catch
        return node
    end
    isempty(cis) && return node

    ci, _ = cis[1]
    for stmt in ci.code
        if stmt isa Expr && stmt.head === :invoke
            mi  = stmt.args[1]::MethodInstance
            m   = mi.def
            mod = m.module

            if mod === Base || mod === Core || mod === Core.Compiler
                continue
            end

            fname = m.name
            childargtypes = Tuple(mi.specTypes.parameters[2:end])
            key = (fname, mod, childargtypes)

            if haskey(seen, key)
                push!(node["children"], Dict("name" => string(fname), "note" => "already visited"))
            else
                seen[key] = true
                ff = getfield(mod, fname)
                push!(node["children"], call_tree(ff, childargtypes; seen=seen))
            end
        end
    end
    return node
end

function print_tree(io, node, depth=0)
    println(io, "  "^depth, node["name"], "(", get(node, "argtypes", ""), ")",
                 haskey(node, "note") ? "  # $(node["note"])" : "")
    for child in get(node, "children", [])
        print_tree(io, child, depth + 1)
    end
end

isdefined(Main, :agen_emit) || include(joinpath(@__DIR__, "STADE.jl"))

function grab_kernel_expr(path::String, name::Symbol)
    parsed = Meta.parseall(read(path, String))
    for e in parsed.args
        if e isa Expr && e.head == :function
            sig = e.args[1]
            if sig isa Expr && sig.head == :call && sig.args[1] == name
                return e
            end
        end
    end
    error("no `function $name(...)` found in $path")
end

const ALL_B_PATH = joinpath(@__DIR__, "all_b.jl")

# generate adjoint + initstacks for `name`, eval them into
# Main alongside the primal itself, and return the generated Exprs
# (handy for eyeballing on failure);
# also write them as files on disk.
function generate_and_eval(name::Symbol)
    primal_expr = grab_kernel_expr(ALL_B_PATH, name)
    adjoint_out = stade_adjoint(primal_expr)
    name_str    = String(name)
    io_write_kernel_file(name_str * "_b.jl", primal_expr, [adjoint_out.initstacks, adjoint_out.adjoint])
    Base.eval(Main, primal_expr)
    Base.eval(Main, adjoint_out.initstacks)
    Base.eval(Main, adjoint_out.adjoint)
    return (adjoint = adjoint_out.adjoint, initstacks = adjoint_out.initstacks)
end

function report(name, fx; trials = 10)
    r = val_check_fixture(fx; trials = trials)
    status = r.ok ? "ok  " : "FAIL"
    println(rpad(name, 22), status, "  max_rel_err=", round(r.max_rel_err, sigdigits = 3))
    return r
end

# generate_and_eval(:calc)

# Write call tree for 'generate_and_eval'
tree = call_tree(generate_and_eval, (Symbol,))
open("call_tree.txt", "w") do io
    print_tree(io, tree)
end