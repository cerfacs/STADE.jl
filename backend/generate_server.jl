using HTTP
using JSON3
using Dates
using Random # STADE.jl's val_* finite-difference oracle calls randn/randn -- it's
             # included (not `using`d) into this module, so it resolves that
             # global lazily against whatever's loaded here by the time it's
             # actually called, same as src/validate_corpus.jl already does.

# ------------------------------------------------------------------
# Load STADE.jl itself. backend/ and src/ are siblings in the repo, so
# resolve the path relative to this file (not the process's current
# working directory) so it works the same whether the server is
# started as `julia generate_server.jl` from backend/, or from
# anywhere else (e.g. a Docker CMD).
# ------------------------------------------------------------------
include(joinpath(@__DIR__, "..", "src", "STADE.jl"))

# ------------------------------------------------------------------
# Web-friendly version of the "generate everything" flow.
#
# Every `stade_*_file` entry point in STADE.jl is path-in/path-out --
# that only makes sense when Julia has its own filesystem to work
# with. A browser only ever hands us the *text* in the left-hand
# pane, so we write that text to a temp file once, run each
# generator against it (writing its own temp output file), read the
# result back, and concatenate everything into one string with a
# banner comment marking where each section starts. No disk I/O is
# ever visible outside this function -- the temp dir is removed
# again before it returns.
#
# The seven generators, in the order they're concatenated:
#   tangent, adjoint, hvp          -- stade_*_file (AD, via io_read_kernel_corpus)
#   cuda, amdgpu, metal, jacc      -- stade_*_file (GPU porting, via io_read_kernel_bundle)
#
# A single generator failing (e.g. GPU porting on a multi-kernel
# `*_multi.jl`-style corpus -- a documented gap, see
# src/validate_corpus.jl's `validate_gpu_ports` docstring) doesn't
# abort the whole request: that section's banner is followed by the
# error message instead of code, and every other section still runs.
# ------------------------------------------------------------------
const GENERATORS = [
    (:tangent, stade_tangent_file),
    (:adjoint, stade_adjoint_file),
    (:hvp,     stade_hvp_file),
    (:cuda,    stade_cuda_file),
    (:amdgpu,  stade_amdgpu_file),
    (:metal,   stade_metal_file),
    (:jacc,    stade_jacc_file),
]

function banner(label::Symbol)
    title = " $(uppercase(string(label))) "
    return "# " * "="^20 * title * "="^20
end

function generate_content(original_content::AbstractString)::String
    mktempdir() do dir
        entry = try
            corpus_entry_name(original_content)
        catch
            "input"
        end
        in_path = joinpath(dir, entry * ".jl")
        write(in_path, original_content)

        sections = String[]
        for (label, fn) in GENERATORS
            out_path = joinpath(dir, "output_$(label).jl")
            body = try
                fn(in_path, out_path)
                read(out_path, String)
            catch err
                "# generation failed: $(sprint(showerror, err))\n"
            end
            push!(sections, banner(label) * "\n" * body)
        end

        return join(sections, "\n")
    end
end

# ------------------------------------------------------------------
# Web-friendly version of the "validate everything" flow.
#
# stade_validate_tangent_file/stade_validate_adjoint_file/stade_validate_hvp_file
# each already implement exactly the "use the baseline YAML if it's
# there, generate one from scratch (via stade_generate_baseline_file,
# which is what io_write_baseline_yaml/io_read_baseline_yaml exist
# for) if it isn't" policy on their own -- see their bodies in
# src/STADE.jl. So the only thing this handler does on top of calling
# them is choose whether that YAML file exists on disk before the
# calls: written from yaml_content first when the person already has
# something in the YAML pane (so their edited values are used
# untouched, never regenerated), or left absent so the first call's
# own stade_generate_baseline_file fires and every later call in the
# same request reuses exactly what it wrote.
#
# The three validators share one baseline file, so tangent, adjoint,
# and hvp are all checked against the same input values.
# ------------------------------------------------------------------
const VALIDATORS = [
    (:tangent, stade_validate_tangent_file),
    (:adjoint, stade_validate_adjoint_file),
    (:hvp,     stade_validate_hvp_file),
]

# a multi-kernel corpus needs its temp file named after its own entry
# kernel -- the one that calls the others rather than being called
# itself -- because io_read_corpus_entry (which stade_generate_baseline_file
# and stade_validate_from_baseline both go through) resolves the entry
# kernel from the file's own basename for exactly that case (see its
# docstring in src/STADE.jl). A single-kernel file has no such
# ambiguity, so its own name always works.
function corpus_entry_name(content::AbstractString)::String
    parsed = Meta.parseall(content)
    defs = [e for e in parsed.args if e isa Expr && e.head == :function]
    isempty(defs) && return "input"
    kernels = Dict{Symbol,Expr}()
    for def in defs
        name, _ = parse_signature(def.args[1])
        haskey(kernels, name) && return "input" # duplicate name -- let the real read fail with a clearer error
        kernels[name] = def
    end
    length(kernels) == 1 && return string(first(keys(kernels)))
    graph, _ = inl_build_call_graph(kernels)
    called = union(values(graph)...)
    roots = [k for k in keys(kernels) if !(k in called)]
    length(roots) == 1 && return string(roots[1])
    return string(first(sort(collect(keys(kernels)); by = string))) # ambiguous -- best effort, let stade_* raise its own error
end

function format_validation_result(result)::String
    status = result.ok ? "PASS" : "FAIL"
    lines = String[
        "status: $status",
        "max relative error over $(length(result.trials)) trials: $(result.max_rel_err)",
    ]
    for (i, t) in enumerate(result.trials)
        push!(lines, "  trial $(i): rel_err = $(t.rel_err)")
    end
    return join(lines, "\n")
end

# Returns (; result, yaml) -- `yaml` is the baseline text to drop into
# the YAML pane, or `nothing` when yaml_content already had something
# in it (so the frontend leaves the pane exactly as the person left it).
function validate_content(source_content::AbstractString, yaml_content::AbstractString)
    mktempdir() do dir
        entry = try
            corpus_entry_name(source_content)
        catch
            "input"
        end
        in_path = joinpath(dir, entry * ".jl")
        write(in_path, source_content)

        yaml_path = joinpath(dir, entry * ".yaml")
        user_supplied_yaml = !isempty(strip(yaml_content))
        user_supplied_yaml && write(yaml_path, yaml_content)

        sections = String[]
        for (label, fn) in VALIDATORS
            body = try
                result = fn(in_path; yaml_path = yaml_path)
                format_validation_result(result)
            catch err
                "generation failed: $(sprint(showerror, err))"
            end
            push!(sections, banner(label) * "\n" * body)
        end

        generated_yaml = user_supplied_yaml ? nothing : (isfile(yaml_path) ? read(yaml_path, String) : nothing)
        return (result = join(sections, "\n\n"), yaml = generated_yaml)
    end
end

# ------------------------------------------------------------------
# CORS
#
# GitHub Pages and this API live on different origins, so browsers
# will block the request unless we send CORS headers. "*" is fine for
# a demo; for anything real, replace it with your exact Pages origin,
# e.g. "https://YOUR_GITHUB_USERNAME.github.io".
# ------------------------------------------------------------------
const ALLOWED_ORIGIN = "https://cerfacs.github.io/STADE.jl"
# const ALLOWED_ORIGIN = "*"

function cors_headers()
    return [
        "Access-Control-Allow-Origin"  => ALLOWED_ORIGIN,
        "Access-Control-Allow-Methods" => "POST, OPTIONS",
        "Access-Control-Allow-Headers" => "Content-Type",
    ]
end

function handle_generate(req::HTTP.Request)
    if req.method == "OPTIONS"
        return HTTP.Response(204, cors_headers())
    end

    try
        body = JSON3.read(String(req.body))
        content = get(body, :content, nothing)
        if content === nothing || !(content isa AbstractString)
            return HTTP.Response(400, cors_headers(), body = JSON3.write((; error = "missing 'content' string field")))
        end

        result = generate_content(content)
        return HTTP.Response(200, cors_headers(),
                              body = JSON3.write((; result = result)))
    catch err
        return HTTP.Response(500, cors_headers(),
                              body = JSON3.write((; error = "server error: $(sprint(showerror, err))")))
    end
end

function handle_health(::HTTP.Request)
    return HTTP.Response(200, cors_headers(), body = JSON3.write((; status = "ok")))
end

function handle_validate(req::HTTP.Request)
    if req.method == "OPTIONS"
        return HTTP.Response(204, cors_headers())
    end

    try
        body = JSON3.read(String(req.body))
        content = get(body, :content, nothing)
        if content === nothing || !(content isa AbstractString)
            return HTTP.Response(400, cors_headers(), body = JSON3.write((; error = "missing 'content' string field")))
        end
        yaml = get(body, :yaml, "")
        yaml isa AbstractString || (yaml = "")

        outcome = validate_content(content, yaml)
        return HTTP.Response(200, cors_headers(),
                              body = JSON3.write((; result = outcome.result, yaml = outcome.yaml)))
    catch err
        return HTTP.Response(500, cors_headers(),
                              body = JSON3.write((; error = "server error: $(sprint(showerror, err))")))
    end
end

router = HTTP.Router()
HTTP.register!(router, "POST", "/generate", handle_generate)
HTTP.register!(router, "OPTIONS", "/generate", handle_generate)
HTTP.register!(router, "POST", "/validate", handle_validate)
HTTP.register!(router, "OPTIONS", "/validate", handle_validate)
HTTP.register!(router, "GET", "/health", handle_health)

port = parse(Int, get(ENV, "PORT", "8081"))
println("generate_file backend listening on 0.0.0.0:$port")
HTTP.serve(router, "0.0.0.0", port)