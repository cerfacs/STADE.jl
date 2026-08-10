using HTTP
using JSON3
using Dates

# ------------------------------------------------------------------
# Pure, web-friendly version of the logic in src/generate_file.jl
#
# The original `generate_file(input_path)` reads a path from disk and
# writes a new file next to it — that only makes sense when Julia has
# its own filesystem to work with. A browser only ever hands us the
# *text* that's in the left-hand pane, so the function this server
# calls takes a string and returns a string, with no disk I/O at all.
#
# If you also want the original file-based version for local/CLI use,
# keep it in src/generate_file.jl and have it call this function too:
#
#     new_content = generate_content(read(input_path, String))
# ------------------------------------------------------------------
function generate_content(original_content::AbstractString)::String
    time_str = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
    return "# GENERATED FILE @ $time_str\n" * original_content
end

# ------------------------------------------------------------------
# CORS
#
# GitHub Pages and this API live on different origins, so browsers
# will block the request unless we send CORS headers. "*" is fine for
# a demo; for anything real, replace it with your exact Pages origin,
# e.g. "https://YOUR_GITHUB_USERNAME.github.io".
# ------------------------------------------------------------------
# const ALLOWED_ORIGIN = "https://luciano-drozda.github.io"
const ALLOWED_ORIGIN = "*"

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

router = HTTP.Router()
HTTP.register!(router, "POST", "/generate", handle_generate)
HTTP.register!(router, "OPTIONS", "/generate", handle_generate)
HTTP.register!(router, "GET", "/health", handle_health)

port = parse(Int, get(ENV, "PORT", "8081"))
println("generate_file backend listening on 0.0.0.0:$port")
HTTP.serve(router, "0.0.0.0", port)
