FROM julia:1.10

WORKDIR /app

# Install dependencies first so Docker can cache this layer
# across deploys where only generate_server.jl changes.
COPY Project.toml .
RUN julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

COPY generate_server.jl .

# Render injects its own $PORT at runtime; generate_server.jl already
# reads ENV["PORT"] with a fallback, so nothing else to configure here.
ENV PORT=8081
EXPOSE 8081

CMD ["julia", "--project=.", "generate_server.jl"]