$ErrorActionPreference = 'Stop'
$root = Resolve-Path (Join-Path $PSScriptRoot '..')
$binary = Join-Path $root 'results\router_scalability.exe'
$output = Join-Path $root 'results\router_scalability.csv'
$queryRoot = Join-Path $root 'results\router-cpp-queries'

g++ -O3 -std=c++17 -pthread (Join-Path $PSScriptRoot 'router_scalability.cpp') -o $binary
if ($LASTEXITCODE -ne 0) { throw 'C++ compilation failed' }

'dataset,n_edges,n_unique_queries,n_requests,fallback_rate,workers,trial,qps,latency_p50_us,latency_p95_us,latency_p99_us,checksum' | Set-Content -Encoding ascii $output
$datasets = @(
    'email_eu',
    'college_msg',
    'wiki_talk',
    'reddit_hyperlinks',
    'sx_superuser',
    'sx_stackoverflow_a2q'
)
foreach ($name in $datasets) {
    & $binary $name (Join-Path $queryRoot "$name.times.i64") (Join-Path $queryRoot "$name.state.i64") (Join-Path $queryRoot "$name.csv") 1000000000 $output
    if ($LASTEXITCODE -ne 0) { throw "Benchmark failed for $name" }
}
