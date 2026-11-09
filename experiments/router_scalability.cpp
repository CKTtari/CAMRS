// C++17 scalability benchmark for the CAMRS reliability-aware COUNT router.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

using Clock = std::chrono::steady_clock;

struct Query {
    std::int64_t left{};
    std::int64_t right{};
    double fraction{};
    double bound{};
    bool fallback{};
};

struct Triple {
    std::int64_t source{};
    std::int64_t destination{};
    std::int64_t time{};
    bool operator==(const Triple& other) const {
        return source == other.source && destination == other.destination && time == other.time;
    }
};

struct TripleHash {
    std::size_t operator()(const Triple& value) const {
        auto mix = [](std::uint64_t x) {
            x += 0x9e3779b97f4a7c15ULL;
            x = (x ^ (x >> 30U)) * 0xbf58476d1ce4e5b9ULL;
            x = (x ^ (x >> 27U)) * 0x94d049bb133111ebULL;
            return x ^ (x >> 31U);
        };
        return static_cast<std::size_t>(mix(value.source) ^ (mix(value.destination) << 1U) ^ (mix(value.time) << 2U));
    }
};

static std::vector<std::int64_t> load_times(const std::string& path, std::size_t row_limit) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("cannot open dataset: " + path);
    const auto bytes = static_cast<std::size_t>(input.tellg());
    input.seekg(0);
    const auto available = bytes / sizeof(std::int64_t);
    const auto count = std::min(available, row_limit);
    std::vector<std::int64_t> times(count);
    input.read(reinterpret_cast<char*>(times.data()), static_cast<std::streamsize>(count * sizeof(std::int64_t)));
    if (!input || times.empty()) throw std::runtime_error("cannot read binary timestamps: " + path);
    return times;
}

static std::vector<Query> load_queries(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open queries: " + path);
    std::vector<Query> queries;
    std::string line;
    std::getline(input, line);
    while (std::getline(input, line)) {
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream parser(line);
        Query query;
        int fallback = 0;
        if (parser >> query.left >> query.right >> query.fraction >> query.bound >> fallback) {
            query.fallback = fallback != 0;
            queries.push_back(query);
        }
    }
    if (queries.empty()) throw std::runtime_error("no queries loaded: " + path);
    return queries;
}

class CountRouter {
public:
    CountRouter(const std::vector<std::int64_t>& times, const std::string& state_path) : times_(times) {
        const auto state = load_state(state_path);
        boundaries_.assign(state.begin(), state.begin() + 501);
        bucket_counts_.assign(state.begin() + 501, state.end());
    }

    std::uint64_t sketch(const Query& query) const {
        long double total = 0.0;
        for (std::size_t index = 0; index < 500; ++index) {
            const auto begin = boundaries_[index];
            const auto end = boundaries_[index + 1];
            if (end <= query.left || begin >= query.right) continue;
            const auto overlap = std::min(end, query.right) - std::max(begin, query.left);
            const auto width = std::max<std::int64_t>(1, end - begin);
            total += static_cast<long double>(bucket_counts_[index]) * overlap / width;
        }
        return static_cast<std::uint64_t>(total);
    }

    std::uint64_t exact(const Query& query) const {
        std::uint64_t total = 0;
        const auto* data = times_.data();
        const auto size = times_.size();
        for (std::size_t index = 0; index < size; ++index) {
            const auto time = data[index];
            total += static_cast<std::uint64_t>(time >= query.left && time <= query.right);
        }
        return total;
    }

    std::uint64_t execute(const Query& query) const {
        return query.fallback ? exact(query) : sketch(query);
    }

private:
    static std::vector<std::int64_t> load_state(const std::string& path) {
        std::ifstream input(path, std::ios::binary);
        if (!input) throw std::runtime_error("cannot open router state: " + path);
        std::vector<std::int64_t> state(1001);
        input.read(reinterpret_cast<char*>(state.data()), static_cast<std::streamsize>(state.size() * sizeof(std::int64_t)));
        if (!input) throw std::runtime_error("cannot read router state: " + path);
        return state;
    }
    const std::vector<std::int64_t>& times_;
    std::vector<std::int64_t> boundaries_;
    std::vector<std::int64_t> bucket_counts_;
};

static double percentile(std::vector<double> values, double probability) {
    if (values.empty()) return 0.0;
    const auto position = probability * (values.size() - 1);
    const auto lower = static_cast<std::size_t>(std::floor(position));
    const auto upper = static_cast<std::size_t>(std::ceil(position));
    std::nth_element(values.begin(), values.begin() + lower, values.end());
    const double left = values[lower];
    if (upper == lower) return left;
    std::nth_element(values.begin(), values.begin() + upper, values.end());
    return left + (values[upper] - left) * (position - lower);
}

struct Result {
    double qps{};
    double p50_us{};
    double p95_us{};
    double p99_us{};
    std::uint64_t checksum{};
};

enum class Mode { Router, Sketch, Exact };

static Result run_trial(const CountRouter& router, const std::vector<Query>& queries,
                        int workers, std::size_t total_requests, Mode mode = Mode::Router) {
    std::atomic<std::size_t> next{0};
    std::vector<std::vector<double>> thread_latencies(workers);
    std::vector<std::uint64_t> checksums(workers, 0);
    std::vector<std::thread> threads;
    threads.reserve(workers);
    const auto wall_start = Clock::now();
    for (int worker = 0; worker < workers; ++worker) {
        threads.emplace_back([&, worker] {
            auto& latencies = thread_latencies[worker];
            latencies.reserve(total_requests / workers + 1);
            while (true) {
                const auto request = next.fetch_add(1, std::memory_order_relaxed);
                if (request >= total_requests) break;
                const auto& query = queries[request % queries.size()];
                const auto start = Clock::now();
                if (mode == Mode::Router) checksums[worker] += router.execute(query);
                else if (mode == Mode::Sketch) checksums[worker] += router.sketch(query);
                else checksums[worker] += router.exact(query);
                const auto end = Clock::now();
                latencies.push_back(std::chrono::duration<double, std::micro>(end - start).count());
            }
        });
    }
    for (auto& thread : threads) thread.join();
    const auto wall_end = Clock::now();
    std::vector<double> latencies;
    latencies.reserve(total_requests);
    for (auto& local : thread_latencies) latencies.insert(latencies.end(), local.begin(), local.end());
    const double wall_seconds = std::chrono::duration<double>(wall_end - wall_start).count();
    std::uint64_t checksum = 0;
    for (const auto value : checksums) checksum += value;
    return {
        total_requests / wall_seconds,
        percentile(latencies, 0.50),
        percentile(latencies, 0.95),
        percentile(latencies, 0.99),
        checksum,
    };
}

int main(int argc, char** argv) {
    if (argc != 7) {
        std::cerr << "usage: router_scalability DATASET DATA_PATH STATE QUERY_CSV ROW_LIMIT SCALABILITY_CSV\n";
        return 2;
    }
    const std::string dataset = argv[1];
    const std::string data_path = argv[2];
    const std::string state_path = argv[3];
    const std::string query_path = argv[4];
    const auto row_limit = static_cast<std::size_t>(std::stoull(argv[5]));
    const std::string output_path = argv[6];
    const auto times = load_times(data_path, row_limit);
    const auto queries = load_queries(query_path);
    const CountRouter router(times, state_path);
    const double fallback_rate = std::count_if(queries.begin(), queries.end(), [](const Query& q) { return q.fallback; }) /
                                 static_cast<double>(queries.size());
    const std::size_t repeats = std::clamp<std::size_t>(100000000ULL / std::max<std::size_t>(1, times.size()) /
                                                        std::max(0.01, fallback_rate) / queries.size(), 10, 50);
    const std::size_t total_requests = repeats * queries.size();

    for (const auto& query : queries) {
        static volatile std::uint64_t warmup = 0;
        warmup ^= router.execute(query);
    }

    std::ofstream output(output_path, std::ios::app);
    if (!output) throw std::runtime_error("cannot open output: " + output_path);
    for (const int workers : {1, 2, 4, 8, 16}) {
        for (int trial = 1; trial <= 3; ++trial) {
            const auto result = run_trial(router, queries, workers, total_requests, Mode::Router);
            output << dataset << ',' << times.size() << ',' << queries.size() << ',' << total_requests << ','
                   << fallback_rate << ',' << workers << ',' << trial << ',' << std::setprecision(12)
                   << result.qps << ',' << result.p50_us << ',' << result.p95_us << ',' << result.p99_us << ','
                   << result.checksum << '\n';
            std::cout << dataset << " workers=" << workers << " trial=" << trial
                      << " qps=" << std::fixed << std::setprecision(1) << result.qps
                      << " p95_us=" << result.p95_us << '\n';
        }
    }
    return 0;
}
