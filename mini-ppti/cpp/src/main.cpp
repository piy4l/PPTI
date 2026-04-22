#include "ckks_runner.h"
#include "profiler.h"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <cmath>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct Config {
    std::string op = "all";
    std::string impl = "optimized";
    std::string append_csv;
    std::string input_file;
    std::string calibration_file;
    std::string nexus_input_dir;
    std::string nexus_calibration_dir;
    std::size_t n = 4;
    int steps = 1;
    double scalar = 2.0;
    double add_const = 5.0;
    int trials = 5;
    int warmup = 2;
    bool csv_only = false;
    std::string tokens = "1,2,3,4";
};

struct AggregateStats {
    std::string op_name;
    std::size_t input_size = 0;
    std::string metadata;
    int trials = 0;
    double avg_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
};

static void print_vector(const std::string& label, const std::vector<double>& x) {
    std::cout << label << ": [";
    for (std::size_t i = 0; i < x.size(); ++i) {
        std::cout << std::fixed << std::setprecision(6) << x[i];
        if (i + 1 < x.size()) {
            std::cout << ", ";
        }
    }
    std::cout << "]\n";
}

static std::vector<double> make_input(std::size_t n) {
    std::vector<double> x(n);
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = static_cast<double>(i + 1);
    }
    return x;
}

static std::vector<double> make_input2(std::size_t n) {
    std::vector<double> y(n);
    for (std::size_t i = 0; i < n; ++i) {
        y[i] = static_cast<double>(2 * (i + 1));
    }
    return y;
}

static std::vector<double> make_benchmark_input(std::size_t n) {
    std::vector<double> x(n);
    const double denom = static_cast<double>(std::max<std::size_t>(1, n));
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = static_cast<double>(i + 1) / denom;
    }
    return x;
}

static std::size_t effective_input_size(const Config& cfg) {
    if (cfg.op == "bench_matmul_qkv" ||
        cfg.op == "bench_matmul_attn_v" ||
        cfg.op == "bench_matmul_ffn1" ||
        cfg.op == "bench_matmul_ffn2" ||
        cfg.op == "bench_layernorm_r128" ||
        cfg.op == "bench_argmax_r128") {
        return 128;
    }
    if (cfg.op == "nexus_gelu") {
        return 32768;
    }
    if (cfg.op == "nexus_layernorm") {
        return 16 * 768;
    }
    if (cfg.op == "nexus_softmax") {
        return 128 * 128;
    }
    if (cfg.op == "nexus_argmax") {
        return 8;
    }
    if (cfg.op == "nexus_matmul") {
        return 768;
    }
    return cfg.n;
}

static uint32_t required_mult_depth(const std::string& op_name) {
    if (op_name == "bench_matmul" ||
        op_name == "bench_matmul_qkv" ||
        op_name == "bench_matmul_attn_v" ||
        op_name == "bench_matmul_ffn1" ||
        op_name == "bench_matmul_ffn2") {
        return 8;
    }
    if (op_name == "bench_argmax" || op_name == "bench_argmax_r128") {
        return 32;
    }
    if (op_name == "bench_layernorm" || op_name == "bench_layernorm_r128") {
        return 12;
    }
    if (op_name == "nexus_gelu") {
        return 6;
    }
    if (op_name == "nexus_softmax") {
        return 6;
    }
    if (op_name == "nexus_layernorm") {
        return 12;
    }
    if (op_name == "nexus_argmax") {
        return 32;
    }
    if (op_name == "nexus_matmul") {
        return 8;
    }
    if (op_name == "tiny_text_classifier" || op_name == "compare_tiny_text_classifier" ||
        op_name == "trace_tiny_text_classifier") {
        return 30;
    }
    if (op_name == "tiny_bert_block" || op_name == "compare_tiny_bert_block" ||
        op_name == "trace_tiny_bert_block") {
        return 28;
    }
    if (op_name == "toy_transformer_block" || op_name == "compare_toy_transformer_block" ||
        op_name == "all") {
        return 20;
    }
    return 2;
}

static std::vector<int32_t> required_rotations(const Config& cfg) {
    const std::size_t n = effective_input_size(cfg);
    std::set<int32_t> rotset = {1, -1, 2, -2, cfg.steps, -cfg.steps};

    if (cfg.op == "bench_matmul" ||
        cfg.op == "bench_matmul_qkv" ||
        cfg.op == "bench_matmul_attn_v" ||
        cfg.op == "bench_matmul_ffn1" ||
        cfg.op == "bench_matmul_ffn2" ||
        cfg.op == "nexus_matmul" ||
        cfg.op == "all") {
        rotset.insert(3);
        rotset.insert(-3);
    }

    if (cfg.op == "bench_layernorm" ||
        cfg.op == "bench_argmax" ||
        cfg.op == "bench_layernorm_r128" ||
        cfg.op == "bench_argmax_r128" ||
        cfg.op == "nexus_layernorm" ||
        cfg.op == "nexus_argmax" ||
        cfg.op == "all") {
        for (std::size_t shift = 1; shift < n; ++shift) {
            rotset.insert(static_cast<int32_t>(shift));
            rotset.insert(-static_cast<int32_t>(shift));
        }
    }

    if (cfg.op == "tiny_bert_block" || cfg.op == "compare_tiny_bert_block" ||
        cfg.op == "trace_tiny_bert_block" || cfg.op == "tiny_text_classifier" ||
        cfg.op == "compare_tiny_text_classifier" || cfg.op == "trace_tiny_text_classifier" ||
        cfg.op == "all") {
        rotset.insert(3);
        rotset.insert(-3);
        rotset.insert(4);
        rotset.insert(-4);
        rotset.insert(8);
        rotset.insert(-8);
        rotset.insert(12);
        rotset.insert(-12);
    }

    return std::vector<int32_t>(rotset.begin(), rotset.end());
}

struct ErrorStats {
    double max_abs_error = 0.0;
    double mean_abs_error = 0.0;
    double rmse = 0.0;
};

static ErrorStats compute_error_stats(const std::vector<double>& expected,
                                      const std::vector<double>& actual) {
    if (expected.size() != actual.size()) {
        throw std::runtime_error("compute_error_stats() requires equal-length inputs");
    }

    ErrorStats stats;
    if (expected.empty()) {
        return stats;
    }

    double abs_sum = 0.0;
    double sq_sum = 0.0;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const double err = std::abs(expected[i] - actual[i]);
        stats.max_abs_error = std::max(stats.max_abs_error, err);
        abs_sum += err;
        sq_sum += err * err;
    }

    stats.mean_abs_error = abs_sum / static_cast<double>(expected.size());
    stats.rmse = std::sqrt(sq_sum / static_cast<double>(expected.size()));
    return stats;
}

static std::string csv_escape(const std::string& value) {
    std::string out = "\"";
    for (char ch : value) {
        if (ch == '"') {
            out += "\"\"";
        } else {
            out += ch;
        }
    }
    out += "\"";
    return out;
}

static void append_results_csv(const std::string& path,
                               const std::vector<AggregateStats>& all_stats,
                               const Config& cfg) {
    if (path.empty() || all_stats.empty()) {
        return;
    }

    const bool exists = std::ifstream(path).good();
    std::ofstream out(path, std::ios::app);
    if (!out) {
        throw std::runtime_error("Could not open append CSV path: " + path);
    }

    if (!exists) {
        out << "system,op,shape,impl,avg_ms,notes\n";
    }

    for (const auto& stat : all_stats) {
        std::string shape;
        std::string notes = stat.metadata;

        std::stringstream ss(stat.metadata);
        std::string item;
        while (std::getline(ss, item, ';')) {
            const auto pos = item.find('=');
            if (pos == std::string::npos) {
                continue;
            }
            const std::string key = item.substr(0, pos);
            const std::string value = item.substr(pos + 1);
            if (key == "shape") {
                shape = value;
            }
        }

        out << "mini_ppti_baseline"
            << "," << stat.op_name
            << "," << shape
            << "," << cfg.impl
            << "," << std::fixed << std::setprecision(6) << stat.avg_ms
            << "," << csv_escape(notes)
            << "\n";
    }
}

static std::vector<int> parse_tokens(const std::string& spec) {
    std::vector<int> tokens;
    std::stringstream ss(spec);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) {
            continue;
        }
        tokens.push_back(std::stoi(item));
    }
    return tokens;
}

static std::vector<double> read_flat_doubles(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Could not open input file: " + path);
    }

    std::vector<double> values;
    double value = 0.0;
    while (in >> value) {
        values.push_back(value);
    }
    return values;
}

static std::vector<std::vector<double>> read_matrix(const std::string& path,
                                                    std::size_t rows,
                                                    std::size_t cols) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Could not open matrix file: " + path);
    }

    std::vector<std::vector<double>> matrix(rows, std::vector<double>(cols, 0.0));
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            if (!(in >> matrix[r][c])) {
                throw std::runtime_error("Unexpected EOF while reading matrix file: " + path);
            }
        }
    }
    return matrix;
}

static std::string join_path(const std::string& dir, const std::string& file) {
    if (dir.empty()) {
        return file;
    }
    if (dir.back() == '/') {
        return dir + file;
    }
    return dir + "/" + file;
}

static std::size_t next_power_of_two(std::size_t n) {
    std::size_t out = 1;
    while (out < n) {
        out <<= 1;
    }
    return out;
}

static double mean_absolute_error_prefix(const std::vector<double>& expected,
                                         const std::vector<double>& actual,
                                         std::size_t count) {
    if (count == 0) {
        return 0.0;
    }
    if (expected.size() < count || actual.size() < count) {
        throw std::runtime_error("mean_absolute_error_prefix() got insufficient data");
    }

    double sum = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        sum += std::abs(expected[i] - actual[i]);
    }
    return sum / static_cast<double>(count);
}

static std::vector<double> apply_block_diagonal_plain_custom(
    const std::vector<double>& x,
    const std::vector<std::vector<double>>& weights) {
    const std::size_t hidden = weights.size();
    if (hidden == 0 || x.size() % hidden != 0) {
        throw std::runtime_error("apply_block_diagonal_plain_custom() got invalid shape");
    }
    for (const auto& row : weights) {
        if (row.size() != hidden) {
            throw std::runtime_error("apply_block_diagonal_plain_custom() requires square weights");
        }
    }

    const std::size_t seq_len = x.size() / hidden;
    std::vector<double> out(x.size(), 0.0);
    for (std::size_t token = 0; token < seq_len; ++token) {
        const std::size_t base = token * hidden;
        for (std::size_t i = 0; i < hidden; ++i) {
            double acc = 0.0;
            for (std::size_t j = 0; j < hidden; ++j) {
                acc += weights[i][j] * x[base + j];
            }
            out[base + i] = acc;
        }
    }
    return out;
}

static void print_nexus_style_result(const std::string& label,
                                     const std::string& shape,
                                     const AggregateStats& stats,
                                     double mae,
                                     const std::string& notes = "") {
    std::cout << "[" << label << "] " << shape << " takes: "
              << std::fixed << std::setprecision(6) << stats.avg_ms
              << " milliseconds\n";
    std::cout << "Mean Absolute Error: " << std::fixed << std::setprecision(6) << mae
              << "\n";
    if (!notes.empty()) {
        std::cout << "Notes: " << notes << "\n";
    }
}

static int argmax_index(const std::vector<double>& values) {
    if (values.empty()) {
        throw std::runtime_error("argmax_index() requires a non-empty vector");
    }

    int best = 0;
    for (std::size_t i = 1; i < values.size(); ++i) {
        if (values[i] > values[best]) {
            best = static_cast<int>(i);
        }
    }
    return best;
}

static std::string layernorm_shape_metadata(std::size_t n) {
    if (n == 128) {
        return "R128;surrogate=1x128_vs_24xR128x768";
    }
    return "1x" + std::to_string(n);
}

static std::string argmax_shape_metadata(std::size_t n) {
    if (n == 128) {
        return "R128";
    }
    return "1x" + std::to_string(n);
}

static std::string matmul_shape_metadata(std::size_t n) {
    if (n % 4 == 0) {
        return "4x4_blockdiag_x" + std::to_string(n / 4);
    }
    return "vector_len_" + std::to_string(n);
}

static void print_usage() {
    std::cout
        << "Usage: ./mini_ppti [options]\n"
        << "Options:\n"
        << "  --op <all|encrypt_only|decrypt_only|encrypt_decrypt|add_plain|add_ct_ct|mul_plain|mul_ct_ct|rotate|toy_transformer_block|compare_toy_transformer_block|tiny_bert_block|compare_tiny_bert_block|trace_tiny_bert_block|tiny_text_classifier|compare_tiny_text_classifier|trace_tiny_text_classifier|bench_matmul|bench_layernorm|bench_argmax|bench_matmul_qkv|bench_matmul_attn_v|bench_matmul_ffn1|bench_matmul_ffn2|bench_layernorm_r128|bench_argmax_r128|nexus_gelu|nexus_layernorm|nexus_softmax|nexus_argmax|nexus_matmul|nexus_suite>\n"
        << "  --impl <optimized|baseline>\n"
        << "  --append-csv <path>\n"
        << "  --input-file <path>\n"
        << "  --calibration-file <path>\n"
        << "  --nexus-input-dir <path>\n"
        << "  --nexus-calibration-dir <path>\n"
        << "  --n <input_size>\n"
        << "  --steps <rotation_steps>\n"
        << "  --scalar <value>\n"
        << "  --add-const <value>\n"
        << "  --tokens <comma-separated token ids>\n"
        << "  --trials <count>\n"
        << "  --warmup <count>\n"
        << "  --csv-only\n"
        << "  --help\n";
}

static Config parse_args(int argc, char* argv[]) {
    Config cfg;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        auto require_value = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for " + name);
            }
            return argv[++i];
        };

        if (arg == "--op") {
            cfg.op = require_value("--op");
        } else if (arg == "--impl") {
            cfg.impl = require_value("--impl");
        } else if (arg == "--append-csv") {
            cfg.append_csv = require_value("--append-csv");
        } else if (arg == "--input-file") {
            cfg.input_file = require_value("--input-file");
        } else if (arg == "--calibration-file") {
            cfg.calibration_file = require_value("--calibration-file");
        } else if (arg == "--nexus-input-dir") {
            cfg.nexus_input_dir = require_value("--nexus-input-dir");
        } else if (arg == "--nexus-calibration-dir") {
            cfg.nexus_calibration_dir = require_value("--nexus-calibration-dir");
        } else if (arg == "--n") {
            cfg.n = static_cast<std::size_t>(std::stoull(require_value("--n")));
        } else if (arg == "--steps") {
            cfg.steps = std::stoi(require_value("--steps"));
        } else if (arg == "--scalar") {
            cfg.scalar = std::stod(require_value("--scalar"));
        } else if (arg == "--add-const") {
            cfg.add_const = std::stod(require_value("--add-const"));
        } else if (arg == "--tokens") {
            cfg.tokens = require_value("--tokens");
        } else if (arg == "--trials") {
            cfg.trials = std::stoi(require_value("--trials"));
        } else if (arg == "--warmup") {
            cfg.warmup = std::stoi(require_value("--warmup"));
        } else if (arg == "--csv-only") {
            cfg.csv_only = true;
        } else if (arg == "--help") {
            print_usage();
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (cfg.n == 0) {
        throw std::runtime_error("--n must be >= 1");
    }
    if (cfg.trials <= 0) {
        throw std::runtime_error("--trials must be >= 1");
    }
    if (cfg.warmup < 0) {
        throw std::runtime_error("--warmup must be >= 0");
    }
    if (cfg.impl != "optimized" && cfg.impl != "baseline") {
        throw std::runtime_error("--impl must be one of: optimized, baseline");
    }

    return cfg;
}

static AggregateStats aggregate_records(const std::vector<ProfileRecord>& records,
                                        const std::string& op_name,
                                        std::size_t input_size,
                                        const std::string& metadata) {
    if (records.empty()) {
        throw std::runtime_error("No profiling records to aggregate");
    }

    double sum = 0.0;
    double mn = std::numeric_limits<double>::max();
    double mx = std::numeric_limits<double>::lowest();

    for (const auto& r : records) {
        sum += r.time_ms;
        mn = std::min(mn, r.time_ms);
        mx = std::max(mx, r.time_ms);
    }

    AggregateStats stats;
    stats.op_name = op_name;
    stats.input_size = input_size;
    stats.metadata = metadata;
    stats.trials = static_cast<int>(records.size());
    stats.avg_ms = sum / static_cast<double>(records.size());
    stats.min_ms = mn;
    stats.max_ms = mx;
    return stats;
}

static void print_aggregate_table(const std::vector<AggregateStats>& all_stats) {
    std::cout << "\n=== Aggregate Summary ===\n";
    std::cout << std::left
              << std::setw(18) << "Operation"
              << std::setw(12) << "Input"
              << std::setw(10) << "Trials"
              << std::setw(14) << "Avg (ms)"
              << std::setw(14) << "Min (ms)"
              << std::setw(14) << "Max (ms)"
              << "Metadata\n";

    for (const auto& s : all_stats) {
        std::cout << std::left
                  << std::setw(18) << s.op_name
                  << std::setw(12) << s.input_size
                  << std::setw(10) << s.trials
                  << std::setw(14) << std::fixed << std::setprecision(6) << s.avg_ms
                  << std::setw(14) << s.min_ms
                  << std::setw(14) << s.max_ms
                  << s.metadata << "\n";
    }
}

static void print_aggregate_csv(const std::vector<AggregateStats>& all_stats) {
    std::cout << "op_name,input_size,trials,avg_ms,min_ms,max_ms,metadata\n";
    for (const auto& s : all_stats) {
        std::cout << s.op_name << ","
                  << s.input_size << ","
                  << s.trials << ","
                  << std::fixed << std::setprecision(6)
                  << s.avg_ms << ","
                  << s.min_ms << ","
                  << s.max_ms << ","
                  << "\"" << s.metadata << "\"\n";
    }
}

template <typename Func>
static AggregateStats benchmark_op(Profiler& profiler,
                                   const std::string& op_name,
                                   std::size_t input_size,
                                   const std::string& metadata,
                                   int warmup,
                                   int trials,
                                   Func fn) {
    for (int i = 0; i < warmup; ++i) {
        fn();
    }

    std::vector<ProfileRecord> records;
    records.reserve(static_cast<std::size_t>(trials));

    for (int t = 0; t < trials; ++t) {
        {
            ScopedTimer timer(profiler, op_name, input_size, metadata);
            fn();
        }
        records.push_back(profiler.get_records().back());
    }

    return aggregate_records(records, op_name, input_size, metadata);
}

int main(int argc, char* argv[]) {
    try {
        Config cfg = parse_args(argc, argv);

        CKKSRuntime runtime;
        Profiler profiler;

        std::vector<int32_t> rotations = required_rotations(cfg);

        const std::size_t init_n = effective_input_size(cfg);
        runtime.init(
            required_mult_depth(cfg.op),
            50,
            static_cast<uint32_t>(std::max<std::size_t>(8, next_power_of_two(init_n))));
        runtime.keygen(rotations);

        std::vector<double> x = make_input(cfg.n);
        std::vector<double> y = make_input2(cfg.n);
        std::vector<double> bench_x = make_benchmark_input(cfg.n);
        std::vector<int> token_ids = parse_tokens(cfg.tokens);
        auto pre_ct = runtime.encrypt_only(x);

        std::vector<AggregateStats> all_stats;

        if (!cfg.csv_only) {
            std::cout << runtime.info() << "\n";
            print_vector("input_x", x);
            print_vector("input_y", y);
            print_vector("bench_x", bench_x);
            std::cout << "tokens: " << cfg.tokens << "\n";
        }

        auto run_encrypt_only = [&]() {
            lbcrypto::Ciphertext<lbcrypto::DCRTPoly> out;
            auto stats = benchmark_op(
                profiler, "encrypt_only", x.size(), "", cfg.warmup, cfg.trials,
                [&]() { out = runtime.encrypt_only(x); });
            all_stats.push_back(stats);
        };

        auto run_decrypt_only = [&]() {
            std::vector<double> out;
            auto stats = benchmark_op(
                profiler, "decrypt_only", x.size(), "", cfg.warmup, cfg.trials,
                [&]() { out = runtime.decrypt_only(pre_ct, x.size()); });
            all_stats.push_back(stats);
            if (!cfg.csv_only) {
                print_vector("decrypt_only", out);
            }
        };

        auto run_encrypt_decrypt = [&]() {
            std::vector<double> out;
            auto stats = benchmark_op(
                profiler, "encrypt_decrypt", x.size(), "", cfg.warmup, cfg.trials,
                [&]() { out = runtime.encrypt_decrypt(x); });
            all_stats.push_back(stats);
            if (!cfg.csv_only) {
                print_vector("encrypt_decrypt", out);
            }
        };

        auto run_add_plain = [&]() {
            std::vector<double> out;
            std::string metadata = "c=" + std::to_string(cfg.add_const);
            auto stats = benchmark_op(
                profiler, "add_plain", x.size(), metadata, cfg.warmup, cfg.trials,
                [&]() { out = runtime.add_plain(x, cfg.add_const); });
            all_stats.push_back(stats);
            if (!cfg.csv_only) {
                print_vector("add_plain", out);
            }
        };

        auto run_add_ct_ct = [&]() {
            std::vector<double> out;
            auto stats = benchmark_op(
                profiler, "add_ct_ct", x.size(), "x+y", cfg.warmup, cfg.trials,
                [&]() { out = runtime.add_ct_ct(x, y); });
            all_stats.push_back(stats);
            if (!cfg.csv_only) {
                print_vector("add_ct_ct", out);
            }
        };

        auto run_mul_plain = [&]() {
            std::vector<double> out;
            std::string metadata = "c=" + std::to_string(cfg.scalar);
            auto stats = benchmark_op(
                profiler, "mul_plain", x.size(), metadata, cfg.warmup, cfg.trials,
                [&]() { out = runtime.mul_plain(x, cfg.scalar); });
            all_stats.push_back(stats);
            if (!cfg.csv_only) {
                print_vector("mul_plain", out);
            }
        };

        auto run_mul_ct_ct = [&]() {
            std::vector<double> out;
            auto stats = benchmark_op(
                profiler, "mul_ct_ct", x.size(), "x*y", cfg.warmup, cfg.trials,
                [&]() { out = runtime.mul_ct_ct(x, y); });
            all_stats.push_back(stats);
            if (!cfg.csv_only) {
                print_vector("mul_ct_ct", out);
            }
        };

        auto run_rotate = [&]() {
            std::vector<double> out;
            std::string metadata = "steps=" + std::to_string(cfg.steps);
            auto stats = benchmark_op(
                profiler, "rotate", x.size(), metadata, cfg.warmup, cfg.trials,
                [&]() { out = runtime.rotate(x, cfg.steps); });
            all_stats.push_back(stats);
            if (!cfg.csv_only) {
                print_vector("rotate", out);
            }
        };

        auto run_toy_transformer_block = [&]() {
            std::vector<double> out;
            auto stats = benchmark_op(
                profiler, "toy_transformer_block", x.size(), "he_only_poly_approx",
                cfg.warmup, cfg.trials,
                [&]() { out = runtime.toy_transformer_block(x); });
            all_stats.push_back(stats);
            if (!cfg.csv_only) {
                print_vector("toy_transformer_block", out);
            }
        };

        auto run_compare_toy_transformer_block = [&]() {
            std::vector<double> plain_out = runtime.toy_transformer_block_plain(x);
            std::vector<double> he_out;
            auto stats = benchmark_op(
                profiler, "compare_toy_transformer_block", x.size(),
                "plaintext_vs_he_poly_approx", cfg.warmup, cfg.trials,
                [&]() { he_out = runtime.toy_transformer_block(x); });
            all_stats.push_back(stats);

            ErrorStats error_stats = compute_error_stats(plain_out, he_out);

            if (!cfg.csv_only) {
                print_vector("toy_transformer_plain", plain_out);
                print_vector("toy_transformer_he", he_out);
                std::cout << std::fixed << std::setprecision(6)
                          << "max_abs_error: " << error_stats.max_abs_error << "\n"
                          << "mean_abs_error: " << error_stats.mean_abs_error << "\n"
                          << "rmse: " << error_stats.rmse << "\n";
            }
        };

        auto run_tiny_bert_block = [&]() {
            std::vector<double> out;
            auto stats = benchmark_op(
                profiler, "tiny_bert_block", x.size(), "tiny_seq4_hidden4",
                cfg.warmup, cfg.trials,
                [&]() { out = runtime.tiny_bert_block(x); });
            all_stats.push_back(stats);
            if (!cfg.csv_only) {
                print_vector("tiny_bert_block", out);
            }
        };

        auto run_compare_tiny_bert_block = [&]() {
            std::vector<double> plain_out = runtime.tiny_bert_block_plain(x);
            std::vector<double> he_out;
            auto stats = benchmark_op(
                profiler, "compare_tiny_bert_block", x.size(),
                "plaintext_vs_he_tiny_seq4_hidden4", cfg.warmup, cfg.trials,
                [&]() { he_out = runtime.tiny_bert_block(x); });
            all_stats.push_back(stats);

            ErrorStats error_stats = compute_error_stats(plain_out, he_out);

            if (!cfg.csv_only) {
                print_vector("tiny_bert_plain", plain_out);
                print_vector("tiny_bert_he", he_out);
                std::cout << std::fixed << std::setprecision(6)
                          << "max_abs_error: " << error_stats.max_abs_error << "\n"
                          << "mean_abs_error: " << error_stats.mean_abs_error << "\n"
                          << "rmse: " << error_stats.rmse << "\n";
            }
        };

        auto run_trace_tiny_bert_block = [&]() {
            runtime.trace_tiny_bert_block(x);
        };

        auto run_tiny_text_classifier = [&]() {
            std::vector<double> logits;
            auto stats = benchmark_op(
                profiler, "tiny_text_classifier", token_ids.size(), cfg.tokens,
                cfg.warmup, cfg.trials,
                [&]() { logits = runtime.tiny_text_classifier(token_ids); });
            all_stats.push_back(stats);
            if (!cfg.csv_only) {
                print_vector("tiny_text_logits", logits);
                std::cout << "tiny_text_pred: " << argmax_index(logits) << "\n";
            }
        };

        auto run_compare_tiny_text_classifier = [&]() {
            std::vector<double> plain_logits = runtime.tiny_text_classifier_plain(token_ids);
            std::vector<double> he_logits;
            auto stats = benchmark_op(
                profiler, "compare_tiny_text_classifier", token_ids.size(), cfg.tokens,
                cfg.warmup, cfg.trials,
                [&]() { he_logits = runtime.tiny_text_classifier(token_ids); });
            all_stats.push_back(stats);

            ErrorStats error_stats = compute_error_stats(plain_logits, he_logits);

            if (!cfg.csv_only) {
                print_vector("tiny_text_plain_logits", plain_logits);
                print_vector("tiny_text_he_logits", he_logits);
                std::cout << "tiny_text_plain_pred: " << argmax_index(plain_logits) << "\n";
                std::cout << "tiny_text_he_pred: " << argmax_index(he_logits) << "\n";
                std::cout << std::fixed << std::setprecision(6)
                          << "max_abs_error: " << error_stats.max_abs_error << "\n"
                          << "mean_abs_error: " << error_stats.mean_abs_error << "\n"
                          << "rmse: " << error_stats.rmse << "\n";
            }
        };

        auto run_trace_tiny_text_classifier = [&]() {
            runtime.trace_tiny_text_classifier(token_ids);
        };

        auto run_bench_layernorm = [&]() {
            std::vector<double> out;
            std::string metadata = "impl=" + cfg.impl + ";shape=" + layernorm_shape_metadata(bench_x.size());
            const bool optimized = (cfg.impl == "optimized");
            auto stats = benchmark_op(
                profiler, "bench_layernorm", bench_x.size(), metadata,
                cfg.warmup, cfg.trials,
                [&]() { out = runtime.bench_layernorm(bench_x, optimized); });
            all_stats.push_back(stats);
            if (!cfg.csv_only) {
                print_vector("bench_layernorm_out", out);
            }
        };

        auto run_bench_layernorm_r128 = [&]() {
            std::vector<double> out;
            std::vector<double> local_x = make_benchmark_input(128);
            std::string metadata =
                "impl=" + cfg.impl +
                ";shape=" + layernorm_shape_metadata(local_x.size());
            const bool optimized = (cfg.impl == "optimized");
            auto stats = benchmark_op(
                profiler, "bench_layernorm_r128", local_x.size(), metadata,
                cfg.warmup, cfg.trials,
                [&]() { out = runtime.bench_layernorm(local_x, optimized); });
            all_stats.push_back(stats);
            if (!cfg.csv_only) {
                print_vector("bench_layernorm_r128_out", out);
            }
        };

        auto run_bench_argmax = [&]() {
            std::vector<double> out;
            std::string metadata = "impl=" + cfg.impl + ";shape=" + argmax_shape_metadata(bench_x.size());
            const bool optimized = (cfg.impl == "optimized");
            auto stats = benchmark_op(
                profiler, "bench_argmax", bench_x.size(), metadata,
                cfg.warmup, cfg.trials,
                [&]() { out = runtime.bench_argmax(bench_x, optimized); });
            all_stats.push_back(stats);
            if (!cfg.csv_only) {
                print_vector("bench_argmax_out", out);
            }
        };

        auto run_bench_argmax_r128 = [&]() {
            std::vector<double> out;
            std::vector<double> local_x = make_benchmark_input(128);
            std::string metadata = "impl=" + cfg.impl + ";shape=" + argmax_shape_metadata(local_x.size());
            const bool optimized = (cfg.impl == "optimized");
            auto stats = benchmark_op(
                profiler, "bench_argmax", local_x.size(), metadata,
                cfg.warmup, cfg.trials,
                [&]() { out = runtime.bench_argmax(local_x, optimized); });
            all_stats.push_back(stats);
            if (!cfg.csv_only) {
                print_vector("bench_argmax_r128_out", out);
            }
        };

        auto run_bench_matmul = [&]() {
            std::vector<double> out;
            std::string metadata = "impl=" + cfg.impl + ";shape=" + matmul_shape_metadata(bench_x.size());
            const bool optimized = (cfg.impl == "optimized");
            auto stats = benchmark_op(
                profiler, "bench_matmul", bench_x.size(), metadata,
                cfg.warmup, cfg.trials,
                [&]() { out = runtime.bench_matmul(bench_x, optimized); });
            all_stats.push_back(stats);
            if (!cfg.csv_only) {
                print_vector("bench_matmul_out", out);
            }
        };

        auto run_bench_matmul_named = [&](const std::string& op_name,
                                          const std::string& shape,
                                          const std::string& surrogate_shape,
                                          std::size_t n) {
            std::vector<double> out;
            std::vector<double> local_x = make_benchmark_input(n);
            std::string metadata =
                "impl=" + cfg.impl +
                ";shape=" + shape +
                ";surrogate=" + surrogate_shape;
            const bool optimized = (cfg.impl == "optimized");
            auto stats = benchmark_op(
                profiler, op_name, local_x.size(), metadata,
                cfg.warmup, cfg.trials,
                [&]() { out = runtime.bench_matmul(local_x, optimized); });
            all_stats.push_back(stats);
            if (!cfg.csv_only) {
                print_vector(op_name + "_out", out);
            }
        };

        auto resolve_data_path = [&](const std::string& explicit_path,
                                     const std::string& dir,
                                     const std::string& file_name) {
            if (!explicit_path.empty()) {
                return explicit_path;
            }
            if (!dir.empty()) {
                return join_path(dir, file_name);
            }
            throw std::runtime_error("Missing path for " + file_name +
                                     ". Pass --input-file/--calibration-file or --nexus-input-dir/--nexus-calibration-dir.");
        };

        auto run_nexus_gelu = [&]() {
            const std::string input_path = resolve_data_path(
                cfg.input_file, cfg.nexus_input_dir, "gelu_input_32768.txt");
            const std::string calibration_path = resolve_data_path(
                cfg.calibration_file, cfg.nexus_calibration_dir, "gelu_calibration_32768.txt");
            const std::vector<double> input = read_flat_doubles(input_path);
            const std::vector<double> calibration = read_flat_doubles(calibration_path);

            std::vector<double> out;
            auto stats = benchmark_op(
                profiler, "nexus_gelu", input.size(),
                "impl=" + cfg.impl + ";shape=32768;input=" + input_path,
                cfg.warmup, cfg.trials,
                [&]() { out = runtime.bench_gelu(input); });
            all_stats.push_back(stats);

            if (!cfg.csv_only) {
                const double mae = mean_absolute_error_prefix(
                    calibration, out, std::min(calibration.size(), out.size()));
                print_nexus_style_result("GELU", std::to_string(input.size()), stats, mae,
                                         "baseline polynomial surrogate");
            }
        };

        auto run_nexus_layernorm = [&]() {
            const std::string input_path = resolve_data_path(
                cfg.input_file, cfg.nexus_input_dir, "layernorm_input_16_768.txt");
            const std::string calibration_path = resolve_data_path(
                cfg.calibration_file, cfg.nexus_calibration_dir, "layernorm_calibration_16_768.txt");
            const std::vector<double> input = read_flat_doubles(input_path);
            const std::vector<double> calibration = read_flat_doubles(calibration_path);
            const bool optimized = (cfg.impl == "optimized");

            std::vector<double> out;
            auto stats = benchmark_op(
                profiler, "nexus_layernorm", input.size(),
                "impl=" + cfg.impl + ";shape=16x768;input=" + input_path,
                cfg.warmup, cfg.trials,
                [&]() { out = runtime.bench_layernorm(input, optimized); });
            all_stats.push_back(stats);

            if (!cfg.csv_only) {
                const std::size_t mae_count = std::min<std::size_t>(
                    768, std::min(calibration.size(), out.size()));
                const double mae = mean_absolute_error_prefix(calibration, out, mae_count);
                print_nexus_style_result("LayerNorm", "16 x 768", stats, mae,
                                         "baseline HE proxy; MAE over first 768 outputs");
            }
        };

        auto run_nexus_softmax = [&]() {
            const std::string input_path = resolve_data_path(
                cfg.input_file, cfg.nexus_input_dir, "softmax_input_128_128.txt");
            const std::string calibration_path = resolve_data_path(
                cfg.calibration_file, cfg.nexus_calibration_dir, "softmax_calibration_128_128.txt");
            const std::vector<double> input = read_flat_doubles(input_path);
            const std::vector<double> calibration = read_flat_doubles(calibration_path);

            std::vector<double> out;
            auto stats = benchmark_op(
                profiler, "nexus_softmax", input.size(),
                "impl=" + cfg.impl + ";shape=128x128;input=" + input_path,
                cfg.warmup, cfg.trials,
                [&]() { out = runtime.bench_softmax(input); });
            all_stats.push_back(stats);

            if (!cfg.csv_only) {
                const std::size_t mae_count = std::min<std::size_t>(
                    128, std::min(calibration.size(), out.size()));
                const double mae = mean_absolute_error_prefix(calibration, out, mae_count);
                print_nexus_style_result("Softmax", "128 x 128", stats, mae,
                                         "baseline polynomial surrogate; MAE over first 128 outputs");
            }
        };

        auto run_nexus_argmax = [&]() {
            const std::string input_path = resolve_data_path(
                cfg.input_file, cfg.nexus_input_dir, "argmax_input_8.txt");
            const std::string calibration_path = resolve_data_path(
                cfg.calibration_file, cfg.nexus_calibration_dir, "argmax_calibration_8.txt");
            const std::vector<double> input = read_flat_doubles(input_path);
            const std::vector<double> calibration = read_flat_doubles(calibration_path);
            const bool optimized = (cfg.impl == "optimized");

            std::vector<double> out;
            auto stats = benchmark_op(
                profiler, "nexus_argmax", input.size(),
                "impl=" + cfg.impl + ";shape=8;input=" + input_path,
                cfg.warmup, cfg.trials,
                [&]() { out = runtime.bench_argmax(input, optimized); });
            all_stats.push_back(stats);

            if (!cfg.csv_only) {
                const double mae = mean_absolute_error_prefix(
                    calibration, out, std::min(calibration.size(), out.size()));
                print_nexus_style_result("Argmax", std::to_string(input.size()), stats, mae,
                                         "baseline pairwise-max surrogate; no sparse slot replication");
            }
        };

        auto run_nexus_matmul = [&]() {
            const std::string lhs_path = resolve_data_path(
                cfg.input_file, cfg.nexus_input_dir, "matrixmul_input_m_128_n_768_k_64_batch_128.txt");
            const std::string rhs_path = join_path(cfg.nexus_input_dir, "matrix_input_n_768_k_64.txt");
            const std::string calibration_path = resolve_data_path(
                cfg.calibration_file, cfg.nexus_calibration_dir, "matrix_output_m_128_k_64_batch_128.txt");

            const auto lhs = read_matrix(lhs_path, 4096, 768);
            const auto rhs = read_matrix(rhs_path, 768, 64);
            const auto calibration = read_matrix(calibration_path, 4096, 64);

            std::vector<std::vector<double>> weights(4, std::vector<double>(4, 0.0));
            for (std::size_t i = 0; i < 4; ++i) {
                for (std::size_t j = 0; j < 4; ++j) {
                    weights[i][j] = rhs[j][i];
                }
            }

            const std::vector<double>& surrogate_input = lhs.front();
            const std::vector<double> surrogate_plain =
                apply_block_diagonal_plain_custom(surrogate_input, weights);

            std::vector<double> out;
            const bool optimized = (cfg.impl == "optimized");
            auto stats = benchmark_op(
                profiler, "nexus_matmul", surrogate_input.size(),
                "impl=" + cfg.impl +
                    ";shape=4096x768_x_768x64" +
                    ";input=" + lhs_path +
                    ";weights=" + rhs_path +
                    ";surrogate=row0_with_rhs_4x4_slice",
                cfg.warmup, cfg.trials,
                [&]() { out = runtime.bench_matmul_custom(surrogate_input, weights, optimized); });
            all_stats.push_back(stats);

            if (!cfg.csv_only) {
                const double avg_error = mean_absolute_error_prefix(
                    surrogate_plain, out, std::min(surrogate_plain.size(), out.size()));
                std::ostringstream notes;
                notes << "baseline 4x4-tiled surrogate from official matrices; "
                      << "official calibration row0[0]=" << std::fixed << std::setprecision(6)
                      << calibration[0][0];
                std::cout << "[MatMul] 4096x768 x 768x64 takes: "
                          << std::fixed << std::setprecision(6) << stats.avg_ms
                          << " milliseconds\n";
                std::cout << "Average error: " << std::fixed << std::setprecision(6)
                          << avg_error << "\n";
                std::cout << "Notes: " << notes.str() << "\n";
            }
        };

        if (cfg.op == "all") {
            run_encrypt_only();
            run_decrypt_only();
            run_encrypt_decrypt();
            run_add_plain();
            run_add_ct_ct();
            run_mul_plain();
            run_mul_ct_ct();
            run_rotate();
            run_toy_transformer_block();
            run_compare_toy_transformer_block();
            run_tiny_bert_block();
            run_compare_tiny_bert_block();
            run_trace_tiny_bert_block();
            run_tiny_text_classifier();
            run_compare_tiny_text_classifier();
            run_trace_tiny_text_classifier();
            run_bench_matmul();
            run_bench_layernorm();
            run_bench_argmax();
            run_bench_matmul_named("bench_matmul_qkv", "R128x768_x_R768x768", "4x4_blockdiag_x32", 128);
            run_bench_matmul_named("bench_matmul_attn_v", "R128x128_x_R128x768", "4x4_blockdiag_x32", 128);
            run_bench_matmul_named("bench_matmul_ffn1", "R128x768_x_R768x3072", "4x4_blockdiag_x32", 128);
            run_bench_matmul_named("bench_matmul_ffn2", "R128x3072_x_R3072x768", "4x4_blockdiag_x32", 128);
            run_bench_layernorm_r128();
            run_bench_argmax_r128();
            run_nexus_gelu();
            run_nexus_layernorm();
            run_nexus_softmax();
            run_nexus_argmax();
            run_nexus_matmul();
        } else if (cfg.op == "encrypt_only") {
            run_encrypt_only();
        } else if (cfg.op == "decrypt_only") {
            run_decrypt_only();
        } else if (cfg.op == "encrypt_decrypt") {
            run_encrypt_decrypt();
        } else if (cfg.op == "add_plain") {
            run_add_plain();
        } else if (cfg.op == "add_ct_ct") {
            run_add_ct_ct();
        } else if (cfg.op == "mul_plain") {
            run_mul_plain();
        } else if (cfg.op == "mul_ct_ct") {
            run_mul_ct_ct();
        } else if (cfg.op == "rotate") {
            run_rotate();
        } else if (cfg.op == "toy_transformer_block") {
            run_toy_transformer_block();
        } else if (cfg.op == "compare_toy_transformer_block") {
            run_compare_toy_transformer_block();
        } else if (cfg.op == "tiny_bert_block") {
            run_tiny_bert_block();
        } else if (cfg.op == "compare_tiny_bert_block") {
            run_compare_tiny_bert_block();
        } else if (cfg.op == "trace_tiny_bert_block") {
            run_trace_tiny_bert_block();
        } else if (cfg.op == "tiny_text_classifier") {
            run_tiny_text_classifier();
        } else if (cfg.op == "compare_tiny_text_classifier") {
            run_compare_tiny_text_classifier();
        } else if (cfg.op == "trace_tiny_text_classifier") {
            run_trace_tiny_text_classifier();
        } else if (cfg.op == "bench_matmul") {
            run_bench_matmul();
        } else if (cfg.op == "bench_layernorm") {
            run_bench_layernorm();
        } else if (cfg.op == "bench_argmax") {
            run_bench_argmax();
        } else if (cfg.op == "bench_matmul_qkv") {
            run_bench_matmul_named("bench_matmul_qkv", "R128x768_x_R768x768", "4x4_blockdiag_x32", 128);
        } else if (cfg.op == "bench_matmul_attn_v") {
            run_bench_matmul_named("bench_matmul_attn_v", "R128x128_x_R128x768", "4x4_blockdiag_x32", 128);
        } else if (cfg.op == "bench_matmul_ffn1") {
            run_bench_matmul_named("bench_matmul_ffn1", "R128x768_x_R768x3072", "4x4_blockdiag_x32", 128);
        } else if (cfg.op == "bench_matmul_ffn2") {
            run_bench_matmul_named("bench_matmul_ffn2", "R128x3072_x_R3072x768", "4x4_blockdiag_x32", 128);
        } else if (cfg.op == "bench_layernorm_r128") {
            run_bench_layernorm_r128();
        } else if (cfg.op == "bench_argmax_r128") {
            run_bench_argmax_r128();
        } else if (cfg.op == "nexus_gelu") {
            run_nexus_gelu();
        } else if (cfg.op == "nexus_layernorm") {
            run_nexus_layernorm();
        } else if (cfg.op == "nexus_softmax") {
            run_nexus_softmax();
        } else if (cfg.op == "nexus_argmax") {
            run_nexus_argmax();
        } else if (cfg.op == "nexus_matmul") {
            run_nexus_matmul();
        } else if (cfg.op == "nexus_suite") {
            run_nexus_gelu();
            run_nexus_layernorm();
            run_nexus_softmax();
            run_nexus_argmax();
            run_nexus_matmul();
        } else {
            throw std::runtime_error("Unsupported --op value: " + cfg.op);
        }

        if (cfg.csv_only) {
            print_aggregate_csv(all_stats);
        } else {
            print_aggregate_table(all_stats);
            std::cout << "\n=== Raw Trial CSV ===\n";
            profiler.print_csv();
        }

        append_results_csv(cfg.append_csv, all_stats, cfg);

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}
