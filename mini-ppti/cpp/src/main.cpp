#include "ckks_runner.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string op = "pipeline_demo";
    std::size_t n = 8;
};

void print_usage(const char* argv0) {
    std::cout << "Usage: " << argv0
              << " [--op pipeline_demo|linear_demo|linear_layer_demo|activation_demo|mlp_demo] [--n 8]\n";
}

Options parse_args(int argc, char* argv[]) {
    Options options;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--op") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--op requires a value");
            }
            options.op = argv[++i];
        } else if (arg == "--n") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--n requires a value");
            }
            options.n = static_cast<std::size_t>(std::stoul(argv[++i]));
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (options.n == 0) {
        throw std::runtime_error("--n must be greater than 0");
    }

    return options;
}

std::vector<double> make_input(std::size_t n) {
    std::vector<double> x(n);
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = static_cast<double>(i + 1);
    }
    return x;
}

std::vector<double> expected_pipeline_output(const std::vector<double>& input) {
    std::vector<double> expected(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        expected[i] = (input[i] * 2.0) + input[i];
    }
    return expected;
}

std::vector<double> expected_activation_output(const std::vector<double>& input) {
    std::vector<double> expected(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        const double x = input[i];
        expected[i] = (x * x) + x + 1.0;
    }
    return expected;
}

std::vector<double> apply_activation_polynomial(const std::vector<double>& input) {
    return expected_activation_output(input);
}

double max_abs_error(const std::vector<double>& expected, const std::vector<double>& actual) {
    double max_error = 0.0;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        max_error = std::max(max_error, std::abs(expected[i] - actual[i]));
    }
    return max_error;
}

bool is_power_of_two(std::size_t value) {
    return value > 0 && (value & (value - 1)) == 0;
}

std::vector<double> make_weights(std::size_t n) {
    std::vector<double> w(n);
    for (std::size_t i = 0; i < n; ++i) {
        w[i] = 0.5 * static_cast<double>(i + 1);
    }
    return w;
}

double dot_product(const std::vector<double>& lhs, const std::vector<double>& rhs) {
    double sum = 0.0;
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        sum += lhs[i] * rhs[i];
    }
    return sum;
}

std::vector<std::vector<double>> make_weight_matrix(std::size_t n) {
    return {
        make_weights(n),
        std::vector<double>(n, 0.0),
        std::vector<double>(n, 1.0)
    };
}

std::vector<std::vector<double>> make_mlp_hidden_weights(std::size_t n) {
    std::vector<std::vector<double>> weights(2, std::vector<double>(n, 0.0));
    for (std::size_t i = 0; i < n; ++i) {
        weights[0][i] = 0.25 * static_cast<double>(i + 1);
        weights[1][i] = (i % 2 == 0) ? 0.5 : -0.25;
    }
    return weights;
}

std::vector<double> make_mlp_output_weights() {
    return {1.5, -0.5};
}

std::vector<double> expected_linear_layer_output(const std::vector<double>& input,
                                                 std::vector<std::vector<double>>& weights) {
    std::vector<double> expected;
    expected.reserve(weights.size());
    for (const auto& row : weights) {
        expected.push_back(dot_product(input, row));
    }
    return expected;
}

std::vector<double> expected_mlp_output(const std::vector<double>& input,
                                        std::vector<std::vector<double>>& hidden_weights,
                                        const std::vector<double>& output_weights) {
    std::vector<double> hidden = expected_linear_layer_output(input, hidden_weights);
    hidden = apply_activation_polynomial(hidden);
    return {dot_product(hidden, output_weights)};
}

CKKSRuntime::Ciphertext apply_activation_polynomial(CKKSRuntime& runtime,
                                                    const CKKSRuntime::Ciphertext& ciphertext) {
    const auto squared = runtime.multiply(ciphertext, ciphertext);
    const auto shifted = runtime.add(squared, ciphertext);
    return runtime.add(shifted, runtime.encrypt(runtime.encode({1.0})));
}

void print_vector(const std::string& label, const std::vector<double>& values) {
    std::cout << label << ": [";
    for (std::size_t i = 0; i < values.size(); ++i) {
        std::cout << std::fixed << std::setprecision(6) << values[i];
        if (i + 1 < values.size()) {
            std::cout << ", ";
        }
    }
    std::cout << "]\n";
}

void run_pipeline_demo(std::size_t n) {
    CKKSRuntime runtime;
    runtime.init(2, 50, static_cast<uint32_t>(n));
    runtime.keygen({1, -1});

    const std::vector<double> input = make_input(n);
    const auto plaintext = runtime.encode(input);
    const auto ciphertext = runtime.encrypt(plaintext);
    const auto multiplied = runtime.multiply_plain(ciphertext, 2.0);
    const auto summed = runtime.add(multiplied, ciphertext);
    const std::vector<double> actual = runtime.decrypt_and_decode(summed, input.size());
    const std::vector<double> expected = expected_pipeline_output(input);

    std::cout << runtime.info() << "\n";
    std::cout << "pipeline: plaintext -> encode -> encrypt -> mul_plain(*2) -> add_ct_ct -> decrypt -> decode\n";
    print_vector("input", input);
    print_vector("expected", expected);
    print_vector("actual", actual);
    std::cout << "max_abs_error: " << std::fixed << std::setprecision(8)
              << max_abs_error(expected, actual) << "\n";
}

CKKSRuntime::Ciphertext reduce_sum_slots(CKKSRuntime& runtime,
                                         CKKSRuntime::Ciphertext ciphertext,
                                         std::size_t n) {
    for (std::size_t step = 1; step < n; step *= 2) {
        ciphertext = runtime.add(ciphertext, runtime.rotate(ciphertext, static_cast<int>(step)));
    }
    return ciphertext;
}

double decrypt_slot0(CKKSRuntime& runtime,
                     const CKKSRuntime::Ciphertext& ciphertext,
                     std::size_t length) {
    return runtime.decrypt_and_decode(ciphertext, length).front();
}

void run_linear_demo(std::size_t n) {
    if (!is_power_of_two(n)) {
        throw std::runtime_error("linear_demo requires --n to be a power of two");
    }

    std::vector<int32_t> rotation_indices;
    for (std::size_t step = 1; step < n; step *= 2) {
        rotation_indices.push_back(static_cast<int32_t>(step));
    }

    CKKSRuntime runtime;
    runtime.init(3, 50, static_cast<uint32_t>(n));
    runtime.keygen(rotation_indices);

    const std::vector<double> input = make_input(n);
    const std::vector<double> weights = make_weights(n);
    const double expected = dot_product(input, weights);

    const auto plaintext = runtime.encode(input);
    const auto ciphertext = runtime.encrypt(plaintext);
    auto accumulated = reduce_sum_slots(runtime, runtime.multiply_plaintext(ciphertext, weights), n);

    const std::vector<double> actual_slots = runtime.decrypt_and_decode(accumulated, input.size());
    const double actual = actual_slots.front();

    std::cout << runtime.info() << "\n";
    std::cout << "pipeline: plaintext -> encode -> encrypt -> mul_ct_pt(weights) -> rotate/add reduce -> decrypt -> decode\n";
    print_vector("input", input);
    print_vector("weights", weights);
    print_vector("decrypted_slots", actual_slots);
    std::cout << "expected_dot_product: " << std::fixed << std::setprecision(6) << expected << "\n";
    std::cout << "actual_slot0: " << std::fixed << std::setprecision(6) << actual << "\n";
    std::cout << "abs_error: " << std::fixed << std::setprecision(8)
              << std::abs(expected - actual) << "\n";
}

void run_linear_layer_demo(std::size_t n) {
    if (!is_power_of_two(n)) {
        throw std::runtime_error("linear_layer_demo requires --n to be a power of two");
    }

    std::vector<int32_t> rotation_indices;
    for (std::size_t step = 1; step < n; step *= 2) {
        rotation_indices.push_back(static_cast<int32_t>(step));
    }

    CKKSRuntime runtime;
    runtime.init(3, 50, static_cast<uint32_t>(n));
    runtime.keygen(rotation_indices);

    const std::vector<double> input = make_input(n);
    auto weights = make_weight_matrix(n);
    weights[1] = std::vector<double>(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        weights[1][i] = (i % 2 == 0) ? 1.0 : 0.0;
    }
    const std::vector<double> expected = expected_linear_layer_output(input, weights);

    const auto plaintext = runtime.encode(input);
    const auto ciphertext = runtime.encrypt(plaintext);

    std::vector<double> actual;
    actual.reserve(weights.size());

    for (const auto& row : weights) {
        const auto reduced = reduce_sum_slots(runtime, runtime.multiply_plaintext(ciphertext, row), n);
        const auto decrypted = runtime.decrypt_and_decode(reduced, input.size());
        actual.push_back(decrypted.front());
    }

    std::cout << runtime.info() << "\n";
    std::cout << "pipeline: plaintext -> encode -> encrypt -> repeated mul_ct_pt(row) -> rotate/add reduce -> decrypt -> decode\n";
    print_vector("input", input);
    for (std::size_t row = 0; row < weights.size(); ++row) {
        print_vector("weights[" + std::to_string(row) + "]", weights[row]);
    }
    print_vector("expected_outputs", expected);
    print_vector("actual_outputs", actual);
    std::cout << "max_abs_error: " << std::fixed << std::setprecision(8)
              << max_abs_error(expected, actual) << "\n";
}

void run_activation_demo(std::size_t n) {
    CKKSRuntime runtime;
    runtime.init(3, 50, static_cast<uint32_t>(n));
    runtime.keygen({1, -1});

    const std::vector<double> input = make_input(n);
    const std::vector<double> ones(n, 1.0);
    const std::vector<double> expected = expected_activation_output(input);

    const auto plaintext = runtime.encode(input);
    const auto ciphertext = runtime.encrypt(plaintext);
    const auto squared = runtime.multiply(ciphertext, ciphertext);
    const auto shifted = runtime.add(squared, ciphertext);
    const auto activated = runtime.add(shifted, runtime.encrypt(runtime.encode(ones)));
    const std::vector<double> actual = runtime.decrypt_and_decode(activated, input.size());

    std::cout << runtime.info() << "\n";
    std::cout << "pipeline: plaintext -> encode -> encrypt -> square -> add_ct_ct -> add_constant(1) -> decrypt -> decode\n";
    print_vector("input", input);
    print_vector("expected", expected);
    print_vector("actual", actual);
    std::cout << "max_abs_error: " << std::fixed << std::setprecision(8)
              << max_abs_error(expected, actual) << "\n";
}

void run_mlp_demo(std::size_t n) {
    if (!is_power_of_two(n)) {
        throw std::runtime_error("mlp_demo requires --n to be a power of two");
    }

    std::vector<int32_t> rotation_indices;
    for (std::size_t step = 1; step < n; step *= 2) {
        rotation_indices.push_back(static_cast<int32_t>(step));
    }

    CKKSRuntime runtime;
    runtime.init(5, 50, static_cast<uint32_t>(n));
    runtime.keygen(rotation_indices);

    const std::vector<double> input = make_input(n);
    auto hidden_weights = make_mlp_hidden_weights(n);
    const std::vector<double> output_weights = make_mlp_output_weights();
    const std::vector<double> expected = expected_mlp_output(input, hidden_weights, output_weights);

    const auto input_ciphertext = runtime.encrypt(runtime.encode(input));

    std::vector<double> hidden_linear;
    hidden_linear.reserve(hidden_weights.size());
    std::vector<CKKSRuntime::Ciphertext> hidden_activated_ciphertexts;
    hidden_activated_ciphertexts.reserve(hidden_weights.size());

    for (const auto& row : hidden_weights) {
        const auto reduced = reduce_sum_slots(runtime, runtime.multiply_plaintext(input_ciphertext, row), n);
        hidden_linear.push_back(decrypt_slot0(runtime, reduced, n));
        hidden_activated_ciphertexts.push_back(apply_activation_polynomial(runtime, reduced));
    }

    const std::vector<double> hidden_activated = apply_activation_polynomial(hidden_linear);
    auto output_ciphertext = runtime.multiply_plain(hidden_activated_ciphertexts[0], output_weights[0]);
    for (std::size_t i = 1; i < hidden_activated_ciphertexts.size(); ++i) {
        output_ciphertext = runtime.add(
            output_ciphertext,
            runtime.multiply_plain(hidden_activated_ciphertexts[i], output_weights[i]));
    }
    const std::vector<double> actual = {decrypt_slot0(runtime, output_ciphertext, 1)};

    std::cout << runtime.info() << "\n";
    std::cout << "pipeline: input -> encrypted linear layer -> encrypted activation -> encrypted output layer -> decrypt\n";
    print_vector("input", input);
    for (std::size_t row = 0; row < hidden_weights.size(); ++row) {
        print_vector("hidden_weights[" + std::to_string(row) + "]", hidden_weights[row]);
    }
    print_vector("hidden_linear_expected", expected_linear_layer_output(input, hidden_weights));
    print_vector("hidden_linear_actual", hidden_linear);
    print_vector("hidden_activated", hidden_activated);
    print_vector("output_weights", output_weights);
    print_vector("expected_output", expected);
    print_vector("actual_output", actual);
    std::cout << "max_abs_error: " << std::fixed << std::setprecision(8)
              << max_abs_error(expected, actual) << "\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        const Options options = parse_args(argc, argv);

        if (options.op == "pipeline_demo") {
            run_pipeline_demo(options.n);
            return 0;
        }
        if (options.op == "linear_demo") {
            run_linear_demo(options.n);
            return 0;
        }
        if (options.op == "linear_layer_demo") {
            run_linear_layer_demo(options.n);
            return 0;
        }
        if (options.op == "activation_demo") {
            run_activation_demo(options.n);
            return 0;
        }
        if (options.op == "mlp_demo") {
            run_mlp_demo(options.n);
            return 0;
        }

        throw std::runtime_error("unsupported --op: " + options.op);
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        print_usage(argv[0]);
        return 1;
    }
}
