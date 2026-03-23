#include "ckks_runner.h"

#include <iomanip>
#include <iostream>
#include <vector>

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

int main() {
    try {
        CKKSRuntime runtime;

        std::cout << runtime.info() << "\n";

        runtime.init(2, 50, 8);
        runtime.keygen({1, -1, 2, -2});

        std::cout << runtime.info() << "\n";

        std::vector<double> x = {1.0, 2.0, 3.0, 4.0};

        auto y0 = runtime.encrypt_decrypt(x);
        auto y1 = runtime.add_plain(x, 5.0);
        auto y2 = runtime.mul_plain(x, 2.0);
        auto y3 = runtime.rotate(x, 1);

        print_vector("input", x);
        print_vector("encrypt_decrypt", y0);
        print_vector("add_plain(+5)", y1);
        print_vector("mul_plain(*2)", y2);
        print_vector("rotate(+1)", y3);

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}