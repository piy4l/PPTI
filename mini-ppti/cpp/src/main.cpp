#include "ckks_runner.h"
#include "profiler.h"

#include <iomanip>
#include <iostream>
#include <string>
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
        Profiler profiler;

        std::cout << runtime.info() << "\n";

        runtime.init(2, 50, 8);
        runtime.keygen({1, -1, 2, -2});

        std::cout << runtime.info() << "\n";

        std::vector<double> x = {1.0, 2.0, 3.0, 4.0};

        std::vector<double> y0, y1, y2, y3;

        {
            ScopedTimer timer(profiler, "encrypt_decrypt", x.size());
            y0 = runtime.encrypt_decrypt(x);
        }

        {
            ScopedTimer timer(profiler, "add_plain", x.size(), "c=5.0");
            y1 = runtime.add_plain(x, 5.0);
        }

        {
            ScopedTimer timer(profiler, "mul_plain", x.size(), "c=2.0");
            y2 = runtime.mul_plain(x, 2.0);
        }

        {
            ScopedTimer timer(profiler, "rotate", x.size(), "steps=1");
            y3 = runtime.rotate(x, 1);
        }

        print_vector("input", x);
        print_vector("encrypt_decrypt", y0);
        print_vector("add_plain(+5)", y1);
        print_vector("mul_plain(*2)", y2);
        print_vector("rotate(+1)", y3);

        profiler.print_summary();
        profiler.print_csv();

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}