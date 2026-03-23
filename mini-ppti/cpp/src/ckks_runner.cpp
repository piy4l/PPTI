#include "ckks_runner.h"

#include <sstream>
#include <stdexcept>

CKKSRuntime::CKKSRuntime()
    : initialized_(false), keys_generated_(false) {}

void CKKSRuntime::init() {
    initialized_ = true;
}

void CKKSRuntime::keygen() {
    if (!initialized_) {
        throw std::runtime_error("CKKSRuntime::keygen() called before init()");
    }
    keys_generated_ = true;
}

std::vector<double> CKKSRuntime::encrypt_decrypt(const std::vector<double>& x) {
    if (!initialized_ || !keys_generated_) {
        throw std::runtime_error("encrypt_decrypt() requires init() and keygen()");
    }

    // Placeholder behavior for now.
    // Later this will perform real CKKS encode/encrypt/decrypt/decode.
    return x;
}

std::vector<double> CKKSRuntime::add_plain(const std::vector<double>& x, double c) {
    if (!initialized_ || !keys_generated_) {
        throw std::runtime_error("add_plain() requires init() and keygen()");
    }

    std::vector<double> y = x;
    for (double& v : y) {
        v += c;
    }
    return y;
}

std::vector<double> CKKSRuntime::mul_plain(const std::vector<double>& x, double c) {
    if (!initialized_ || !keys_generated_) {
        throw std::runtime_error("mul_plain() requires init() and keygen()");
    }

    std::vector<double> y = x;
    for (double& v : y) {
        v *= c;
    }
    return y;
}

std::vector<double> CKKSRuntime::rotate(const std::vector<double>& x, int steps) {
    if (!initialized_ || !keys_generated_) {
        throw std::runtime_error("rotate() requires init() and keygen()");
    }

    if (x.empty()) {
        return x;
    }

    std::vector<double> y(x.size());
    const int n = static_cast<int>(x.size());

    int k = steps % n;
    if (k < 0) {
        k += n;
    }

    for (int i = 0; i < n; ++i) {
        y[(i + k) % n] = x[i];
    }

    return y;
}

std::string CKKSRuntime::info() const {
    std::ostringstream oss;
    oss << "CKKSRuntime(initialized=" << (initialized_ ? "true" : "false")
        << ", keys_generated=" << (keys_generated_ ? "true" : "false") << ")";
    return oss.str();
}