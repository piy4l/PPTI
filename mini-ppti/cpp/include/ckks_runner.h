#ifndef CKKS_RUNNER_H
#define CKKS_RUNNER_H

#include <vector>
#include <string>

class CKKSRuntime {
public:
    CKKSRuntime();

    void init();
    void keygen();

    std::vector<double> encrypt_decrypt(const std::vector<double>& x);
    std::vector<double> add_plain(const std::vector<double>& x, double c);
    std::vector<double> mul_plain(const std::vector<double>& x, double c);
    std::vector<double> rotate(const std::vector<double>& x, int steps);

    std::string info() const;

private:
    bool initialized_;
    bool keys_generated_;
};

#endif