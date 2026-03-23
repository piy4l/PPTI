#ifndef CKKS_RUNNER_H
#define CKKS_RUNNER_H

#include "openfhe.h"

#include <string>
#include <vector>

class CKKSRuntime {
public:
    CKKSRuntime();

    void init(uint32_t multDepth = 2,
              uint32_t scaleModSize = 50,
              uint32_t batchSize = 8);

    void keygen(const std::vector<int32_t>& rotationIndices = {1, -1, 2, -2});

    std::vector<double> encrypt_decrypt(const std::vector<double>& x);
    std::vector<double> add_plain(const std::vector<double>& x, double c);
    std::vector<double> mul_plain(const std::vector<double>& x, double c);
    std::vector<double> rotate(const std::vector<double>& x, int steps);

    std::string info() const;

private:
    void require_ready(const std::string& fn) const;
    std::vector<double> decrypt_to_vector(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ct,
                                          std::size_t length) const;

private:
    bool initialized_;
    bool keys_generated_;

    uint32_t multDepth_;
    uint32_t scaleModSize_;
    uint32_t batchSize_;

    lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc_;
    lbcrypto::KeyPair<lbcrypto::DCRTPoly> keys_;
};

#endif