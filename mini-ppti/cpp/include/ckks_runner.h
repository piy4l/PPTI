#ifndef CKKS_RUNNER_H
#define CKKS_RUNNER_H

#include "openfhe.h"

#include <cstdint>
#include <string>
#include <vector>

class CKKSRuntime {
public:
    using Plaintext = lbcrypto::Plaintext;
    using Ciphertext = lbcrypto::Ciphertext<lbcrypto::DCRTPoly>;

    CKKSRuntime();

    void init(uint32_t multDepth = 2,
              uint32_t scaleModSize = 50,
              uint32_t batchSize = 8);

    void keygen(const std::vector<int32_t>& rotationIndices = {1, -1, 2, -2});

    Plaintext encode(const std::vector<double>& x) const;
    Ciphertext encrypt(const Plaintext& plaintext) const;
    std::vector<double> decrypt_and_decode(const Ciphertext& ciphertext,
                                           std::size_t length) const;

    Ciphertext add(const Ciphertext& lhs, const Ciphertext& rhs) const;
    Ciphertext multiply(const Ciphertext& lhs, const Ciphertext& rhs) const;
    Ciphertext multiply_plain(const Ciphertext& lhs, double scalar) const;
    Ciphertext multiply_plaintext(const Ciphertext& lhs,
                                 const std::vector<double>& rhs) const;
    Ciphertext rotate(const Ciphertext& ciphertext, int steps) const;

    std::vector<double> encrypt_decrypt(const std::vector<double>& x) const;
    std::vector<double> add_plain(const std::vector<double>& x, double c) const;
    std::vector<double> mul_plain(const std::vector<double>& x, double c) const;
    std::vector<double> rotate_plain(const std::vector<double>& x, int steps) const;

    std::string info() const;

private:
    void require_initialized(const std::string& fn) const;
    void require_ready(const std::string& fn) const;
    std::vector<double> decrypt_to_vector(const Ciphertext& ct,
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
