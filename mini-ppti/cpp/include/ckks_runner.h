#ifndef CKKS_RUNNER_H
#define CKKS_RUNNER_H

#include "openfhe.h"

#include <cstddef>
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
    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> encrypt_only(const std::vector<double>& x);
    std::vector<double> decrypt_only(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ct,
                                     std::size_t length);

    std::vector<double> add_plain(const std::vector<double>& x, double c);
    std::vector<double> mul_plain(const std::vector<double>& x, double c);
    std::vector<double> add_ct_ct(const std::vector<double>& x,
                                  const std::vector<double>& y);
    std::vector<double> mul_ct_ct(const std::vector<double>& x,
                                  const std::vector<double>& y);
    std::vector<double> rotate(const std::vector<double>& x, int steps);
    std::vector<double> toy_transformer_block_plain(const std::vector<double>& x);
    std::vector<double> toy_transformer_block(const std::vector<double>& x);
    std::vector<double> tiny_bert_block_plain(const std::vector<double>& x);
    std::vector<double> tiny_bert_block(const std::vector<double>& x);
    std::vector<double> tiny_text_classifier_plain(const std::vector<int>& token_ids);
    std::vector<double> tiny_text_classifier(const std::vector<int>& token_ids);
    std::vector<double> bench_gelu(const std::vector<double>& x);
    std::vector<double> bench_softmax(const std::vector<double>& x);
    std::vector<double> bench_matmul_custom(const std::vector<double>& x,
                                            const std::vector<std::vector<double>>& weights,
                                            bool optimized);
    std::vector<double> bench_matmul(const std::vector<double>& x, bool optimized);
    std::vector<double> bench_layernorm(const std::vector<double>& x, bool optimized);
    std::vector<double> bench_argmax(const std::vector<double>& x, bool optimized);
    void trace_tiny_bert_block(const std::vector<double>& x);
    void trace_tiny_text_classifier(const std::vector<int>& token_ids);

    std::string info() const;

private:
    void require_ready(const std::string& fn) const;
    std::vector<double> decrypt_to_vector(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ct,
                                          std::size_t length) const;
    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> add_constant(
        const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ct,
        double c,
        std::size_t length) const;
    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> mul_constant(
        const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ct,
        double c) const;
    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> softmax_like_approx(
        const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ct,
        std::size_t length) const;
    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> gelu_like_approx(
        const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ct,
        std::size_t length) const;
    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> layernorm_like_approx(
        const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ct,
        std::size_t length) const;
    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> apply_block_diagonal_he(
        const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ct,
        const std::vector<std::vector<double>>& weights,
        std::size_t seq_len,
        std::size_t hidden) const;
    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> apply_block_diagonal_he_baseline(
        const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ct,
        const std::vector<std::vector<double>>& weights,
        std::size_t seq_len,
        std::size_t hidden) const;
    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> apply_sequence_mix_he(
        const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ct,
        const std::vector<std::vector<double>>& weights,
        std::size_t seq_len,
        std::size_t hidden) const;
    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> tiny_bert_block_he_core(
        const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& input_ct,
        std::size_t length) const;
    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> reduce_sum_he(
        const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ct,
        std::size_t length,
        bool optimized) const;
    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> max_pairwise_he(
        const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& lhs,
        const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& rhs) const;

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
