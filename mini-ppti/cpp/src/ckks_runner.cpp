#include "ckks_runner.h"

#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

using namespace lbcrypto;

namespace {

constexpr std::size_t kTinyBertSeqLen = 4;
constexpr std::size_t kTinyBertHidden = 4;
constexpr std::size_t kTinyBertSlots = kTinyBertSeqLen * kTinyBertHidden;

using Matrix = std::vector<std::vector<double>>;

const Matrix& tiny_bert_wq() {
    static const Matrix w = {
        {0.12, -0.04, 0.03, 0.02},
        {0.01, 0.11, -0.05, 0.04},
        {-0.02, 0.05, 0.10, 0.03},
        {0.04, 0.02, 0.01, 0.09},
    };
    return w;
}

const Matrix& tiny_bert_wk() {
    static const Matrix w = {
        {0.10, 0.03, -0.02, 0.01},
        {-0.03, 0.09, 0.04, 0.02},
        {0.02, -0.01, 0.11, 0.05},
        {0.01, 0.02, 0.03, 0.08},
    };
    return w;
}

const Matrix& tiny_bert_wv() {
    static const Matrix w = {
        {0.11, 0.02, 0.01, -0.03},
        {0.04, 0.10, 0.03, 0.00},
        {-0.01, 0.02, 0.12, 0.04},
        {0.03, -0.02, 0.05, 0.09},
    };
    return w;
}

const Matrix& tiny_bert_wo() {
    static const Matrix w = {
        {0.09, 0.02, 0.00, 0.03},
        {-0.01, 0.08, 0.04, 0.02},
        {0.03, 0.01, 0.10, -0.02},
        {0.02, 0.03, 0.01, 0.07},
    };
    return w;
}

const Matrix& tiny_bert_w1() {
    static const Matrix w = {
        {0.16, -0.03, 0.02, 0.01},
        {0.04, 0.15, -0.02, 0.03},
        {0.02, 0.01, 0.14, -0.04},
        {-0.01, 0.03, 0.02, 0.13},
    };
    return w;
}

const Matrix& tiny_bert_w2() {
    static const Matrix w = {
        {0.07, 0.02, -0.01, 0.01},
        {0.01, 0.08, 0.02, 0.00},
        {0.00, 0.01, 0.07, 0.02},
        {0.02, -0.01, 0.01, 0.08},
    };
    return w;
}

const Matrix& tiny_bert_attention_mix() {
    static const Matrix w = {
        {0.45, 0.25, 0.20, 0.10},
        {0.20, 0.40, 0.25, 0.15},
        {0.15, 0.25, 0.40, 0.20},
        {0.10, 0.20, 0.25, 0.45},
    };
    return w;
}

const Matrix& bench_matmul_weights() {
    static const Matrix w = {
        {0.16, -0.03, 0.02, 0.01},
        {0.04, 0.15, -0.02, 0.03},
        {0.02, 0.01, 0.14, -0.04},
        {-0.01, 0.03, 0.02, 0.13},
    };
    return w;
}

const Matrix& tiny_text_avg_pool() {
    static const Matrix w = {
        {0.25, 0.25, 0.25, 0.25},
        {0.25, 0.25, 0.25, 0.25},
        {0.25, 0.25, 0.25, 0.25},
        {0.25, 0.25, 0.25, 0.25},
    };
    return w;
}

const Matrix& tiny_text_classifier_head() {
    static const Matrix w = {
        {0.20, -0.10, 0.05, 0.08},
        {-0.12, 0.18, 0.07, -0.04},
        {0.00, 0.00, 0.00, 0.00},
        {0.00, 0.00, 0.00, 0.00},
    };
    return w;
}

const Matrix& tiny_text_token_embeddings() {
    static const Matrix w = {
        {0.10, 0.00, 0.05, -0.02},
        {0.08, 0.03, 0.02, 0.01},
        {0.02, 0.09, -0.01, 0.04},
        {0.00, 0.04, 0.10, 0.03},
        {-0.03, 0.02, 0.07, 0.09},
        {0.05, -0.02, 0.04, 0.08},
        {0.07, 0.06, -0.03, 0.02},
        {-0.01, 0.08, 0.05, 0.00},
    };
    return w;
}

const Matrix& tiny_text_position_embeddings() {
    static const Matrix w = {
        {0.01, 0.02, 0.00, 0.01},
        {0.02, 0.01, 0.01, 0.00},
        {0.00, 0.03, 0.02, 0.01},
        {0.01, 0.00, 0.03, 0.02},
    };
    return w;
}

void require_tiny_bert_shape(std::size_t n) {
    if (n != kTinyBertSlots) {
        throw std::runtime_error(
            "tiny_bert_block requires n=16 (seq_len=4, hidden_size=4)");
    }
}

void require_tiny_text_tokens(const std::vector<int>& token_ids) {
    if (token_ids.size() != kTinyBertSeqLen) {
        throw std::runtime_error("tiny_text_classifier requires exactly 4 token ids");
    }

    const int vocab_size = static_cast<int>(tiny_text_token_embeddings().size());
    for (int token_id : token_ids) {
        if (token_id < 0 || token_id >= vocab_size) {
            throw std::runtime_error("token id out of range for tiny_text_classifier");
        }
    }
}

std::vector<double> rotate_plain(const std::vector<double>& x, int shift) {
    const std::size_t n = x.size();
    std::vector<double> out(n);
    if (n == 0) {
        return out;
    }

    const int normalized = ((shift % static_cast<int>(n)) + static_cast<int>(n)) %
                           static_cast<int>(n);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = x[(i + normalized) % n];
    }
    return out;
}

std::vector<double> add_constant_plain(const std::vector<double>& x, double c) {
    std::vector<double> out(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        out[i] = x[i] + c;
    }
    return out;
}

std::vector<double> mul_constant_plain(const std::vector<double>& x, double c) {
    std::vector<double> out(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        out[i] = x[i] * c;
    }
    return out;
}

std::vector<double> add_vec_plain(const std::vector<double>& x,
                                  const std::vector<double>& y) {
    if (x.size() != y.size()) {
        throw std::runtime_error("add_vec_plain() requires equal-length inputs");
    }

    std::vector<double> out(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        out[i] = x[i] + y[i];
    }
    return out;
}

std::vector<double> mul_vec_plain(const std::vector<double>& x,
                                  const std::vector<double>& y) {
    if (x.size() != y.size()) {
        throw std::runtime_error("mul_vec_plain() requires equal-length inputs");
    }

    std::vector<double> out(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        out[i] = x[i] * y[i];
    }
    return out;
}

std::vector<double> softmax_like_approx_plain(const std::vector<double>& x) {
    auto x2 = mul_vec_plain(x, x);
    auto out = add_constant_plain(x, 1.0);
    auto half_x2 = mul_constant_plain(x2, 0.5);
    return add_vec_plain(out, half_x2);
}

std::vector<double> gelu_like_approx_plain(const std::vector<double>& x) {
    auto x2 = mul_vec_plain(x, x);
    auto x3 = mul_vec_plain(x2, x);
    auto half_x = mul_constant_plain(x, 0.5);
    auto eighth_x3 = mul_constant_plain(x3, 0.125);
    auto out = add_constant_plain(half_x, 0.5);
    return add_vec_plain(out, eighth_x3);
}

std::vector<double> layernorm_like_approx_plain(const std::vector<double>& x) {
    auto x2 = mul_vec_plain(x, x);
    auto neg_quarter_x2 = mul_constant_plain(x2, -0.25);
    auto scaled_x = mul_constant_plain(x, 0.75);
    auto out = add_constant_plain(scaled_x, 0.5);
    return add_vec_plain(out, neg_quarter_x2);
}

std::vector<double> toy_transformer_block_plain(const std::vector<double>& x) {
    auto normalized_input = mul_constant_plain(x, 0.1);

    auto q = mul_constant_plain(normalized_input, 1.10);
    auto k = mul_constant_plain(normalized_input, 0.90);
    auto v = mul_constant_plain(normalized_input, 1.30);

    auto attn_scores = mul_vec_plain(q, k);
    auto attn_scale = mul_constant_plain(attn_scores, 0.25);
    auto attn_weights = softmax_like_approx_plain(attn_scale);
    auto attn_values = mul_vec_plain(attn_weights, v);
    auto attn_out = mul_constant_plain(attn_values, 0.80);
    auto attn_residual = add_vec_plain(attn_out, normalized_input);

    auto ln1_out = layernorm_like_approx_plain(attn_residual);

    auto mlp_fc1 = mul_constant_plain(ln1_out, 1.70);
    auto mlp_act = gelu_like_approx_plain(mlp_fc1);
    auto mlp_fc2 = mul_constant_plain(mlp_act, 0.60);
    auto mlp_residual = add_vec_plain(mlp_fc2, attn_residual);

    return layernorm_like_approx_plain(mlp_residual);
}

std::vector<double> apply_block_diagonal_plain(const std::vector<double>& x,
                                               const Matrix& w,
                                               std::size_t seq_len,
                                               std::size_t hidden) {
    if (x.size() != seq_len * hidden) {
        throw std::runtime_error("apply_block_diagonal_plain() got unexpected shape");
    }

    std::vector<double> out(x.size(), 0.0);
    for (std::size_t token = 0; token < seq_len; ++token) {
        const std::size_t base = token * hidden;
        for (std::size_t i = 0; i < hidden; ++i) {
            double acc = 0.0;
            for (std::size_t j = 0; j < hidden; ++j) {
                acc += w[i][j] * x[base + j];
            }
            out[base + i] = acc;
        }
    }
    return out;
}

std::vector<double> apply_sequence_mix_plain(const std::vector<double>& x,
                                             const Matrix& w,
                                             std::size_t seq_len,
                                             std::size_t hidden) {
    if (x.size() != seq_len * hidden) {
        throw std::runtime_error("apply_sequence_mix_plain() got unexpected shape");
    }

    std::vector<double> out(x.size(), 0.0);
    for (std::size_t token_out = 0; token_out < seq_len; ++token_out) {
        for (std::size_t channel = 0; channel < hidden; ++channel) {
            double acc = 0.0;
            for (std::size_t token_in = 0; token_in < seq_len; ++token_in) {
                acc += w[token_out][token_in] * x[token_in * hidden + channel];
            }
            out[token_out * hidden + channel] = acc;
        }
    }
    return out;
}

std::vector<double> tiny_bert_block_plain(const std::vector<double>& x) {
    require_tiny_bert_shape(x.size());

    auto normalized_input = mul_constant_plain(x, 0.1);

    auto q = apply_block_diagonal_plain(
        normalized_input, tiny_bert_wq(), kTinyBertSeqLen, kTinyBertHidden);
    auto k = apply_block_diagonal_plain(
        normalized_input, tiny_bert_wk(), kTinyBertSeqLen, kTinyBertHidden);
    auto v = apply_block_diagonal_plain(
        normalized_input, tiny_bert_wv(), kTinyBertSeqLen, kTinyBertHidden);

    auto gates = softmax_like_approx_plain(mul_vec_plain(q, k));
    auto gated_v = mul_vec_plain(gates, v);
    auto mixed = apply_sequence_mix_plain(
        gated_v, tiny_bert_attention_mix(), kTinyBertSeqLen, kTinyBertHidden);
    auto attn_out = apply_block_diagonal_plain(
        mixed, tiny_bert_wo(), kTinyBertSeqLen, kTinyBertHidden);
    auto attn_residual = add_vec_plain(attn_out, normalized_input);

    auto ln1_out = layernorm_like_approx_plain(attn_residual);
    auto mlp_fc1 = apply_block_diagonal_plain(
        ln1_out, tiny_bert_w1(), kTinyBertSeqLen, kTinyBertHidden);
    auto mlp_act = gelu_like_approx_plain(mlp_fc1);
    auto mlp_fc2 = apply_block_diagonal_plain(
        mlp_act, tiny_bert_w2(), kTinyBertSeqLen, kTinyBertHidden);
    auto mlp_residual = add_vec_plain(mlp_fc2, attn_residual);

    return layernorm_like_approx_plain(mlp_residual);
}

std::vector<double> encode_tiny_text_tokens_plain(const std::vector<int>& token_ids) {
    require_tiny_text_tokens(token_ids);

    std::vector<double> out(kTinyBertSlots, 0.0);
    for (std::size_t pos = 0; pos < kTinyBertSeqLen; ++pos) {
        const auto& token_embed = tiny_text_token_embeddings()[static_cast<std::size_t>(token_ids[pos])];
        const auto& pos_embed = tiny_text_position_embeddings()[pos];
        for (std::size_t channel = 0; channel < kTinyBertHidden; ++channel) {
            out[pos * kTinyBertHidden + channel] = token_embed[channel] + pos_embed[channel];
        }
    }
    return out;
}

std::vector<double> tiny_text_classifier_plain(const std::vector<int>& token_ids) {
    auto encoded = encode_tiny_text_tokens_plain(token_ids);
    auto hidden = tiny_bert_block_plain(encoded);
    auto pooled = apply_sequence_mix_plain(
        hidden, tiny_text_avg_pool(), kTinyBertSeqLen, kTinyBertHidden);
    auto logits_repeated = apply_block_diagonal_plain(
        pooled, tiny_text_classifier_head(), kTinyBertSeqLen, kTinyBertHidden);
    return {logits_repeated[0], logits_repeated[1]};
}

struct StageErrorStats {
    double max_abs_error = 0.0;
    double mean_abs_error = 0.0;
    double rmse = 0.0;
};

StageErrorStats compute_stage_error_stats(const std::vector<double>& plain,
                                          const std::vector<double>& he) {
    if (plain.size() != he.size()) {
        throw std::runtime_error("compute_stage_error_stats() requires equal-length inputs");
    }

    StageErrorStats stats;
    if (plain.empty()) {
        return stats;
    }

    double abs_sum = 0.0;
    double sq_sum = 0.0;
    for (std::size_t i = 0; i < plain.size(); ++i) {
        const double err = std::abs(plain[i] - he[i]);
        stats.max_abs_error = std::max(stats.max_abs_error, err);
        abs_sum += err;
        sq_sum += err * err;
    }

    stats.mean_abs_error = abs_sum / static_cast<double>(plain.size());
    stats.rmse = std::sqrt(sq_sum / static_cast<double>(plain.size()));
    return stats;
}

}  // namespace

CKKSRuntime::CKKSRuntime()
    : initialized_(false),
      keys_generated_(false),
      multDepth_(0),
      scaleModSize_(0),
      batchSize_(0) {}

void CKKSRuntime::init(uint32_t multDepth, uint32_t scaleModSize, uint32_t batchSize) {
    multDepth_ = multDepth;
    scaleModSize_ = scaleModSize;
    batchSize_ = batchSize;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(multDepth_);
    parameters.SetScalingModSize(scaleModSize_);
    parameters.SetBatchSize(batchSize_);
    parameters.SetRingDim(std::max<uint32_t>(16384, batchSize_ * 2));
    parameters.SetSecurityLevel(HEStd_NotSet);

    cc_ = GenCryptoContext(parameters);
    cc_->Enable(PKE);
    cc_->Enable(KEYSWITCH);
    cc_->Enable(LEVELEDSHE);

    initialized_ = true;
    keys_generated_ = false;
}

void CKKSRuntime::keygen(const std::vector<int32_t>& rotationIndices) {
    if (!initialized_) {
        throw std::runtime_error("CKKSRuntime::keygen() called before init()");
    }

    keys_ = cc_->KeyGen();
    cc_->EvalMultKeyGen(keys_.secretKey);
    cc_->EvalRotateKeyGen(keys_.secretKey, rotationIndices);

    keys_generated_ = true;
}

void CKKSRuntime::require_ready(const std::string& fn) const {
    if (!initialized_ || !keys_generated_) {
        throw std::runtime_error(fn + " requires init() and keygen()");
    }
}

std::vector<double> CKKSRuntime::decrypt_to_vector(const Ciphertext<DCRTPoly>& ct,
                                                   std::size_t length) const {
    Plaintext decrypted;
    cc_->Decrypt(keys_.secretKey, ct, &decrypted);
    decrypted->SetLength(length);
    return decrypted->GetRealPackedValue();
}

Ciphertext<DCRTPoly> CKKSRuntime::encrypt_only(const std::vector<double>& x) {
    require_ready("encrypt_only()");
    Plaintext ptxt = cc_->MakeCKKSPackedPlaintext(x);
    return cc_->Encrypt(keys_.publicKey, ptxt);
}

std::vector<double> CKKSRuntime::decrypt_only(const Ciphertext<DCRTPoly>& ct,
                                              std::size_t length) {
    require_ready("decrypt_only()");
    return decrypt_to_vector(ct, length);
}

Ciphertext<DCRTPoly> CKKSRuntime::add_constant(const Ciphertext<DCRTPoly>& ct,
                                               double c,
                                               std::size_t length) const {
    std::vector<double> cvec(length, c);
    Plaintext cptxt = cc_->MakeCKKSPackedPlaintext(cvec);
    return cc_->EvalAdd(ct, cptxt);
}

Ciphertext<DCRTPoly> CKKSRuntime::mul_constant(const Ciphertext<DCRTPoly>& ct,
                                               double c) const {
    return cc_->EvalMult(ct, c);
}

Ciphertext<DCRTPoly> CKKSRuntime::softmax_like_approx(const Ciphertext<DCRTPoly>& ct,
                                                      std::size_t length) const {
    auto x2 = cc_->EvalMult(ct, ct);
    auto out = add_constant(ct, 1.0, length);
    auto half_x2 = mul_constant(x2, 0.5);
    return cc_->EvalAdd(out, half_x2);
}

Ciphertext<DCRTPoly> CKKSRuntime::gelu_like_approx(const Ciphertext<DCRTPoly>& ct,
                                                   std::size_t length) const {
    auto x2 = cc_->EvalMult(ct, ct);
    auto x3 = cc_->EvalMult(x2, ct);
    auto half_x = mul_constant(ct, 0.5);
    auto eighth_x3 = mul_constant(x3, 0.125);
    auto out = add_constant(half_x, 0.5, length);
    return cc_->EvalAdd(out, eighth_x3);
}

Ciphertext<DCRTPoly> CKKSRuntime::layernorm_like_approx(const Ciphertext<DCRTPoly>& ct,
                                                        std::size_t length) const {
    auto x2 = cc_->EvalMult(ct, ct);
    auto neg_quarter_x2 = mul_constant(x2, -0.25);
    auto scaled_x = mul_constant(ct, 0.75);
    auto out = add_constant(scaled_x, 0.5, length);
    return cc_->EvalAdd(out, neg_quarter_x2);
}

Ciphertext<DCRTPoly> CKKSRuntime::apply_block_diagonal_he(
    const Ciphertext<DCRTPoly>& ct,
    const std::vector<std::vector<double>>& weights,
    std::size_t seq_len,
    std::size_t hidden) const {
    const std::size_t length = seq_len * hidden;
    Ciphertext<DCRTPoly> acc;
    bool initialized = false;

    for (int offset = -static_cast<int>(hidden) + 1;
         offset <= static_cast<int>(hidden) - 1;
         ++offset) {
        std::vector<double> mask(length, 0.0);
        bool has_nonzero = false;

        for (std::size_t token = 0; token < seq_len; ++token) {
            const std::size_t base = token * hidden;
            for (std::size_t out_col = 0; out_col < hidden; ++out_col) {
                const int in_col = static_cast<int>(out_col) - offset;
                if (in_col < 0 || in_col >= static_cast<int>(hidden)) {
                    continue;
                }
                const double coeff = weights[out_col][static_cast<std::size_t>(in_col)];
                if (coeff == 0.0) {
                    continue;
                }
                mask[base + out_col] = coeff;
                has_nonzero = true;
            }
        }

        if (!has_nonzero) {
            continue;
        }

        auto rotated = (offset == 0) ? ct : cc_->EvalRotate(ct, offset);
        Plaintext mask_ptxt = cc_->MakeCKKSPackedPlaintext(mask);
        auto term = cc_->EvalMult(rotated, mask_ptxt);
        if (!initialized) {
            acc = term;
            initialized = true;
        } else {
            acc = cc_->EvalAdd(acc, term);
        }
    }

    if (!initialized) {
        throw std::runtime_error("apply_block_diagonal_he() produced no terms");
    }

    return acc;
}

Ciphertext<DCRTPoly> CKKSRuntime::apply_block_diagonal_he_baseline(
    const Ciphertext<DCRTPoly>& ct,
    const std::vector<std::vector<double>>& weights,
    std::size_t seq_len,
    std::size_t hidden) const {
    const std::size_t length = seq_len * hidden;
    Ciphertext<DCRTPoly> acc;
    bool initialized = false;

    for (std::size_t token = 0; token < seq_len; ++token) {
        for (std::size_t out_col = 0; out_col < hidden; ++out_col) {
            for (std::size_t in_col = 0; in_col < hidden; ++in_col) {
                const double coeff = weights[out_col][in_col];
                if (coeff == 0.0) {
                    continue;
                }

                std::vector<double> mask(length, 0.0);
                mask[token * hidden + out_col] = coeff;

                const int shift = static_cast<int>(out_col) - static_cast<int>(in_col);
                auto rotated = (shift == 0) ? ct : cc_->EvalRotate(ct, shift);
                Plaintext mask_ptxt = cc_->MakeCKKSPackedPlaintext(mask);
                auto term = cc_->EvalMult(rotated, mask_ptxt);

                if (!initialized) {
                    acc = term;
                    initialized = true;
                } else {
                    acc = cc_->EvalAdd(acc, term);
                }
            }
        }
    }

    if (!initialized) {
        throw std::runtime_error("apply_block_diagonal_he_baseline() produced no terms");
    }

    return acc;
}

Ciphertext<DCRTPoly> CKKSRuntime::apply_sequence_mix_he(
    const Ciphertext<DCRTPoly>& ct,
    const std::vector<std::vector<double>>& weights,
    std::size_t seq_len,
    std::size_t hidden) const {
    const std::size_t length = seq_len * hidden;
    Ciphertext<DCRTPoly> acc;
    bool initialized = false;

    for (int token_offset = -static_cast<int>(seq_len) + 1;
         token_offset <= static_cast<int>(seq_len) - 1;
         ++token_offset) {
        std::vector<double> mask(length, 0.0);
        bool has_nonzero = false;

        for (std::size_t token_out = 0; token_out < seq_len; ++token_out) {
            const int token_in = static_cast<int>(token_out) - token_offset;
            if (token_in < 0 || token_in >= static_cast<int>(seq_len)) {
                continue;
            }
            const double coeff = weights[token_out][static_cast<std::size_t>(token_in)];
            if (coeff == 0.0) {
                continue;
            }
            for (std::size_t channel = 0; channel < hidden; ++channel) {
                mask[token_out * hidden + channel] = coeff;
            }
            has_nonzero = true;
        }

        if (!has_nonzero) {
            continue;
        }

        const int shift = token_offset * static_cast<int>(hidden);
        auto rotated = (shift == 0) ? ct : cc_->EvalRotate(ct, shift);
        Plaintext mask_ptxt = cc_->MakeCKKSPackedPlaintext(mask);
        auto term = cc_->EvalMult(rotated, mask_ptxt);
        if (!initialized) {
            acc = term;
            initialized = true;
        } else {
            acc = cc_->EvalAdd(acc, term);
        }
    }

    if (!initialized) {
        throw std::runtime_error("apply_sequence_mix_he() produced no terms");
    }

    return acc;
}

Ciphertext<DCRTPoly> CKKSRuntime::reduce_sum_he(const Ciphertext<DCRTPoly>& ct,
                                                std::size_t length,
                                                bool optimized) const {
    Ciphertext<DCRTPoly> acc = ct;
    if (optimized) {
        for (std::size_t shift = 1; shift < length; shift <<= 1) {
            acc = cc_->EvalAdd(acc, cc_->EvalRotate(acc, static_cast<int>(shift)));
        }
        return acc;
    }

    for (std::size_t shift = 1; shift < length; ++shift) {
        acc = cc_->EvalAdd(acc, cc_->EvalRotate(ct, static_cast<int>(shift)));
    }
    return acc;
}

Ciphertext<DCRTPoly> CKKSRuntime::max_pairwise_he(const Ciphertext<DCRTPoly>& lhs,
                                                  const Ciphertext<DCRTPoly>& rhs) const {
    auto diff = cc_->EvalSub(lhs, rhs);
    auto abs_like = cc_->EvalMult(diff, diff);
    auto sum = cc_->EvalAdd(lhs, rhs);
    auto sum_plus_abs = cc_->EvalAdd(sum, abs_like);
    return mul_constant(sum_plus_abs, 0.5);
}

Ciphertext<DCRTPoly> CKKSRuntime::tiny_bert_block_he_core(
    const Ciphertext<DCRTPoly>& input_ct,
    std::size_t length) const {
    auto normalized_input = mul_constant(input_ct, 0.1);

    auto q = apply_block_diagonal_he(
        normalized_input, tiny_bert_wq(), kTinyBertSeqLen, kTinyBertHidden);
    auto k = apply_block_diagonal_he(
        normalized_input, tiny_bert_wk(), kTinyBertSeqLen, kTinyBertHidden);
    auto v = apply_block_diagonal_he(
        normalized_input, tiny_bert_wv(), kTinyBertSeqLen, kTinyBertHidden);

    auto gates = softmax_like_approx(cc_->EvalMult(q, k), length);
    auto gated_v = cc_->EvalMult(gates, v);
    auto mixed = apply_sequence_mix_he(
        gated_v, tiny_bert_attention_mix(), kTinyBertSeqLen, kTinyBertHidden);
    auto attn_out = apply_block_diagonal_he(
        mixed, tiny_bert_wo(), kTinyBertSeqLen, kTinyBertHidden);
    auto attn_residual = cc_->EvalAdd(attn_out, normalized_input);

    auto ln1_out = layernorm_like_approx(attn_residual, length);
    auto mlp_fc1 = apply_block_diagonal_he(
        ln1_out, tiny_bert_w1(), kTinyBertSeqLen, kTinyBertHidden);
    auto mlp_act = gelu_like_approx(mlp_fc1, length);
    auto mlp_fc2 = apply_block_diagonal_he(
        mlp_act, tiny_bert_w2(), kTinyBertSeqLen, kTinyBertHidden);
    auto mlp_residual = cc_->EvalAdd(mlp_fc2, attn_residual);

    return layernorm_like_approx(mlp_residual, length);
}

std::vector<double> CKKSRuntime::encrypt_decrypt(const std::vector<double>& x) {
    require_ready("encrypt_decrypt()");
    auto ct = encrypt_only(x);
    return decrypt_only(ct, x.size());
}

std::vector<double> CKKSRuntime::add_plain(const std::vector<double>& x, double c) {
    require_ready("add_plain()");

    auto ct = encrypt_only(x);

    std::vector<double> cvec(x.size(), c);
    Plaintext cptxt = cc_->MakeCKKSPackedPlaintext(cvec);

    auto out = cc_->EvalAdd(ct, cptxt);
    return decrypt_to_vector(out, x.size());
}

std::vector<double> CKKSRuntime::mul_plain(const std::vector<double>& x, double c) {
    require_ready("mul_plain()");

    auto ct = encrypt_only(x);
    auto out = cc_->EvalMult(ct, c);
    return decrypt_to_vector(out, x.size());
}

std::vector<double> CKKSRuntime::add_ct_ct(const std::vector<double>& x,
                                           const std::vector<double>& y) {
    require_ready("add_ct_ct()");

    if (x.size() != y.size()) {
        throw std::runtime_error("add_ct_ct() requires x and y to have same length");
    }

    auto ctx = encrypt_only(x);
    auto cty = encrypt_only(y);
    auto out = cc_->EvalAdd(ctx, cty);
    return decrypt_to_vector(out, x.size());
}

std::vector<double> CKKSRuntime::mul_ct_ct(const std::vector<double>& x,
                                           const std::vector<double>& y) {
    require_ready("mul_ct_ct()");

    if (x.size() != y.size()) {
        throw std::runtime_error("mul_ct_ct() requires x and y to have same length");
    }

    auto ctx = encrypt_only(x);
    auto cty = encrypt_only(y);
    auto out = cc_->EvalMult(ctx, cty);
    return decrypt_to_vector(out, x.size());
}

std::vector<double> CKKSRuntime::rotate(const std::vector<double>& x, int steps) {
    require_ready("rotate()");

    auto ct = encrypt_only(x);
    auto out = cc_->EvalRotate(ct, steps);
    return decrypt_to_vector(out, x.size());
}

std::vector<double> CKKSRuntime::toy_transformer_block_plain(const std::vector<double>& x) {
    return ::toy_transformer_block_plain(x);
}

std::vector<double> CKKSRuntime::toy_transformer_block(const std::vector<double>& x) {
    require_ready("toy_transformer_block()");

    auto input_ct = encrypt_only(x);
    auto normalized_input = mul_constant(input_ct, 0.1);

    auto q = mul_constant(normalized_input, 1.10);
    auto k = mul_constant(normalized_input, 0.90);
    auto v = mul_constant(normalized_input, 1.30);

    auto attn_scores = cc_->EvalMult(q, k);
    auto attn_scale = mul_constant(attn_scores, 0.25);
    auto attn_weights = softmax_like_approx(attn_scale, x.size());
    auto attn_values = cc_->EvalMult(attn_weights, v);
    auto attn_out = mul_constant(attn_values, 0.80);
    auto attn_residual = cc_->EvalAdd(attn_out, normalized_input);

    auto ln1_out = layernorm_like_approx(attn_residual, x.size());

    auto mlp_fc1 = mul_constant(ln1_out, 1.70);
    auto mlp_act = gelu_like_approx(mlp_fc1, x.size());
    auto mlp_fc2 = mul_constant(mlp_act, 0.60);
    auto mlp_residual = cc_->EvalAdd(mlp_fc2, attn_residual);

    auto out = layernorm_like_approx(mlp_residual, x.size());
    return decrypt_to_vector(out, x.size());
}

std::vector<double> CKKSRuntime::tiny_bert_block_plain(const std::vector<double>& x) {
    return ::tiny_bert_block_plain(x);
}

std::vector<double> CKKSRuntime::tiny_bert_block(const std::vector<double>& x) {
    require_ready("tiny_bert_block()");
    require_tiny_bert_shape(x.size());

    auto input_ct = encrypt_only(x);
    auto out = tiny_bert_block_he_core(input_ct, x.size());
    return decrypt_to_vector(out, x.size());
}

std::vector<double> CKKSRuntime::bench_layernorm(const std::vector<double>& x, bool optimized) {
    require_ready("bench_layernorm()");

    auto input_ct = encrypt_only(x);
    auto sum = reduce_sum_he(input_ct, x.size(), optimized);
    auto mean = mul_constant(sum, 1.0 / static_cast<double>(x.size()));
    auto centered = cc_->EvalSub(input_ct, mean);
    auto centered_sq = cc_->EvalMult(centered, centered);
    auto var_sum = reduce_sum_he(centered_sq, x.size(), optimized);
    auto var = mul_constant(var_sum, 1.0 / static_cast<double>(x.size()));
    auto var2 = cc_->EvalMult(var, var);
    auto inv_std = add_constant(
        cc_->EvalAdd(mul_constant(var, -0.5), mul_constant(var2, 0.375)),
        1.0,
        x.size());
    auto out = cc_->EvalMult(centered, inv_std);
    return decrypt_to_vector(out, x.size());
}

std::vector<double> CKKSRuntime::bench_gelu(const std::vector<double>& x) {
    require_ready("bench_gelu()");

    auto input_ct = encrypt_only(x);
    auto out = gelu_like_approx(input_ct, x.size());
    return decrypt_to_vector(out, x.size());
}

std::vector<double> CKKSRuntime::bench_softmax(const std::vector<double>& x) {
    require_ready("bench_softmax()");

    auto input_ct = encrypt_only(x);
    auto out = softmax_like_approx(input_ct, x.size());
    return decrypt_to_vector(out, x.size());
}

std::vector<double> CKKSRuntime::bench_matmul(const std::vector<double>& x, bool optimized) {
    require_ready("bench_matmul()");

    if (x.empty() || x.size() % kTinyBertHidden != 0) {
        throw std::runtime_error("bench_matmul() requires n to be a positive multiple of 4");
    }

    const std::size_t seq_len = x.size() / kTinyBertHidden;
    auto input_ct = encrypt_only(x);
    auto out = optimized
        ? apply_block_diagonal_he(input_ct, bench_matmul_weights(), seq_len, kTinyBertHidden)
        : apply_block_diagonal_he_baseline(input_ct, bench_matmul_weights(), seq_len, kTinyBertHidden);
    return decrypt_to_vector(out, x.size());
}

std::vector<double> CKKSRuntime::bench_matmul_custom(
    const std::vector<double>& x,
    const std::vector<std::vector<double>>& weights,
    bool optimized) {
    require_ready("bench_matmul_custom()");

    if (weights.empty() || weights.size() != weights[0].size()) {
        throw std::runtime_error("bench_matmul_custom() requires a non-empty square weight matrix");
    }

    const std::size_t hidden = weights.size();
    for (const auto& row : weights) {
        if (row.size() != hidden) {
            throw std::runtime_error("bench_matmul_custom() requires a square weight matrix");
        }
    }

    if (x.empty() || x.size() % hidden != 0) {
        throw std::runtime_error("bench_matmul_custom() requires input length to be a positive multiple of the weight size");
    }

    const std::size_t seq_len = x.size() / hidden;
    auto input_ct = encrypt_only(x);
    auto out = optimized
        ? apply_block_diagonal_he(input_ct, weights, seq_len, hidden)
        : apply_block_diagonal_he_baseline(input_ct, weights, seq_len, hidden);
    return decrypt_to_vector(out, x.size());
}

std::vector<double> CKKSRuntime::bench_argmax(const std::vector<double>& x, bool optimized) {
    require_ready("bench_argmax()");

    auto input_ct = encrypt_only(x);
    Ciphertext<DCRTPoly> acc = input_ct;

    if (optimized) {
        for (std::size_t shift = 1; shift < x.size(); shift <<= 1) {
            auto rotated = cc_->EvalRotate(acc, static_cast<int>(shift));
            acc = max_pairwise_he(acc, rotated);
        }
    } else {
        for (std::size_t shift = 1; shift < x.size(); ++shift) {
            auto rotated = cc_->EvalRotate(input_ct, static_cast<int>(shift));
            acc = max_pairwise_he(acc, rotated);
        }
    }

    return decrypt_to_vector(acc, x.size());
}

std::vector<double> CKKSRuntime::tiny_text_classifier_plain(const std::vector<int>& token_ids) {
    return ::tiny_text_classifier_plain(token_ids);
}

std::vector<double> CKKSRuntime::tiny_text_classifier(const std::vector<int>& token_ids) {
    require_ready("tiny_text_classifier()");

    auto encoded = encode_tiny_text_tokens_plain(token_ids);
    auto input_ct = encrypt_only(encoded);
    auto hidden = tiny_bert_block_he_core(input_ct, encoded.size());
    auto pooled = apply_sequence_mix_he(
        hidden, tiny_text_avg_pool(), kTinyBertSeqLen, kTinyBertHidden);
    auto logits_repeated = apply_block_diagonal_he(
        pooled, tiny_text_classifier_head(), kTinyBertSeqLen, kTinyBertHidden);
    auto logits = decrypt_to_vector(logits_repeated, encoded.size());
    return {logits[0], logits[1]};
}

void CKKSRuntime::trace_tiny_text_classifier(const std::vector<int>& token_ids) {
    require_ready("trace_tiny_text_classifier()");

    auto print_stage = [&](const std::string& label, const std::vector<double>& values) {
        std::cout << label << ": [";
        for (std::size_t i = 0; i < values.size(); ++i) {
            std::cout << std::fixed << std::setprecision(6) << values[i];
            if (i + 1 < values.size()) {
                std::cout << ", ";
            }
        }
        std::cout << "]\n";
    };
    auto print_stage_error = [&](const std::string& label,
                                 const std::vector<double>& plain,
                                 const std::vector<double>& he) {
        const StageErrorStats stats = compute_stage_error_stats(plain, he);
        std::cout << label << "_error"
                  << ": max_abs=" << std::fixed << std::setprecision(6) << stats.max_abs_error
                  << ", mean_abs=" << stats.mean_abs_error
                  << ", rmse=" << stats.rmse << "\n";
    };

    auto encoded = encode_tiny_text_tokens_plain(token_ids);
    auto hidden_plain = tiny_bert_block_plain(encoded);
    auto pooled_plain = apply_sequence_mix_plain(
        hidden_plain, tiny_text_avg_pool(), kTinyBertSeqLen, kTinyBertHidden);
    auto logits_plain_repeated = apply_block_diagonal_plain(
        pooled_plain, tiny_text_classifier_head(), kTinyBertSeqLen, kTinyBertHidden);
    std::vector<double> logits_plain = {
        logits_plain_repeated[0],
        logits_plain_repeated[1],
    };

    std::cout << "=== Tiny Text Trace: Plaintext ===\n";
    print_stage("encoded_input", encoded);
    print_stage("hidden_out", hidden_plain);
    print_stage("pooled_hidden", pooled_plain);
    print_stage("logits_full", logits_plain_repeated);
    print_stage("logits", logits_plain);

    std::cout << "\n=== Tiny Text Trace: HE Decrypted Stages ===\n";
    auto input_ct = encrypt_only(encoded);
    auto hidden_he = tiny_bert_block_he_core(input_ct, encoded.size());
    auto pooled_he = apply_sequence_mix_he(
        hidden_he, tiny_text_avg_pool(), kTinyBertSeqLen, kTinyBertHidden);
    auto logits_he_repeated = apply_block_diagonal_he(
        pooled_he, tiny_text_classifier_head(), kTinyBertSeqLen, kTinyBertHidden);
    auto logits_he_full = decrypt_to_vector(logits_he_repeated, encoded.size());
    std::vector<double> logits_he = {logits_he_full[0], logits_he_full[1]};

    print_stage("encoded_input", encoded);
    auto hidden_he_values = decrypt_to_vector(hidden_he, encoded.size());
    auto pooled_he_values = decrypt_to_vector(pooled_he, encoded.size());
    print_stage("hidden_out", hidden_he_values);
    print_stage("pooled_hidden", pooled_he_values);
    print_stage("logits_full", logits_he_full);
    print_stage("logits", logits_he);
    print_stage_error("encoded_input", encoded, encoded);
    print_stage_error("hidden_out", hidden_plain, hidden_he_values);
    print_stage_error("pooled_hidden", pooled_plain, pooled_he_values);
    print_stage_error("logits_full", logits_plain_repeated, logits_he_full);
    print_stage_error("logits", logits_plain, logits_he);
}

void CKKSRuntime::trace_tiny_bert_block(const std::vector<double>& x) {
    require_ready("trace_tiny_bert_block()");
    require_tiny_bert_shape(x.size());

    auto print_stage = [&](const std::string& label, const std::vector<double>& values) {
        std::cout << label << ": [";
        for (std::size_t i = 0; i < values.size(); ++i) {
            std::cout << std::fixed << std::setprecision(6) << values[i];
            if (i + 1 < values.size()) {
                std::cout << ", ";
            }
        }
        std::cout << "]\n";
    };
    auto print_stage_error = [&](const std::string& label,
                                 const std::vector<double>& plain,
                                 const std::vector<double>& he) {
        const StageErrorStats stats = compute_stage_error_stats(plain, he);
        std::cout << label << "_error"
                  << ": max_abs=" << std::fixed << std::setprecision(6) << stats.max_abs_error
                  << ", mean_abs=" << stats.mean_abs_error
                  << ", rmse=" << stats.rmse << "\n";
    };

    std::cout << "=== Tiny BERT Trace: Plaintext ===\n";
    auto plain_normalized_input = mul_constant_plain(x, 0.1);
    auto plain_q = apply_block_diagonal_plain(
        plain_normalized_input, tiny_bert_wq(), kTinyBertSeqLen, kTinyBertHidden);
    auto plain_k = apply_block_diagonal_plain(
        plain_normalized_input, tiny_bert_wk(), kTinyBertSeqLen, kTinyBertHidden);
    auto plain_v = apply_block_diagonal_plain(
        plain_normalized_input, tiny_bert_wv(), kTinyBertSeqLen, kTinyBertHidden);
    auto plain_gates = softmax_like_approx_plain(mul_vec_plain(plain_q, plain_k));
    auto plain_gated_v = mul_vec_plain(plain_gates, plain_v);
    auto plain_mixed = apply_sequence_mix_plain(
        plain_gated_v, tiny_bert_attention_mix(), kTinyBertSeqLen, kTinyBertHidden);
    auto plain_attn_out = apply_block_diagonal_plain(
        plain_mixed, tiny_bert_wo(), kTinyBertSeqLen, kTinyBertHidden);
    auto plain_attn_residual = add_vec_plain(plain_attn_out, plain_normalized_input);
    auto plain_ln1_out = layernorm_like_approx_plain(plain_attn_residual);
    auto plain_mlp_fc1 = apply_block_diagonal_plain(
        plain_ln1_out, tiny_bert_w1(), kTinyBertSeqLen, kTinyBertHidden);
    auto plain_mlp_act = gelu_like_approx_plain(plain_mlp_fc1);
    auto plain_mlp_fc2 = apply_block_diagonal_plain(
        plain_mlp_act, tiny_bert_w2(), kTinyBertSeqLen, kTinyBertHidden);
    auto plain_mlp_residual = add_vec_plain(plain_mlp_fc2, plain_attn_residual);
    auto plain_out = layernorm_like_approx_plain(plain_mlp_residual);

    print_stage("input", x);
    print_stage("normalized_input", plain_normalized_input);
    print_stage("q_proj", plain_q);
    print_stage("k_proj", plain_k);
    print_stage("v_proj", plain_v);
    print_stage("attention_gates", plain_gates);
    print_stage("gated_values", plain_gated_v);
    print_stage("sequence_mixed", plain_mixed);
    print_stage("attention_out", plain_attn_out);
    print_stage("attention_residual", plain_attn_residual);
    print_stage("layernorm1_out", plain_ln1_out);
    print_stage("mlp_fc1", plain_mlp_fc1);
    print_stage("mlp_activation", plain_mlp_act);
    print_stage("mlp_fc2", plain_mlp_fc2);
    print_stage("mlp_residual", plain_mlp_residual);
    print_stage("final_out", plain_out);

    std::cout << "\n=== Tiny BERT Trace: HE Decrypted Stages ===\n";
    auto input_ct = encrypt_only(x);
    auto normalized_input = mul_constant(input_ct, 0.1);
    auto q = apply_block_diagonal_he(
        normalized_input, tiny_bert_wq(), kTinyBertSeqLen, kTinyBertHidden);
    auto k = apply_block_diagonal_he(
        normalized_input, tiny_bert_wk(), kTinyBertSeqLen, kTinyBertHidden);
    auto v = apply_block_diagonal_he(
        normalized_input, tiny_bert_wv(), kTinyBertSeqLen, kTinyBertHidden);
    auto gates = softmax_like_approx(cc_->EvalMult(q, k), x.size());
    auto gated_v = cc_->EvalMult(gates, v);
    auto mixed = apply_sequence_mix_he(
        gated_v, tiny_bert_attention_mix(), kTinyBertSeqLen, kTinyBertHidden);
    auto attn_out = apply_block_diagonal_he(
        mixed, tiny_bert_wo(), kTinyBertSeqLen, kTinyBertHidden);
    auto attn_residual = cc_->EvalAdd(attn_out, normalized_input);
    auto ln1_out = layernorm_like_approx(attn_residual, x.size());
    auto mlp_fc1 = apply_block_diagonal_he(
        ln1_out, tiny_bert_w1(), kTinyBertSeqLen, kTinyBertHidden);
    auto mlp_act = gelu_like_approx(mlp_fc1, x.size());
    auto mlp_fc2 = apply_block_diagonal_he(
        mlp_act, tiny_bert_w2(), kTinyBertSeqLen, kTinyBertHidden);
    auto mlp_residual = cc_->EvalAdd(mlp_fc2, attn_residual);
    auto out = layernorm_like_approx(mlp_residual, x.size());

    auto he_normalized_input = decrypt_to_vector(normalized_input, x.size());
    auto he_q = decrypt_to_vector(q, x.size());
    auto he_k = decrypt_to_vector(k, x.size());
    auto he_v = decrypt_to_vector(v, x.size());
    auto he_gates = decrypt_to_vector(gates, x.size());
    auto he_gated_v = decrypt_to_vector(gated_v, x.size());
    auto he_mixed = decrypt_to_vector(mixed, x.size());
    auto he_attn_out = decrypt_to_vector(attn_out, x.size());
    auto he_attn_residual = decrypt_to_vector(attn_residual, x.size());
    auto he_ln1_out = decrypt_to_vector(ln1_out, x.size());
    auto he_mlp_fc1 = decrypt_to_vector(mlp_fc1, x.size());
    auto he_mlp_act = decrypt_to_vector(mlp_act, x.size());
    auto he_mlp_fc2 = decrypt_to_vector(mlp_fc2, x.size());
    auto he_mlp_residual = decrypt_to_vector(mlp_residual, x.size());
    auto he_out = decrypt_to_vector(out, x.size());

    print_stage("normalized_input", he_normalized_input);
    print_stage("q_proj", he_q);
    print_stage("k_proj", he_k);
    print_stage("v_proj", he_v);
    print_stage("attention_gates", he_gates);
    print_stage("gated_values", he_gated_v);
    print_stage("sequence_mixed", he_mixed);
    print_stage("attention_out", he_attn_out);
    print_stage("attention_residual", he_attn_residual);
    print_stage("layernorm1_out", he_ln1_out);
    print_stage("mlp_fc1", he_mlp_fc1);
    print_stage("mlp_activation", he_mlp_act);
    print_stage("mlp_fc2", he_mlp_fc2);
    print_stage("mlp_residual", he_mlp_residual);
    print_stage("final_out", he_out);
    print_stage_error("normalized_input", plain_normalized_input, he_normalized_input);
    print_stage_error("q_proj", plain_q, he_q);
    print_stage_error("k_proj", plain_k, he_k);
    print_stage_error("v_proj", plain_v, he_v);
    print_stage_error("attention_gates", plain_gates, he_gates);
    print_stage_error("gated_values", plain_gated_v, he_gated_v);
    print_stage_error("sequence_mixed", plain_mixed, he_mixed);
    print_stage_error("attention_out", plain_attn_out, he_attn_out);
    print_stage_error("attention_residual", plain_attn_residual, he_attn_residual);
    print_stage_error("layernorm1_out", plain_ln1_out, he_ln1_out);
    print_stage_error("mlp_fc1", plain_mlp_fc1, he_mlp_fc1);
    print_stage_error("mlp_activation", plain_mlp_act, he_mlp_act);
    print_stage_error("mlp_fc2", plain_mlp_fc2, he_mlp_fc2);
    print_stage_error("mlp_residual", plain_mlp_residual, he_mlp_residual);
    print_stage_error("final_out", plain_out, he_out);
}

std::string CKKSRuntime::info() const {
    std::ostringstream oss;
    oss << "CKKSRuntime(initialized=" << (initialized_ ? "true" : "false")
        << ", keys_generated=" << (keys_generated_ ? "true" : "false");

    if (initialized_ && cc_) {
        oss << ", ring_dimension=" << cc_->GetRingDimension()
            << ", multDepth=" << multDepth_
            << ", scaleModSize=" << scaleModSize_
            << ", batchSize=" << batchSize_;
    }

    oss << ")";
    return oss.str();
}
