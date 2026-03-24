#include "ckks_runner.h"

#include <sstream>
#include <stdexcept>

using namespace lbcrypto;

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

void CKKSRuntime::require_initialized(const std::string& fn) const {
    if (!initialized_) {
        throw std::runtime_error(fn + " requires init()");
    }
}

void CKKSRuntime::require_ready(const std::string& fn) const {
    if (!initialized_ || !keys_generated_) {
        throw std::runtime_error(fn + " requires init() and keygen()");
    }
}

CKKSRuntime::Plaintext CKKSRuntime::encode(const std::vector<double>& x) const {
    require_initialized("encode()");
    return cc_->MakeCKKSPackedPlaintext(x);
}

CKKSRuntime::Ciphertext CKKSRuntime::encrypt(const Plaintext& plaintext) const {
    require_ready("encrypt()");
    return cc_->Encrypt(keys_.publicKey, plaintext);
}

std::vector<double> CKKSRuntime::decrypt_and_decode(const Ciphertext& ciphertext,
                                                    std::size_t length) const {
    require_ready("decrypt_and_decode()");
    return decrypt_to_vector(ciphertext, length);
}

CKKSRuntime::Ciphertext CKKSRuntime::add(const Ciphertext& lhs, const Ciphertext& rhs) const {
    require_ready("add()");
    return cc_->EvalAdd(lhs, rhs);
}

CKKSRuntime::Ciphertext CKKSRuntime::multiply(const Ciphertext& lhs,
                                              const Ciphertext& rhs) const {
    require_ready("multiply()");
    return cc_->EvalMult(lhs, rhs);
}

CKKSRuntime::Ciphertext CKKSRuntime::multiply_plain(const Ciphertext& lhs, double scalar) const {
    require_ready("multiply_plain()");
    return cc_->EvalMult(lhs, scalar);
}

CKKSRuntime::Ciphertext CKKSRuntime::multiply_plaintext(const Ciphertext& lhs,
                                                       const std::vector<double>& rhs) const {
    require_ready("multiply_plaintext()");
    auto rhs_plaintext = encode(rhs);
    return cc_->EvalMult(lhs, rhs_plaintext);
}

CKKSRuntime::Ciphertext CKKSRuntime::rotate(const Ciphertext& ciphertext, int steps) const {
    require_ready("rotate()");
    return cc_->EvalRotate(ciphertext, steps);
}

std::vector<double> CKKSRuntime::decrypt_to_vector(const Ciphertext& ct,
                                                   std::size_t length) const {
    Plaintext decrypted;
    cc_->Decrypt(keys_.secretKey, ct, &decrypted);
    decrypted->SetLength(length);
    return decrypted->GetRealPackedValue();
}

std::vector<double> CKKSRuntime::encrypt_decrypt(const std::vector<double>& x) const {
    auto ptxt = encode(x);
    auto ct = encrypt(ptxt);
    return decrypt_and_decode(ct, x.size());
}

std::vector<double> CKKSRuntime::add_plain(const std::vector<double>& x, double c) const {
    auto ptxt = encode(x);
    auto ct = encrypt(ptxt);
    std::vector<double> cvec(x.size(), c);
    auto cptxt = encode(cvec);
    auto cct = encrypt(cptxt);
    auto out = add(ct, cct);
    return decrypt_and_decode(out, x.size());
}

std::vector<double> CKKSRuntime::mul_plain(const std::vector<double>& x, double c) const {
    auto ptxt = encode(x);
    auto ct = encrypt(ptxt);
    auto out = multiply_plain(ct, c);
    return decrypt_and_decode(out, x.size());
}

std::vector<double> CKKSRuntime::rotate_plain(const std::vector<double>& x, int steps) const {
    auto ptxt = encode(x);
    auto ct = encrypt(ptxt);
    auto out = rotate(ct, steps);
    return decrypt_and_decode(out, x.size());
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
