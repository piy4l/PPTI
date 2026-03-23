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

std::vector<double> CKKSRuntime::encrypt_decrypt(const std::vector<double>& x) {
    require_ready("encrypt_decrypt()");

    Plaintext ptxt = cc_->MakeCKKSPackedPlaintext(x);
    auto ct = cc_->Encrypt(keys_.publicKey, ptxt);
    return decrypt_to_vector(ct, x.size());
}

std::vector<double> CKKSRuntime::add_plain(const std::vector<double>& x, double c) {
    require_ready("add_plain()");

    Plaintext ptxt = cc_->MakeCKKSPackedPlaintext(x);
    auto ct = cc_->Encrypt(keys_.publicKey, ptxt);

    std::vector<double> cvec(x.size(), c);
    Plaintext cptxt = cc_->MakeCKKSPackedPlaintext(cvec);

    auto out = cc_->EvalAdd(ct, cptxt);
    return decrypt_to_vector(out, x.size());
}

std::vector<double> CKKSRuntime::mul_plain(const std::vector<double>& x, double c) {
    require_ready("mul_plain()");

    Plaintext ptxt = cc_->MakeCKKSPackedPlaintext(x);
    auto ct = cc_->Encrypt(keys_.publicKey, ptxt);

    auto out = cc_->EvalMult(ct, c);
    return decrypt_to_vector(out, x.size());
}

std::vector<double> CKKSRuntime::rotate(const std::vector<double>& x, int steps) {
    require_ready("rotate()");

    Plaintext ptxt = cc_->MakeCKKSPackedPlaintext(x);
    auto ct = cc_->Encrypt(keys_.publicKey, ptxt);

    auto out = cc_->EvalRotate(ct, steps);
    return decrypt_to_vector(out, x.size());
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