// parakeet_q1_0_smoke.cpp - Parakeet with pre-quantized Q1_0 encoder weights.
//
// parakeet_q1_0.gguf is the minimal Parakeet model (d_model = d_ff = 128)
// with every encoder linear and both conv-module pointwise kernels in Q1_0,
// as a converter of a natively binarized checkpoint writes it. The pointwise
// kernels are stored [in, out]: a block-quantized row cannot have the
// ne[0] == 1 of the F32/F16 conv layout [1, in, out].
// parakeet_q1_0_as_f32.gguf holds the same weights decoded to F32, in the
// usual layouts.
//
// Asserts, through the public load path:
//   1. both files load, and the Q1_0 tensors keep their type and layout
//      (no load-time promotion or reshape);
//   2. both transcribe a clip with the default pointwise dispatch and with
//      TRANSCRIBE_CONV_NO_DIRECT_PW=1, which must not send a Q1_0 kernel to
//      the im2col path.
// Numerical parity of the Q1_0 conv module lives in quant_q1_0_unit.

#include "arch/parakeet/parakeet.h"
#include "arch/parakeet/weights.h"
#include "ggml.h"
#include "transcribe-model.h"
#include "transcribe.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#ifndef TRANSCRIBE_TEST_FIXTURES_DIR
#    error "TRANSCRIBE_TEST_FIXTURES_DIR must be defined by the build system"
#endif

namespace {

int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

constexpr int64_t kDModel = 128;
constexpr int64_t kDFF    = 128;

void set_env(const char * key, const char * value) {
#if defined(_WIN32)
    _putenv_s(key, value);
#else
    setenv(key, value, 1);
#endif
}

void unset_env(const char * key) {
#if defined(_WIN32)
    _putenv_s(key, "");
#else
    unsetenv(key);
#endif
}

std::string fixture(const char * name) {
    return std::string(TRANSCRIBE_TEST_FIXTURES_DIR) + "/" + name;
}

bool has_ne(const ggml_tensor * t, int64_t ne0, int64_t ne1, int64_t ne2) {
    return t != nullptr && t->ne[0] == ne0 && t->ne[1] == ne1 && t->ne[2] == ne2 && t->ne[3] == 1;
}

void check_layout(const transcribe_model * model, bool q1) {
    const auto * pm = static_cast<const transcribe::parakeet::ParakeetModel *>(model);
    CHECK(pm->weights.blocks.size() == 2);
    for (const auto & b : pm->weights.blocks) {
        if (q1) {
            CHECK(b.conv_pw1_w->type == GGML_TYPE_Q1_0);
            CHECK(b.conv_pw2_w->type == GGML_TYPE_Q1_0);
            CHECK(has_ne(b.conv_pw1_w, kDModel, 2 * kDModel, 1));
            CHECK(has_ne(b.conv_pw2_w, kDModel, kDModel, 1));
            CHECK(b.ff1_lin1_w->type == GGML_TYPE_Q1_0);
            CHECK(b.ff2_lin2_w->type == GGML_TYPE_Q1_0);
            CHECK(has_ne(b.ff2_lin2_w, kDFF, kDModel, 1));
            CHECK(b.attn_pos_w->type == GGML_TYPE_Q1_0);
        } else {
            CHECK(b.conv_pw1_w->type == GGML_TYPE_F32);
            CHECK(b.conv_pw2_w->type == GGML_TYPE_F32);
            CHECK(has_ne(b.conv_pw1_w, 1, kDModel, 2 * kDModel));
            CHECK(has_ne(b.conv_pw2_w, 1, kDModel, kDModel));
            CHECK(b.ff1_lin1_w->type == GGML_TYPE_F32);
        }
    }
}

std::vector<float> make_clip() {
    // 1.5 s at 16 kHz: a rising tone over seeded noise.
    std::vector<float> pcm(24000);
    uint32_t           rng = 12345u;
    for (size_t i = 0; i < pcm.size(); ++i) {
        rng               = rng * 1664525u + 1013904223u;
        const float noise = static_cast<float>(rng >> 8) / static_cast<float>(1u << 24) - 0.5f;
        const float t     = static_cast<float>(i) / 16000.0f;
        const float tone  = std::sin(2.0f * 3.14159265f * (200.0f + 400.0f * t) * t);
        pcm[i]            = 0.3f * tone + 0.05f * noise;
    }
    return pcm;
}

void check_model(const char * name, bool q1, transcribe_backend_request backend, const std::vector<float> & pcm) {
    transcribe_model_load_params mp;
    transcribe_model_load_params_init(&mp);
    mp.backend                 = backend;
    transcribe_model *      m  = nullptr;
    const transcribe_status st = transcribe_model_load_file(fixture(name).c_str(), &mp, &m);
    CHECK(st == TRANSCRIBE_OK && m != nullptr);
    if (m == nullptr) {
        return;
    }
    check_layout(m, q1);

    const char * policies[2] = { nullptr, "1" };
    for (const char * no_direct : policies) {
        if (no_direct == nullptr) {
            unset_env("TRANSCRIBE_CONV_NO_DIRECT_PW");
        } else {
            set_env("TRANSCRIBE_CONV_NO_DIRECT_PW", no_direct);
        }
        transcribe_session * s = nullptr;
        CHECK(transcribe_session_init(m, nullptr, &s) == TRANSCRIBE_OK);
        if (s != nullptr) {
            CHECK(transcribe_run(s, pcm.data(), static_cast<int>(pcm.size()), nullptr) == TRANSCRIBE_OK);
            const char * text = transcribe_full_text(s);
            std::fprintf(stderr, "%s on %s, NO_DIRECT_PW=%s: \"%s\"\n", name, transcribe_model_backend(m),
                         no_direct == nullptr ? "unset" : no_direct, text == nullptr ? "" : text);
            transcribe_session_free(s);
        }
    }
    unset_env("TRANSCRIBE_CONV_NO_DIRECT_PW");
    transcribe_model_free(m);
}

}  // namespace

int main() {
    const std::vector<float> pcm = make_clip();

    const transcribe_backend_request backends[2] = { TRANSCRIBE_BACKEND_CPU, TRANSCRIBE_BACKEND_AUTO };
    for (transcribe_backend_request backend : backends) {
        check_model("parakeet_q1_0.gguf", true, backend, pcm);
        check_model("parakeet_q1_0_as_f32.gguf", false, backend, pcm);
    }

    if (g_failures > 0) {
        std::fprintf(stderr, "\n%d assertion(s) FAILED\n", g_failures);
    }
    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
