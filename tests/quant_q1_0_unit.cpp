// quant_q1_0_unit.cpp - contract test for pre-quantized Q1_0 weights.
//
// Q1_0 stores one sign bit per weight and one F16 scale per 128-weight block:
// bit j of a block is (qs[j / 8] >> (j % 8)) & 1, and a set bit means +d, a
// clear bit -d. Converters of natively binarized checkpoints write that layout
// directly, so this test packs it by hand rather than through ggml's
// quantizer, and checks the loader side of the contract:
//
//   - a Q1_0 matrix in a GGUF passes the shared linear allowlist
//     (TRANSCRIBE_QUANT_LINEAR_TYPES, which every family's GET_LIN uses),
//   - it loads through load_common::load_weights, and
//   - ggml_mul_mat on it matches the same product over the +-d matrix it
//     encodes, for a single column (mat-vec kernels) and for many, and
//   - conformer::conv_module with Q1_0 pointwise kernels, stored in the linear
//     layout [in, out], matches the module over F32 kernels holding the same
//     +-d values in the conv layout [1, in, out], under either pointwise
//     policy (direct mul_mat, or the im2col fallback that cannot take a
//     quantized kernel).
//
// Runs on the CPU, and also on the default accelerator (Metal, Vulkan, ...)
// when the build has one.

#include "conformer/conformer.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "transcribe-backend.h"
#include "transcribe-load-common.h"
#include "transcribe-path.h"
#include "transcribe-weights-util.h"
#include "transcribe.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

constexpr const char * kTag    = "q1_0-test";
constexpr const char * kWeight = "enc.blocks.0.ff1.linear1.weight";
constexpr int64_t      kQK     = 128;  // weights per Q1_0 block
constexpr int64_t      kIn     = 512;  // ne[0]: 4 blocks per row
constexpr int64_t      kOut    = 64;   // ne[1]

struct BlockQ1 {
    ggml_fp16_t d;
    uint8_t     qs[kQK / 8];
};

static_assert(sizeof(BlockQ1) == 18, "Q1_0 block is an F16 scale plus 16 sign bytes");

uint32_t g_rng = 0x2545f491u;

float next_unit() {
    g_rng = g_rng * 1664525u + 1013904223u;
    return static_cast<float>(g_rng >> 8) / static_cast<float>(1u << 24) * 2.0f - 1.0f;
}

// A Q1_0 matrix packed by hand, plus the +-d values it encodes.
struct Q1Matrix {
    std::vector<BlockQ1> blocks;
    std::vector<float>   values;  // row-major [n_out][n_in]
};

// n_in must be a multiple of kQK. Scales are around `scale` and differ per
// block, so a scale applied to the wrong block shows.
Q1Matrix make_q1_matrix(int64_t n_out, int64_t n_in, float scale) {
    Q1Matrix m;
    m.blocks.resize(static_cast<size_t>(n_out * n_in / kQK));
    m.values.resize(static_cast<size_t>(n_out * n_in));
    for (size_t b = 0; b < m.blocks.size(); ++b) {
        BlockQ1 & blk = m.blocks[b];
        blk.d         = ggml_fp32_to_fp16(scale * (0.5f + 0.1f * static_cast<float>(b % 11)));
        const float d = ggml_fp16_to_fp32(blk.d);
        std::memset(blk.qs, 0, sizeof(blk.qs));
        for (int64_t j = 0; j < kQK; ++j) {
            const bool positive = next_unit() > 0.0f;
            if (positive) {
                blk.qs[j / 8] |= static_cast<uint8_t>(1u << (j % 8));
            }
            m.values[b * kQK + static_cast<size_t>(j)] = positive ? d : -d;
        }
    }
    return m;
}

std::string temp_gguf_path() {
    std::error_code ec;
    auto            dir = std::filesystem::temp_directory_path(ec);
    if (ec) {
        dir = std::filesystem::path(".");
    }
    // Random per run, so concurrent runs (two build trees under ctest) do not
    // share, truncate or delete each other's file. UTF-8, which is what gguf
    // and load_common expect for paths on every platform.
    std::random_device rd;
    const auto         tick = std::chrono::steady_clock::now().time_since_epoch().count();
    return (dir / ("transcribe_quant_q1_0_" + std::to_string(rd()) + "_" + std::to_string(tick) + ".gguf")).u8string();
}

bool write_gguf(const std::string & path, const Q1Matrix & m) {
    ggml_init_params ip  = { ggml_tensor_overhead() + m.blocks.size() * sizeof(BlockQ1) + 1024, nullptr, false };
    ggml_context *   ctx = ggml_init(ip);
    if (ctx == nullptr) {
        return false;
    }
    ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_Q1_0, kIn, kOut);
    ggml_set_name(t, kWeight);
    if (ggml_nbytes(t) != m.blocks.size() * sizeof(BlockQ1)) {
        ggml_free(ctx);
        return false;
    }
    std::memcpy(t->data, m.blocks.data(), ggml_nbytes(t));

    gguf_context * g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "q1_0-test");
    gguf_add_tensor(g, t);
    const bool ok = gguf_write_to_file(g, path.c_str(), /*only_meta=*/false);
    gguf_free(g);
    ggml_free(ctx);
    return ok;
}

// out[c][r] = sum_k w[r][k] * x[c][k], plus the per-output magnitude
// sum_k |w[r][k] * x[c][k]| that bounds rounding error in any backend.
void reference_matmul(const std::vector<float> & w,
                      const std::vector<float> & x,
                      int64_t                    n_cols,
                      std::vector<double> &      out,
                      std::vector<double> &      mag) {
    out.assign(static_cast<size_t>(n_cols * kOut), 0.0);
    mag.assign(static_cast<size_t>(n_cols * kOut), 0.0);
    for (int64_t c = 0; c < n_cols; ++c) {
        for (int64_t r = 0; r < kOut; ++r) {
            double acc = 0.0, abs_acc = 0.0;
            for (int64_t k = 0; k < kIn; ++k) {
                const double p = static_cast<double>(w[r * kIn + k]) * static_cast<double>(x[c * kIn + k]);
                acc += p;
                abs_acc += std::fabs(p);
            }
            out[c * kOut + r] = acc;
            mag[c * kOut + r] = abs_acc;
        }
    }
}

struct BackendPlanName {
    transcribe::BackendPlan plan;
    const char *            label;
};

// Load the GGUF the way a family loader does, then multiply on plan.primary.
void check_backend(const std::string & path, const Q1Matrix & m, const BackendPlanName & bp) {
    using namespace transcribe;
    const BackendPlan & plan = bp.plan;

    ggml_context *   ctx_meta = nullptr;
    gguf_init_params gp       = { /*no_alloc=*/true, &ctx_meta };
    gguf_context *   gguf     = gguf_init_from_file(path.c_str(), gp);
    CHECK(gguf != nullptr && ctx_meta != nullptr);
    if (gguf == nullptr || ctx_meta == nullptr) {
        return;
    }

    // The loader gate: a Q1_0 matrix must pass the shared linear allowlist.
    ggml_tensor * w =
        transcribe::weights::find_tensor(ctx_meta, kWeight, { TRANSCRIBE_QUANT_LINEAR_TYPES }, { kIn, kOut }, kTag);
    CHECK(w != nullptr);
    if (w == nullptr) {
        gguf_free(gguf);
        ggml_free(ctx_meta);
        return;
    }
    CHECK(w->type == GGML_TYPE_Q1_0);

    load_common::WeightStorage storage;
    const transcribe_status    st = load_common::load_weights(path, gguf, ctx_meta, plan, nullptr, kTag, storage);
    CHECK(st == TRANSCRIBE_OK);
    if (st != TRANSCRIBE_OK) {
        storage.reset();
        gguf_free(gguf);
        ggml_free(ctx_meta);
        return;
    }

    // One column exercises the mat-vec kernels, many the mat-mat ones.
    const int64_t      cols[2] = { 1, 19 };
    ggml_init_params   cp      = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context *     ctx     = ggml_init(cp);
    ggml_tensor *      x[2]    = {};
    ggml_tensor *      y[2]    = {};
    ggml_cgraph *      gf      = ggml_new_graph(ctx);
    std::vector<float> xs[2];
    for (int i = 0; i < 2; ++i) {
        x[i] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kIn, cols[i]);
        ggml_set_input(x[i]);
        y[i] = ggml_mul_mat(ctx, w, x[i]);
        ggml_set_output(y[i]);
        ggml_build_forward_expand(gf, y[i]);
        xs[i].resize(static_cast<size_t>(kIn * cols[i]));
        for (float & v : xs[i]) {
            v = next_unit();
        }
    }

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(plan.primary));
    CHECK(ggml_gallocr_alloc_graph(galloc, gf));
    for (int i = 0; i < 2; ++i) {
        ggml_backend_tensor_set(x[i], xs[i].data(), 0, ggml_nbytes(x[i]));
    }
    CHECK(ggml_backend_graph_compute(plan.primary, gf) == GGML_STATUS_SUCCESS);

    for (int i = 0; i < 2; ++i) {
        std::vector<float> got(static_cast<size_t>(kOut * cols[i]));
        ggml_backend_tensor_get(y[i], got.data(), 0, ggml_nbytes(y[i]));
        std::vector<double> want, mag;
        reference_matmul(m.values, xs[i], cols[i], want, mag);
        // Backends may round activations to 8 bits before the dot product
        // (CPU: Q8_0); that error is bounded by ~1/127 of sum |w x|. A wrong
        // sign bit, bit order or block scale costs far more than 1% of it.
        double worst = 0.0;
        for (size_t j = 0; j < got.size(); ++j) {
            if (!std::isfinite(got[j])) {
                worst = INFINITY;
                break;
            }
            const double rel = std::fabs(static_cast<double>(got[j]) - want[j]) / (mag[j] + 1e-12);
            worst            = rel > worst ? rel : worst;
        }
        std::fprintf(stderr, "%s: Q1_0 mul_mat %lldx%lld x %lld cols: worst |err| / sum|w x| = %.2e\n", bp.label,
                     static_cast<long long>(kOut), static_cast<long long>(kIn), static_cast<long long>(cols[i]), worst);
        CHECK(worst < 1e-2);
    }

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    storage.reset();
    gguf_free(gguf);
    ggml_free(ctx_meta);
}

// ||a - b|| / ||b|| over two equally sized vectors.
double rel_l2(const std::vector<float> & a, const std::vector<float> & b) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        num += diff * diff;
        den += static_cast<double>(b[i]) * static_cast<double>(b[i]);
    }
    return std::sqrt(num / (den + 1e-30));
}

void check_conv_module(const BackendPlanName & bp) {
    namespace conf      = transcribe::conformer;
    constexpr int64_t d = 128, T = 23, k = 3;

    const Q1Matrix     pw1 = make_q1_matrix(2 * d, d, 0.05f);
    const Q1Matrix     pw2 = make_q1_matrix(d, d, 0.05f);
    std::vector<float> dw(k * d), bn_scale(d), bn_bias(d), pw1_b(2 * d), pw2_b(d), xs(d * T);
    for (float & v : dw) {
        v = 0.3f * next_unit();
    }
    for (float & v : bn_scale) {
        v = 1.0f + 0.2f * next_unit();
    }
    for (float & v : bn_bias) {
        v = 0.1f * next_unit();
    }
    for (float & v : pw1_b) {
        v = 0.1f * next_unit();
    }
    for (float & v : pw2_b) {
        v = 0.1f * next_unit();
    }
    for (float & v : xs) {
        v = next_unit();
    }

    ggml_init_params wp   = { 16 * ggml_tensor_overhead(), nullptr, true };
    ggml_context *   wctx = ggml_init(wp);
    ggml_tensor *    q1_1 = ggml_new_tensor_2d(wctx, GGML_TYPE_Q1_0, d, 2 * d);
    ggml_tensor *    q1_2 = ggml_new_tensor_2d(wctx, GGML_TYPE_Q1_0, d, d);
    ggml_tensor *    f_1  = ggml_new_tensor_3d(wctx, GGML_TYPE_F32, 1, d, 2 * d);
    ggml_tensor *    f_2  = ggml_new_tensor_3d(wctx, GGML_TYPE_F32, 1, d, d);
    ggml_tensor *    t_dw = ggml_new_tensor_3d(wctx, GGML_TYPE_F32, k, 1, d);
    ggml_tensor *    t_bs = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, d);
    ggml_tensor *    t_bb = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, d);
    ggml_tensor *    t_b1 = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, 2 * d);
    ggml_tensor *    t_b2 = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, d);

    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, bp.plan.primary);
    CHECK(wbuf != nullptr);
    if (wbuf == nullptr) {
        ggml_free(wctx);
        return;
    }
    ggml_backend_tensor_set(q1_1, pw1.blocks.data(), 0, ggml_nbytes(q1_1));
    ggml_backend_tensor_set(q1_2, pw2.blocks.data(), 0, ggml_nbytes(q1_2));
    ggml_backend_tensor_set(f_1, pw1.values.data(), 0, ggml_nbytes(f_1));
    ggml_backend_tensor_set(f_2, pw2.values.data(), 0, ggml_nbytes(f_2));
    ggml_backend_tensor_set(t_dw, dw.data(), 0, ggml_nbytes(t_dw));
    ggml_backend_tensor_set(t_bs, bn_scale.data(), 0, ggml_nbytes(t_bs));
    ggml_backend_tensor_set(t_bb, bn_bias.data(), 0, ggml_nbytes(t_bb));
    ggml_backend_tensor_set(t_b1, pw1_b.data(), 0, ggml_nbytes(t_b1));
    ggml_backend_tensor_set(t_b2, pw2_b.data(), 0, ggml_nbytes(t_b2));

    conf::BlockView vq{};
    vq.conv_pw1_w          = q1_1;
    vq.conv_pw1_b          = t_b1;
    vq.conv_dw_w           = t_dw;
    vq.conv_pw2_w          = q1_2;
    vq.conv_pw2_b          = t_b2;
    vq.conv_bn_fused_scale = t_bs;
    vq.conv_bn_fused_bias  = t_bb;
    conf::BlockView vf     = vq;
    vf.conv_pw1_w          = f_1;
    vf.conv_pw2_w          = f_2;

    // [direct_pw][0 = Q1_0, 1 = F32]
    ggml_init_params cp  = { 1024 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context *   ctx = ggml_init(cp);
    ggml_tensor *    x   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, T, 1);
    ggml_set_input(x);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_tensor * y[2][2];
    for (int direct = 0; direct < 2; ++direct) {
        conf::BlockParams p{};
        p.d_model          = static_cast<int>(d);
        p.n_head           = 1;
        p.conv_kernel      = static_cast<int>(k);
        p.kv_type          = GGML_TYPE_COUNT;
        p.use_flash        = false;
        p.policy.direct_pw = direct != 0;
        for (int f32 = 0; f32 < 2; ++f32) {
            y[direct][f32] = conf::conv_module(ctx, x, f32 ? vf : vq, p);
            ggml_set_output(y[direct][f32]);
            ggml_build_forward_expand(gf, y[direct][f32]);
        }
    }

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(bp.plan.primary));
    CHECK(ggml_gallocr_alloc_graph(galloc, gf));
    ggml_backend_tensor_set(x, xs.data(), 0, ggml_nbytes(x));
    CHECK(ggml_backend_graph_compute(bp.plan.primary, gf) == GGML_STATUS_SUCCESS);

    for (int direct = 0; direct < 2; ++direct) {
        std::vector<float> got[2];
        for (int f32 = 0; f32 < 2; ++f32) {
            CHECK(ggml_nelements(y[direct][f32]) == d * T);
            got[f32].resize(static_cast<size_t>(ggml_nelements(y[direct][f32])));
            ggml_backend_tensor_get(y[direct][f32], got[f32].data(), 0, ggml_nbytes(y[direct][f32]));
        }
        // Same activation-rounding allowance as the mul_mat check, over two
        // pointwise layers; a wrong bit or scale is an O(1) error.
        const double err    = rel_l2(got[0], got[1]);
        // The reference must be a real signal, or a backend that computed
        // nothing would pass with err == 0.
        double       ref_ms = 0.0;
        for (float v : got[1]) {
            ref_ms += static_cast<double>(v) * static_cast<double>(v) / static_cast<double>(got[1].size());
        }
        std::fprintf(stderr,
                     "%s: conv_module, Q1_0 pointwise vs F32, %s policy: rms(ref) = %.3f, ||err|| / ||ref|| = %.2e\n",
                     bp.label, direct ? "direct" : "im2col", std::sqrt(ref_ms), err);
        CHECK(std::isfinite(ref_ms) && std::sqrt(ref_ms) > 1e-2);
        CHECK(err < 2e-2);
    }

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    ggml_backend_buffer_free(wbuf);
    ggml_free(wctx);
}

}  // namespace

int main() {
    using namespace transcribe;

    const Q1Matrix    m    = make_q1_matrix(kOut, kIn, 0.02f);
    const std::string path = temp_gguf_path();
    if (!write_gguf(path, m)) {
        std::fprintf(stderr, "SETUP FAIL: could not write %s\n", path.c_str());
        return EXIT_FAILURE;
    }

    const transcribe_backend_request requests[2] = { TRANSCRIBE_BACKEND_CPU, TRANSCRIBE_BACKEND_AUTO };
    for (transcribe_backend_request req : requests) {
        BackendPlanName bp;
        bp.label = req == TRANSCRIBE_BACKEND_CPU ? "cpu" : "auto";
        if (load_common::init_backends(req, nullptr, kTag, bp.plan) != TRANSCRIBE_OK) {
            CHECK(req != TRANSCRIBE_BACKEND_CPU);
            continue;
        }
        if (req == TRANSCRIBE_BACKEND_AUTO && bp.plan.primary_kind == BackendKind::Cpu) {
            std::fprintf(stderr, "auto: no accelerator, CPU already covered\n");
        } else {
            bp.label = req == TRANSCRIBE_BACKEND_CPU ? "cpu" : ggml_backend_name(bp.plan.primary);
            check_backend(path, m, bp);
            check_conv_module(bp);
        }
        for (ggml_backend_t be : bp.plan.scheduler_list) {
            ggml_backend_free(be);
        }
    }

    std::error_code ec;
    std::filesystem::remove(transcribe::path_from_utf8(path), ec);

    if (g_failures > 0) {
        std::fprintf(stderr, "\n%d assertion(s) FAILED\n", g_failures);
    }
    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
