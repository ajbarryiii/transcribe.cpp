// conformer_input_quant_unit.cpp - contract test for conformer::InputQuant.
//
// Int8PerFrame is the activation quantizer checkpoints such as kea-b were
// trained with: per frame (one ne[0] row), across the input features,
//
//   s = max(max|x| / 127, 1e-30), q = round-half-even(x / s), x' = q * s
//
// in F32, at the input of every block projection (FF linear1/2, attention
// q/k/v/out/pos, conv pointwise 1/2). Asserts:
//
//   1. fake_quant_int8_per_frame matches that formula, computed on the host,
//      with either RowMax (the row max is exact both ways): bit-exact on the
//      CPU, including exact ties (half to even), all-zero
//      and floor-scaled rows, and 4096-wide rows. An accelerator's division
//      need not be correctly rounded, so there a row's scale may be an ulp
//      off (every x' of the row with it) and a near-tie code may move one
//      step (rarely); nothing else, and never on the tie or floor rows;
//   2. on the CPU, build_conformer_block over F32 weights matches a host
//      reference of the block (FF1, rel-pos MHSA, conv module, FF2, LN), with
//      the quantizer applied at the nine projection inputs when Int8PerFrame
//      is on. With InputQuant::None the same reference without the quantizer
//      must match, which anchors the reference itself; the quantized block
//      must differ from it by far more than the tolerance. Accelerators run
//      the block as a smoke test only: their F32 matmuls round the operands
//      (5e-4 relative on Vulkan here), and the quantizer turns differences of
//      that size into code flips, so only the CPU can pin the block down. The
//      graph itself is the same on every backend.
//
// Runs on the CPU, and also on the default accelerator (Metal, Vulkan, ...)
// when the build has one. The graphs run on the backend directly (no
// scheduler fallback), so every op must be supported there.

#include "conformer/conformer.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "transcribe-backend.h"
#include "transcribe-load-common.h"
#include "transcribe.h"

#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

namespace conf = transcribe::conformer;

int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

constexpr const char * kTag = "input-quant-test";

std::mt19937 g_rng(20261104u);

float next_normal(float sigma) {
    std::normal_distribution<float> d(0.0f, sigma);
    return d(g_rng);
}

std::vector<float> normals(size_t n, float sigma) {
    std::vector<float> v(n);
    for (float & x : v) {
        x = next_normal(sigma);
    }
    return v;
}

// The training quantizer on one frame, in F32 with IEEE division.
void ref_fake_quant(float * x, int64_t k) {
    float amax = 0.0f;
    for (int64_t i = 0; i < k; ++i) {
        amax = std::fmax(amax, std::fabs(x[i]));
    }
    float s = amax / 127.0f;
    if (s < 1e-30f) {
        s = 1e-30f;
    }
    for (int64_t i = 0; i < k; ++i) {
        x[i] = std::nearbyint(x[i] / s) * s;  // FE_TONEAREST: half to even
    }
}

void ref_fake_quant_rows(std::vector<float> & x, int64_t k) {
    for (size_t r = 0; r * k < x.size(); ++r) {
        ref_fake_quant(x.data() + r * k, k);
    }
}

double rel_l2(const std::vector<float> & a, const std::vector<float> & b) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        num += d * d;
        den += static_cast<double>(b[i]) * static_cast<double>(b[i]);
    }
    return std::sqrt(num / (den + 1e-30));
}

struct Backend {
    ggml_backend_t backend;
    const char *   label;
    bool           cpu;
};

// Runs gf on the backend after `fill` uploads the inputs.
template <typename Fill> bool compute(const Backend & be, ggml_cgraph * gf, Fill fill, ggml_gallocr_t * out_galloc) {
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be.backend));
    *out_galloc           = galloc;
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        return false;
    }
    fill();
    return ggml_backend_graph_compute(be.backend, gf) == GGML_STATUS_SUCCESS;
}

std::vector<float> read(ggml_tensor * t) {
    std::vector<float> v(static_cast<size_t>(ggml_nelements(t)));
    ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
    return v;
}

// ----- 1. the quantizer -----

void check_quantizer(const Backend & be, int64_t k, int64_t t, int64_t b, conf::RowMax row_max) {
    const int64_t      rows = t * b;
    std::vector<float> xs   = normals(static_cast<size_t>(k * rows), 1.0f);
    // Heavy-tailed outliers, so scales vary across rows.
    for (int64_t r = 0; r < rows; ++r) {
        xs[static_cast<size_t>(r * k + (r * 7) % k)] *= static_cast<float>(1 + r % 9);
    }
    // Row 0: max|x| = 127 makes s = 1, so x / s is exact and every x.5 is a
    // tie: half to even.
    float * r0 = xs.data();
    for (int64_t i = 0; i < k; ++i) {
        r0[i] = static_cast<float>(static_cast<int>(i % 254) - 127) + 0.5f;
    }
    r0[0] = 127.0f;
    r0[1] = -126.5f;
    // Row 1: all zero. Row 2: below the floor (s = 1e-30, q = 0). Row 3:
    // floor-scaled with non-zero codes (x / 1e-30 around 10).
    for (int64_t i = 0; i < k; ++i) {
        xs[static_cast<size_t>(1 * k + i)] = 0.0f;
        xs[static_cast<size_t>(2 * k + i)] = 1e-35f * static_cast<float>(i % 5);
        xs[static_cast<size_t>(3 * k + i)] = 1e-29f * static_cast<float>(i % 3 - 1);
    }
    std::vector<float> want = xs;
    ref_fake_quant_rows(want, k);
    // Spot-check the reference on the tie row: -126.5 -> -126, 0.5 -> 0,
    // 1.5 -> 2, 2.5 -> 2.
    CHECK(want[1] == -126.0f);
    if (k >= 256) {
        CHECK(want[127] == 0.0f && want[128] == 2.0f && want[129] == 2.0f);
    }

    ggml_init_params cp  = { 64 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context *   ctx = ggml_init(cp);
    ggml_tensor *    x   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, t, b);
    ggml_set_input(x);
    ggml_tensor * y = conf::fake_quant_int8_per_frame(ctx, x, row_max);
    ggml_set_output(y);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);

    ggml_gallocr_t galloc = nullptr;
    const bool     ok     = compute(be, gf, [&] { ggml_backend_tensor_set(x, xs.data(), 0, ggml_nbytes(x)); }, &galloc);
    CHECK(ok);
    if (ok) {
        const std::vector<float> got = read(y);
        CHECK(got.size() == want.size());
        int64_t exact = 0, ulp = 0, one_step = 0, worse = 0, special_bad = 0;
        for (int64_t r = 0; r < rows; ++r) {
            float amax = 0.0f;
            for (int64_t i = 0; i < k; ++i) {
                amax = std::fmax(amax, std::fabs(xs[static_cast<size_t>(r * k + i)]));
            }
            const double step = std::fmax(amax / 127.0, 1e-30);
            for (int64_t i = 0; i < k; ++i) {
                const size_t j = static_cast<size_t>(r * k + i);
                if (got[j] == want[j]) {
                    ++exact;
                    continue;
                }
                if (r < 4) {
                    ++special_bad;
                }
                // A scale off by an ulp moves x' = q * s by about an ulp of
                // it; a flipped near-tie code moves it by one step.
                const double d = std::fabs(static_cast<double>(got[j]) - want[j]);
                if (d <= 4e-7 * std::fabs(want[j])) {
                    ++ulp;
                } else if (std::fabs(d - step) <= 1e-5 * step) {
                    ++one_step;
                } else {
                    ++worse;
                }
            }
        }
        std::fprintf(
            stderr,
            "%s: fake_quant %s [%lld, %lld, %lld]: %lld exact, %lld within ~an ulp, %lld one step off, %lld worse\n",
            be.label, row_max == conf::RowMax::ArgMax ? "argmax" : "top_k", (long long) k, (long long) t, (long long) b,
            (long long) exact, (long long) ulp, (long long) one_step, (long long) worse);
        CHECK(worse == 0);
        CHECK(special_bad == 0);
        if (be.cpu) {
            CHECK(ulp == 0 && one_step == 0);
        } else {
            CHECK(one_step * 10000 <= rows * k);
        }
    }
    ggml_gallocr_free(galloc);
    ggml_free(ctx);
}

// ----- 2. the block -----

struct Dims {
    int64_t d = 64, ff = 128, heads = 2, kconv = 5, T = 9, B = 1;
};

// Host weights, row-major [out][in] for linears (= ggml ne [in, out]).
struct HostBlock {
    std::vector<float> ln_ff1_w, ln_ff1_b, ff1_1, ff1_2;
    std::vector<float> ln_att_w, ln_att_b, wq, wk, wv, wo, wpos, pos_u, pos_v;
    std::vector<float> ln_conv_w, ln_conv_b, pw1, dw, bn_s, bn_b, pw2;
    std::vector<float> ln_ff2_w, ln_ff2_b, ff2_1, ff2_2;
    std::vector<float> ln_out_w, ln_out_b;
};

std::vector<float> near_one(int64_t n) {
    std::vector<float> v = normals(static_cast<size_t>(n), 0.1f);
    for (float & x : v) {
        x += 1.0f;
    }
    return v;
}

HostBlock make_block(const Dims & D) {
    HostBlock     h;
    const int64_t d = D.d, f = D.ff, hd = D.d / D.heads;
    const float   sd = 1.0f / std::sqrt(static_cast<float>(d)), sf = 1.0f / std::sqrt(static_cast<float>(f));
    h.ln_ff1_w  = near_one(d);
    h.ln_ff1_b  = normals(d, 0.1f);
    h.ff1_1     = normals(f * d, sd);
    h.ff1_2     = normals(d * f, sf);
    h.ln_att_w  = near_one(d);
    h.ln_att_b  = normals(d, 0.1f);
    h.wq        = normals(d * d, sd);
    h.wk        = normals(d * d, sd);
    h.wv        = normals(d * d, sd);
    h.wo        = normals(d * d, sd);
    h.wpos      = normals(d * d, sd);
    h.pos_u     = normals(hd * D.heads, 0.2f);
    h.pos_v     = normals(hd * D.heads, 0.2f);
    h.ln_conv_w = near_one(d);
    h.ln_conv_b = normals(d, 0.1f);
    h.pw1       = normals(2 * d * d, sd);
    h.dw        = normals(d * D.kconv, 0.4f);
    h.bn_s      = near_one(d);
    h.bn_b      = normals(d, 0.1f);
    h.pw2       = normals(d * d, sd);
    h.ln_ff2_w  = near_one(d);
    h.ln_ff2_b  = normals(d, 0.1f);
    h.ff2_1     = normals(f * d, sd);
    h.ff2_2     = normals(d * f, sf);
    h.ln_out_w  = near_one(d);
    h.ln_out_b  = normals(d, 0.1f);
    return h;
}

// Frames are rows of length `in`; y [n, out].
std::vector<float> linear(const std::vector<float> & w, const std::vector<float> & x, int64_t in, int64_t out) {
    const int64_t      n = static_cast<int64_t>(x.size()) / in;
    std::vector<float> y(static_cast<size_t>(n * out));
    for (int64_t r = 0; r < n; ++r) {
        for (int64_t o = 0; o < out; ++o) {
            double acc = 0.0;
            for (int64_t i = 0; i < in; ++i) {
                acc += static_cast<double>(w[o * in + i]) * x[r * in + i];
            }
            y[r * out + o] = static_cast<float>(acc);
        }
    }
    return y;
}

std::vector<float> qlin(const std::vector<float> & w, std::vector<float> x, int64_t in, int64_t out, bool q) {
    if (q) {
        ref_fake_quant_rows(x, in);
    }
    return linear(w, x, in, out);
}

std::vector<float> layer_norm(const std::vector<float> & x,
                              const std::vector<float> & g,
                              const std::vector<float> & b,
                              int64_t                    d) {
    std::vector<float> y(x.size());
    for (size_t r = 0; r * d < x.size(); ++r) {
        double mean = 0.0, var = 0.0;
        for (int64_t i = 0; i < d; ++i) {
            mean += x[r * d + i];
        }
        mean /= d;
        for (int64_t i = 0; i < d; ++i) {
            var += (x[r * d + i] - mean) * (x[r * d + i] - mean);
        }
        var /= d;
        for (int64_t i = 0; i < d; ++i) {
            y[r * d + i] =
                static_cast<float>((x[r * d + i] - mean) / std::sqrt(var + conf::kLayerNormEps) * g[i] + b[i]);
        }
    }
    return y;
}

float silu(float v) {
    return v / (1.0f + std::exp(-v));
}

std::vector<float> ff(const std::vector<float> & x,
                      const std::vector<float> & w1,
                      const std::vector<float> & w2,
                      const Dims &               D,
                      bool                       q) {
    std::vector<float> h = qlin(w1, x, D.d, D.ff, q);
    for (float & v : h) {
        v = silu(v);
    }
    return qlin(w2, h, D.ff, D.d, q);
}

// x [B][T][d]; pos [2T-1][d], row r = relative offset (T-1) - r.
std::vector<float> mhsa(const std::vector<float> & x,
                        const std::vector<float> & pos,
                        const HostBlock &          h,
                        const Dims &               D,
                        bool                       q) {
    const int64_t      d = D.d, T = D.T, H = D.heads, hd = D.d / D.heads;
    const double       scale = 1.0 / std::sqrt(static_cast<double>(hd));
    std::vector<float> Q = qlin(h.wq, x, d, d, q), K = qlin(h.wk, x, d, d, q), V = qlin(h.wv, x, d, d, q);
    std::vector<float> P = qlin(h.wpos, pos, d, d, q);
    std::vector<float> o(x.size());
    for (int64_t b = 0; b < D.B; ++b) {
        for (int64_t hh = 0; hh < H; ++hh) {
            for (int64_t i = 0; i < T; ++i) {
                std::vector<double> s(static_cast<size_t>(T));
                double              mx = -1e300;
                for (int64_t j = 0; j < T; ++j) {
                    double        ac = 0.0, bd = 0.0;
                    const int64_t r = T - 1 + j - i;
                    for (int64_t e = 0; e < hd; ++e) {
                        const int64_t c  = hh * hd + e;
                        const double  qi = Q[(b * T + i) * d + c];
                        ac += (qi + h.pos_u[c]) * K[(b * T + j) * d + c];
                        bd += (qi + h.pos_v[c]) * P[r * d + c];
                    }
                    s[j] = (ac + bd) * scale;
                    mx   = std::fmax(mx, s[j]);
                }
                double z = 0.0;
                for (double & v : s) {
                    v = std::exp(v - mx);
                    z += v;
                }
                for (int64_t e = 0; e < hd; ++e) {
                    const int64_t c   = hh * hd + e;
                    double        acc = 0.0;
                    for (int64_t j = 0; j < T; ++j) {
                        acc += s[j] / z * V[(b * T + j) * d + c];
                    }
                    o[(b * T + i) * d + c] = static_cast<float>(acc);
                }
            }
        }
    }
    return qlin(h.wo, o, d, d, q);
}

std::vector<float> conv(const std::vector<float> & x, const HostBlock & h, const Dims & D, bool q) {
    const int64_t      d = D.d, T = D.T, k = D.kconv, pad = (D.kconv - 1) / 2;
    std::vector<float> a = qlin(h.pw1, x, d, 2 * d, q);
    std::vector<float> g(x.size()), c(x.size());
    for (int64_t r = 0; r < D.B * T; ++r) {
        for (int64_t i = 0; i < d; ++i) {
            const float v = a[r * 2 * d + d + i];
            g[r * d + i]  = a[r * 2 * d + i] / (1.0f + std::exp(-v));
        }
    }
    for (int64_t b = 0; b < D.B; ++b) {
        for (int64_t t = 0; t < T; ++t) {
            for (int64_t i = 0; i < d; ++i) {
                double acc = 0.0;
                for (int64_t j = 0; j < k; ++j) {
                    const int64_t tt = t + j - pad;
                    if (tt >= 0 && tt < T) {
                        acc += static_cast<double>(h.dw[i * k + j]) * g[(b * T + tt) * d + i];
                    }
                }
                c[(b * T + t) * d + i] = silu(static_cast<float>(acc) * h.bn_s[i] + h.bn_b[i]);
            }
        }
    }
    return qlin(h.pw2, c, d, d, q);
}

std::vector<float> ref_block(std::vector<float>         x,
                             const std::vector<float> & pos,
                             const HostBlock &          h,
                             const Dims &               D,
                             bool                       q) {
    auto add = [&](const std::vector<float> & y, float w) {
        for (size_t i = 0; i < x.size(); ++i) {
            x[i] += w * y[i];
        }
    };
    add(ff(layer_norm(x, h.ln_ff1_w, h.ln_ff1_b, D.d), h.ff1_1, h.ff1_2, D, q), 0.5f);
    add(mhsa(layer_norm(x, h.ln_att_w, h.ln_att_b, D.d), pos, h, D, q), 1.0f);
    add(conv(layer_norm(x, h.ln_conv_w, h.ln_conv_b, D.d), h, D, q), 1.0f);
    add(ff(layer_norm(x, h.ln_ff2_w, h.ln_ff2_b, D.d), h.ff2_1, h.ff2_2, D, q), 0.5f);
    return layer_norm(x, h.ln_out_w, h.ln_out_b, D.d);
}

// B == 1 lets the im2col pointwise policy stand (B > 1 always goes direct).
void check_block(const Backend & be, int64_t batch) {
    Dims D;
    D.B                          = batch;
    const HostBlock          h   = make_block(D);
    const int64_t            d   = D.d;
    const std::vector<float> xs  = normals(static_cast<size_t>(d * D.T * D.B), 1.0f);
    const std::vector<float> pos = normals(static_cast<size_t>(d * (2 * D.T - 1)), 1.0f);

    ggml_init_params wp   = { 64 * ggml_tensor_overhead(), nullptr, true };
    ggml_context *   wctx = ggml_init(wp);

    struct Up {
        ggml_tensor *              t;
        const std::vector<float> * v;
    };

    std::vector<Up> ups;
    auto            w2 = [&](const std::vector<float> & v, int64_t ne0, int64_t ne1) {
        ggml_tensor * t = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, ne0, ne1);
        ups.push_back({ t, &v });
        return t;
    };
    auto w3 = [&](const std::vector<float> & v, int64_t ne0, int64_t ne1, int64_t ne2) {
        ggml_tensor * t = ggml_new_tensor_3d(wctx, GGML_TYPE_F32, ne0, ne1, ne2);
        ups.push_back({ t, &v });
        return t;
    };
    auto w1 = [&](const std::vector<float> & v) {
        ggml_tensor * t = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, static_cast<int64_t>(v.size()));
        ups.push_back({ t, &v });
        return t;
    };
    conf::BlockView v{};
    v.norm_ff1_w          = w1(h.ln_ff1_w);
    v.norm_ff1_b          = w1(h.ln_ff1_b);
    v.ff1_lin1_w          = w2(h.ff1_1, d, D.ff);
    v.ff1_lin2_w          = w2(h.ff1_2, D.ff, d);
    v.norm_attn_w         = w1(h.ln_att_w);
    v.norm_attn_b         = w1(h.ln_att_b);
    v.attn_q_w            = w2(h.wq, d, d);
    v.attn_k_w            = w2(h.wk, d, d);
    v.attn_v_w            = w2(h.wv, d, d);
    v.attn_out_w          = w2(h.wo, d, d);
    v.attn_pos_w          = w2(h.wpos, d, d);
    v.attn_pos_u          = w2(h.pos_u, d / D.heads, D.heads);
    v.attn_pos_v          = w2(h.pos_v, d / D.heads, D.heads);
    v.norm_conv_w         = w1(h.ln_conv_w);
    v.norm_conv_b         = w1(h.ln_conv_b);
    // Pointwise kernels in the conv layout [1, in, out], which both
    // pointwise paths take.
    v.conv_pw1_w          = w3(h.pw1, 1, d, 2 * d);
    v.conv_dw_w           = w3(h.dw, D.kconv, 1, d);
    v.conv_pw2_w          = w3(h.pw2, 1, d, d);
    v.conv_bn_fused_scale = w1(h.bn_s);
    v.conv_bn_fused_bias  = w1(h.bn_b);
    v.norm_ff2_w          = w1(h.ln_ff2_w);
    v.norm_ff2_b          = w1(h.ln_ff2_b);
    v.ff2_lin1_w          = w2(h.ff2_1, d, D.ff);
    v.ff2_lin2_w          = w2(h.ff2_2, D.ff, d);
    v.norm_out_w          = w1(h.ln_out_w);
    v.norm_out_b          = w1(h.ln_out_b);

    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, be.backend);
    CHECK(wbuf != nullptr);
    if (wbuf == nullptr) {
        ggml_free(wctx);
        return;
    }
    for (const Up & u : ups) {
        CHECK(static_cast<int64_t>(u.v->size()) == ggml_nelements(u.t));
        ggml_backend_tensor_set(u.t, u.v->data(), 0, ggml_nbytes(u.t));
    }

    // [quant][direct_pw]
    ggml_init_params cp  = { 4096 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context *   ctx = ggml_init(cp);
    ggml_tensor *    x   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, D.T, D.B);
    ggml_tensor *    pe  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, 2 * D.T - 1);
    ggml_set_input(x);
    ggml_set_input(pe);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_tensor * y[2][2];
    for (int q = 0; q < 2; ++q) {
        for (int direct = 0; direct < 2; ++direct) {
            conf::BlockParams p{};
            p.d_model             = static_cast<int>(d);
            p.n_head              = static_cast<int>(D.heads);
            p.conv_kernel         = static_cast<int>(D.kconv);
            p.kv_type             = GGML_TYPE_COUNT;
            p.use_flash           = false;
            p.policy.direct_pw    = direct != 0;
            p.input_quant.mode    = q ? conf::InputQuant::Int8PerFrame : conf::InputQuant::None;
            // The runtime's choice for this backend (check 1 covers both).
            p.input_quant.row_max = conf::row_max_for_backend(ggml_backend_name(be.backend));
            y[q][direct]          = conf::build_conformer_block(ctx, x, pe, v, p);
            ggml_set_output(y[q][direct]);
            ggml_build_forward_expand(gf, y[q][direct]);
        }
    }

    ggml_gallocr_t galloc = nullptr;
    const bool     ok     = compute(
        be, gf,
        [&] {
            ggml_backend_tensor_set(x, xs.data(), 0, ggml_nbytes(x));
            ggml_backend_tensor_set(pe, pos.data(), 0, ggml_nbytes(pe));
        },
        &galloc);
    CHECK(ok);
    if (ok) {
        const std::vector<float> want[2] = { ref_block(xs, pos, h, D, false), ref_block(xs, pos, h, D, true) };
        const double             apart   = rel_l2(want[0], want[1]);
        for (int q = 0; q < 2; ++q) {
            for (int direct = 0; direct < 2; ++direct) {
                const std::vector<float> got = read(y[q][direct]);
                CHECK(got.size() == want[q].size());
                const double err = rel_l2(got, want[q]);
                std::fprintf(stderr,
                             "%s: block B=%lld, input_quant=%s, %s pointwise policy: ||err|| / ||ref|| = %.2e "
                             "(quantized vs float reference: %.2e)\n",
                             be.label, (long long) D.B, q ? "int8_per_frame" : "none", direct ? "direct" : "im2col",
                             err, apart);
                CHECK(std::isfinite(err));
                if (be.cpu) {
                    // Float graph vs double host sums; the quantized graph may
                    // also flip a rare near-tie code. A missing or extra
                    // quantizer costs `apart`.
                    CHECK(err < 2e-4);
                } else {
                    CHECK(err < 1.5 * apart);
                }
            }
        }
        CHECK(apart > 5e-3);
    }

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    ggml_backend_buffer_free(wbuf);
    ggml_free(wctx);
}

}  // namespace

int main() {
    using namespace transcribe;
    std::fesetround(FE_TONEAREST);

    // Metal (registry "MTL", backends "MTL0", ...) takes argmax, the rest top_k.
    CHECK(conf::row_max_for_backend("MTL0") == conf::RowMax::ArgMax);
    CHECK(conf::row_max_for_backend("CPU") == conf::RowMax::TopK);
    CHECK(conf::row_max_for_backend("Vulkan0") == conf::RowMax::TopK);
    CHECK(conf::row_max_for_backend(nullptr) == conf::RowMax::TopK);

    const transcribe_backend_request requests[2] = { TRANSCRIBE_BACKEND_CPU, TRANSCRIBE_BACKEND_AUTO };
    for (transcribe_backend_request req : requests) {
        BackendPlan plan;
        if (load_common::init_backends(req, nullptr, kTag, plan) != TRANSCRIBE_OK) {
            CHECK(req != TRANSCRIBE_BACKEND_CPU);
            continue;
        }
        if (req == TRANSCRIBE_BACKEND_AUTO && plan.primary_kind == BackendKind::Cpu) {
            std::fprintf(stderr, "auto: no accelerator, CPU already covered\n");
        } else {
            Backend be;
            be.backend                      = plan.primary;
            be.cpu                          = req == TRANSCRIBE_BACKEND_CPU;
            be.label                        = be.cpu ? "cpu" : ggml_backend_name(plan.primary);
            const conf::RowMax row_maxes[2] = { conf::RowMax::TopK, conf::RowMax::ArgMax };
            for (conf::RowMax rm : row_maxes) {
                check_quantizer(be, 1024, 37, 2, rm);
                check_quantizer(be, 4096, 5, 1, rm);
                check_quantizer(be, 128, 300, 1, rm);
            }
            check_block(be, 1);
            check_block(be, 2);
        }
        for (ggml_backend_t b : plan.scheduler_list) {
            ggml_backend_free(b);
        }
    }

    if (g_failures > 0) {
        std::fprintf(stderr, "\n%d assertion(s) FAILED\n", g_failures);
    }
    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
