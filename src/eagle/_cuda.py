"""EAGLE の fused CUDA カーネル (JIT コンパイル)。

PyTorch 本体の fused Adam/SGD と同じ発想で、1 パラメータテンソルにつき
1 カーネル起動で以下を全て行う:
  weight decay → Δp/Δg 計算 → モーメント更新 → マスク判定 →
  ベース更新 or EAGLE 更新 → 使用回数カウント (block 内 reduction + atomicAdd)

初回呼び出し時に nvcc でコンパイルされ、~/.cache/torch_extensions にキャッシュされる。
"""

import os
import warnings

_CPP_SRC = r"""
#include <torch/extension.h>
void eagle_step_adam(at::Tensor p, at::Tensor g, at::Tensor exp_avg,
                     at::Tensor exp_avg_sq, at::Tensor prev_p, at::Tensor prev_g,
                     at::Tensor threshold, at::Tensor eagle_count, at::Tensor base_count,
                     at::Tensor jumped, at::Tensor cooldown, at::Tensor h_ema,
                     at::Tensor h_var,
                     double lr, double beta1, double beta2, double eps,
                     double wd, double bc1, double bc2, double coeff,
                     double snr_c, double kappa, int64_t cooldown_steps,
                     int64_t allow_eagle, double beta_h, int64_t use_h_ema,
                     int64_t clean_measure, double msnr, double conf_c,
                     int64_t mom_jump, int64_t always_jump);
void eagle_step_sgd(at::Tensor p, at::Tensor g, at::Tensor buf,
                    at::Tensor prev_p, at::Tensor prev_g,
                    at::Tensor threshold, at::Tensor eagle_count, at::Tensor base_count,
                    double lr, double momentum, double wd, double coeff,
                    double kappa);
"""

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <algorithm>

namespace {

constexpr int kThreads = 256;

int num_blocks(int64_t n) {
  return (int)std::min<int64_t>((n + kThreads - 1) / kThreads, 4096);
}

// 共通部: マスク判定と更新の適用。base_step はベース更新量 (lr 込み)。
// thr: |Δg| がこれ未満ならベース更新 (SNR ゲート有効時は呼び出し側で解決済み)
// lim: EAGLE ステップの絶対値上限 (trust region)。<= 0 で無効
// force_base: true ならベース更新を強制 (クールダウン中)
// 戻り値: EAGLE 更新を適用したか
__device__ __forceinline__ bool apply_update(
    float& pi, float gi, float pgi, float dp, float dg, float thr,
    float base_step, float coeff, float lim, bool force_base,
    unsigned long long& my_eagle, unsigned long long& my_base) {
  const bool stable = (pgi * gi >= 0.f) && (gi * dg >= 0.f);
  const bool base = force_base || stable || (fabsf(dg) < thr);
  if (base) {
    pi -= base_step;
    my_base++;
    return false;
  }
  const float dgs = (fabsf(dg) < 1e-8f) ? 1e-8f : dg;
  float d = coeff * gi * dp / dgs;
  if (lim > 0.f) d = fminf(fmaxf(d, -lim), lim);
  pi -= d;
  my_eagle++;
  return true;
}

__global__ void eagle_adam_kernel(
    float* __restrict__ p, const float* __restrict__ g,
    float* __restrict__ exp_avg, float* __restrict__ exp_avg_sq,
    float* __restrict__ prev_p, float* __restrict__ prev_g,
    const float* __restrict__ threshold,
    uint8_t* __restrict__ jumped, uint8_t* __restrict__ cooldown,
    float* __restrict__ h_ema, float* __restrict__ h_var,
    const float lr, const float beta1, const float beta2, const float eps,
    const float wd, const float bc1, const float bc2, const float coeff,
    const float snr_c, const float kappa, const int cooldown_steps,
    const int allow_eagle, const float beta_h, const int use_h_ema,
    const int clean_measure, const float msnr, const float conf_c,
    const int mom_jump, const int always_jump,
    const int64_t n,
    unsigned long long* __restrict__ eagle_count,
    unsigned long long* __restrict__ base_count) {
  __shared__ unsigned long long s_eagle, s_base;
  if (threadIdx.x == 0) { s_eagle = 0; s_base = 0; }
  __syncthreads();

  const float thr = *threshold;
  unsigned long long my_eagle = 0, my_base = 0;

  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (int64_t)gridDim.x * blockDim.x) {
    float pi = p[i];
    if (wd != 0.f) pi *= (1.f - lr * wd);
    const float gi = g[i];
    const float pgi = prev_g[i];
    const float dp = pi - prev_p[i];
    const float dg = gi - pgi;

    const float mi = beta1 * exp_avg[i] + (1.f - beta1) * gi;
    const float vi = beta2 * exp_avg_sq[i] + (1.f - beta2) * gi * gi;
    exp_avg[i] = mi;
    exp_avg_sq[i] = vi;
    prev_p[i] = pi;
    prev_g[i] = gi;

    const float vhat_sqrt = sqrtf(vi) / bc2;
    const float denom = vhat_sqrt + eps;
    const float base_step = (lr / bc1) * mi / denom;
    // B2: SNR ゲート (|Δg| を勾配の典型スケール √v̂ と比較)
    const float eff_thr = (snr_c > 0.f) ? snr_c * vhat_sqrt : thr;
    // B1: trust region (EAGLE ステップを κ·|adam_step| に制限)
    const float lim = (kappa > 0.f) ? kappa * fabsf(base_step) : -1.f;

    // ペアバッチ割線: allow_eagle=0 のステップ (異バッチの Δg) は全要素
    // ベース更新に固定し、クールダウン状態も凍結する
    bool force_base = (allow_eagle == 0);
    // B3: 失敗判定つきクールダウン。前ステップで EAGLE ジャンプした要素が
    // 勾配符号反転かつ |g| 増大していたら「失敗」→ K ステップ Adam に固定
    uint8_t cd = 0;
    if (cooldown_steps > 0 && allow_eagle) {
      cd = cooldown[i];
      if (jumped[i] && (pgi * gi < 0.f) && (fabsf(gi) > fabsf(pgi)))
        cd = (uint8_t)cooldown_steps;
      force_base = force_base || (cd > 0);
    }

    bool took_eagle;
    if (use_h_ema) {
      // 間欠ペア + 曲率 EMA: clean ステップ (同一バッチ 2 発目) のみ
      // Δg/Δp を測定して EMA を更新し、ジャンプは常にキャッシュ曲率で行う
      float he = h_ema[i];
      if (clean_measure && fabsf(dp) > 1e-12f) {
        const float hn = dg / dp;
        if (hn > 0.f) {  // 正曲率のみ採用 (負曲率へのニュートン步は発散方向)
          if (conf_c > 0.f) {
            // 曲率の分散 EMA も更新 (信頼度ゲート用)。初回は CV=1 で初期化
            float hv = h_var[i];
            if (he > 0.f) {
              const float delta = hn - he;
              he = beta_h * he + (1.f - beta_h) * hn;
              hv = beta_h * hv + (1.f - beta_h) * delta * delta;
            } else {
              he = hn;
              hv = hn * hn;
            }
            h_var[i] = hv;
          } else {
            he = (he > 0.f) ? beta_h * he + (1.f - beta_h) * hn : hn;
          }
          h_ema[i] = he;
        }
      }
      // always_jump では振動判定を外し、EMA 有効な全要素で発火する
      bool fire = !force_base && he > 0.f;
      if (!always_jump) {
        const bool stable = (pgi * gi >= 0.f) && (gi * dg >= 0.f);
        fire = fire && !stable;
      }
      // G1: Δg ベースの SNR 発火ゲート (勾配変化が典型スケールを超えるか)
      if (snr_c > 0.f) fire = fire && (fabsf(dg) >= snr_c * vhat_sqrt);
      // G2: モーメンタム SNR 発火ゲート (|m̂|/√v̂ — Adam が一貫した信号を
      // 見ている座標のみ発火。ノイズ床では m̂→0 で自動停止)
      if (msnr > 0.f) fire = fire && (fabsf(mi) / bc1 >= msnr * vhat_sqrt);
      // 曲率信頼度ゲート: 測定の変動係数が conf_c 未満 (= 曲率推定が安定)
      // の座標のみ発火
      if (conf_c > 0.f) fire = fire && (h_var[i] < conf_c * conf_c * he * he);
      if (fire) {
        // 分子: 生の勾配 g、または mom_jump ならノイズ平均された m̂
        const float num = mom_jump ? (mi / bc1) : gi;
        // 分母フロアは vectorized 実装 (h_ema.clamp(min=1e-12)) と一致させる
        float d = coeff * num / fmaxf(he, 1e-12f);
        if (lim > 0.f) d = fminf(fmaxf(d, -lim), lim);
        pi -= d;
        my_eagle++;
        took_eagle = true;
      } else {
        pi -= base_step;
        my_base++;
        took_eagle = false;
      }
    } else {
      took_eagle = apply_update(
          pi, gi, pgi, dp, dg, eff_thr, base_step, coeff, lim, force_base,
          my_eagle, my_base);
    }
    if (cooldown_steps > 0 && allow_eagle) {
      cooldown[i] = (cd > 0) ? (uint8_t)(cd - 1) : cd;
      jumped[i] = took_eagle ? 1 : 0;
    }
    p[i] = pi;
  }

  atomicAdd(&s_eagle, my_eagle);
  atomicAdd(&s_base, my_base);
  __syncthreads();
  if (threadIdx.x == 0) {
    atomicAdd(eagle_count, s_eagle);
    atomicAdd(base_count, s_base);
  }
}

__global__ void eagle_sgd_kernel(
    float* __restrict__ p, const float* __restrict__ g,
    float* __restrict__ buf,
    float* __restrict__ prev_p, float* __restrict__ prev_g,
    const float* __restrict__ threshold,
    const float lr, const float momentum, const float wd, const float coeff,
    const float kappa,
    const int64_t n,
    unsigned long long* __restrict__ eagle_count,
    unsigned long long* __restrict__ base_count) {
  __shared__ unsigned long long s_eagle, s_base;
  if (threadIdx.x == 0) { s_eagle = 0; s_base = 0; }
  __syncthreads();

  const float thr = *threshold;
  unsigned long long my_eagle = 0, my_base = 0;

  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (int64_t)gridDim.x * blockDim.x) {
    float pi = p[i];
    if (wd != 0.f) pi *= (1.f - lr * wd);
    const float gi = g[i];
    const float pgi = prev_g[i];
    const float dp = pi - prev_p[i];
    const float dg = gi - pgi;

    const float bi = momentum * buf[i] + gi;
    buf[i] = bi;
    prev_p[i] = pi;
    prev_g[i] = gi;

    const float base_step = lr * bi;
    const float lim = (kappa > 0.f) ? kappa * fabsf(base_step) : -1.f;
    apply_update(pi, gi, pgi, dp, dg, thr, base_step, coeff, lim, false,
                 my_eagle, my_base);
    p[i] = pi;
  }

  atomicAdd(&s_eagle, my_eagle);
  atomicAdd(&s_base, my_base);
  __syncthreads();
  if (threadIdx.x == 0) {
    atomicAdd(eagle_count, s_eagle);
    atomicAdd(base_count, s_base);
  }
}

}  // namespace

void eagle_step_adam(at::Tensor p, at::Tensor g, at::Tensor exp_avg,
                     at::Tensor exp_avg_sq, at::Tensor prev_p, at::Tensor prev_g,
                     at::Tensor threshold, at::Tensor eagle_count, at::Tensor base_count,
                     at::Tensor jumped, at::Tensor cooldown, at::Tensor h_ema,
                     at::Tensor h_var,
                     double lr, double beta1, double beta2, double eps,
                     double wd, double bc1, double bc2, double coeff,
                     double snr_c, double kappa, int64_t cooldown_steps,
                     int64_t allow_eagle, double beta_h, int64_t use_h_ema,
                     int64_t clean_measure, double msnr, double conf_c,
                     int64_t mom_jump, int64_t always_jump) {
  const int64_t n = p.numel();
  if (n == 0) return;
  auto stream = at::cuda::getCurrentCUDAStream();
  eagle_adam_kernel<<<num_blocks(n), kThreads, 0, stream>>>(
      p.data_ptr<float>(), g.data_ptr<float>(),
      exp_avg.data_ptr<float>(), exp_avg_sq.data_ptr<float>(),
      prev_p.data_ptr<float>(), prev_g.data_ptr<float>(),
      threshold.data_ptr<float>(),
      jumped.data_ptr<uint8_t>(), cooldown.data_ptr<uint8_t>(),
      h_ema.data_ptr<float>(), h_var.data_ptr<float>(),
      (float)lr, (float)beta1, (float)beta2, (float)eps,
      (float)wd, (float)bc1, (float)bc2, (float)coeff,
      (float)snr_c, (float)kappa, (int)cooldown_steps, (int)allow_eagle,
      (float)beta_h, (int)use_h_ema, (int)clean_measure, (float)msnr,
      (float)conf_c, (int)mom_jump, (int)always_jump, n,
      reinterpret_cast<unsigned long long*>(eagle_count.data_ptr<int64_t>()),
      reinterpret_cast<unsigned long long*>(base_count.data_ptr<int64_t>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void eagle_step_sgd(at::Tensor p, at::Tensor g, at::Tensor buf,
                    at::Tensor prev_p, at::Tensor prev_g,
                    at::Tensor threshold, at::Tensor eagle_count, at::Tensor base_count,
                    double lr, double momentum, double wd, double coeff,
                    double kappa) {
  const int64_t n = p.numel();
  if (n == 0) return;
  auto stream = at::cuda::getCurrentCUDAStream();
  eagle_sgd_kernel<<<num_blocks(n), kThreads, 0, stream>>>(
      p.data_ptr<float>(), g.data_ptr<float>(), buf.data_ptr<float>(),
      prev_p.data_ptr<float>(), prev_g.data_ptr<float>(),
      threshold.data_ptr<float>(),
      (float)lr, (float)momentum, (float)wd, (float)coeff,
      (float)kappa, n,
      reinterpret_cast<unsigned long long*>(eagle_count.data_ptr<int64_t>()),
      reinterpret_cast<unsigned long long*>(base_count.data_ptr<int64_t>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

_ext = None
_compile_failed = False


def get_extension():
    """fused カーネル拡張を返す。コンパイル不可の環境では None (フォールバック)。"""
    global _ext, _compile_failed
    if _ext is not None or _compile_failed:
        return _ext
    try:
        from torch.utils.cpp_extension import load_inline
        cuda_flags = ["-O3"]
        # ホスト gcc が nvcc の対応表より新しいだけで弾かれる環境向けの
        # 明示的オプトイン (既定では付けない)。使う場合は必ず
        # experiments/verify_fused.py で vectorized 経路との数値一致を
        # 確認してから計測に用いること。
        if os.environ.get("EAGLE_ALLOW_UNSUPPORTED_COMPILER") == "1":
            cuda_flags.append("-allow-unsupported-compiler")
        _ext = load_inline(
            name="eagle_cuda_ext",
            cpp_sources=[_CPP_SRC],
            cuda_sources=[_CUDA_SRC],
            functions=["eagle_step_adam", "eagle_step_sgd"],
            extra_cuda_cflags=cuda_flags,
            verbose=False,
        )
    except Exception as e:  # nvcc がない・コンパイル失敗など
        warnings.warn(f"EAGLE fused CUDA カーネルのコンパイルに失敗: {e}\n"
                      f"ベクトル化フォールバック実装を使用します。")
        _compile_failed = True
    return _ext
