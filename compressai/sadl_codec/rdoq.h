#pragma once
#include "common.h"
#include <future>
#include <mutex>
#include <sadl/model.h>
#include <thread>

using namespace std;
constexpr float kRatioRateToDoRDOQ = 0.25; //
atomic<int> counter;

struct CodecProperties {
  int down;             // downscale by encoder
  int receiptive_field; // in image
  double lambda;
};

double psnr(double mse) { return -10. * log10(mse / (255. * 255.)); }

double mse(const Image &im0, const Image &im1) {
  double s = 0.; // approx RGB PSNR
  for (int k = 0; k < im0.size[0] * im0.size[1] * 3; ++k) {
    float e = (float)im0.data_[k] - im1.data_[k];
    s += e * e;
  }
  return s / (im0.size[0] * im0.size[1] * 3);
}

double psnrReconstructed(sadl::Model<Type> &decoder, const sadl::Tensor<LatentType> &latent, const Image &im) {
  std::vector<sadl::Tensor<Type>> inputs{1};
  inputs[0].resize(latent.dims());
  for (int k = 0; k < latent.size(); ++k)
    inputs[0][k] = latent[k];

  offsetLatent(inputs[0]);

  if (!decoder.apply(inputs)) {
    std::cerr << "[ERROR] Pb inference" << endl;
    exit(-1);
  }
  const double e = mse(im, toImage(decoder.result(), im.size));
  return psnr(e);
}

double cost(int v, const Cdf &cdf) {
  const double Q = 1. / (1 << Cdf::precision);
  int idx = v - cdf.min_val;
  double p = (cdf.cproba[idx + 1] - cdf.cproba[idx]) * Q;
  return -log2(p);
}

double getApproximateChannelCost(const sadl::Tensor<Type> &t, int c) {
  double H = 0.;
  const auto &cdf = kCdfs[c][0];
  for (int i = 0; i < t.dims()[1]; ++i)
    for (int j = 0; j < t.dims()[2]; ++j) {
      // const auto &cdf=cdfs[getContext(t,k,kprev,i,j)];
      int v = t(0, i, j, c); // no round because we control the value to be int
      if (v <= cdf.min_val || v >= cdf.max_val) {
        std::cout << "[ERROR] value out of range: " << v << " [" << cdf.min_val << ' ' << cdf.max_val << "]" << endl;
        exit(-1);
      }
      H += cost(v, cdf);
    }
  return H;
}

double SE_part(const Image &org, int i0, int j0, int rf, const Image &cropped) {
  const int ri0 = cropped.size[0] / 2;
  const int rj0 = cropped.size[1] / 2;
  double se = 0.;
  for (int i = -rf; i <= rf; ++i)
    for (int j = -rf; j <= rf; ++j) {
      if (i0 + i >= 0 && i0 + i < org.size[0] && j0 + j >= 0 && j0 + j < org.size[1]) {
        for (int k = 0; k < 3; ++k) {
          float e = (float)org(i0 + i, j0 + j, k) - cropped(ri0 + i, rj0 + j, k);
          se += e * e;
        }
      }
    }
  return se / 3.; // normalize by image size
}

// note: approx R by using kCdfs
// R/nb_pix+L DR/nb_pix < R_newR/nb_pix + L D_newR/nb_pix
// R = R_c_i_j + R_other
// R_c_i_j+L D < R_new_c_i_j + L D_new
// D = SE = (SE_area+SE_other)
// R_c_i_j+L SE_area < R_new_c_i_j + L SE_new_are

// note: MT very messy: take into account update randmly (do not wait for processing of other channels), so not causal and not deterministic
void rdoqChannel(sadl::Tensor<Type> &t, int c, int cprev, const Image &im, const CodecProperties &prop, double totR, bool full_ctx,
                 bool mt) {
  double Rc = getApproximateChannelCost(t, c); // approximate, initial rate
  if (Rc < totR / kNbCdfs * kRatioRateToDoRDOQ)
    return;
  double D{}, R{};
  const int d = t.dims()[3];
  const int scale = (1 << prop.down);
  const int rf_latent = ceil((float)(prop.receiptive_field * 2) / scale);
  thread_local static auto decoder = loadDecoder();
  thread_local static std::vector<sadl::Tensor<Type>> inputs{1};
  thread_local static sadl::Tensor<Type> bak_input; // to avoid copy
  thread_local static bool once = false;
  if (!once) {
    sadl::Dimensions dims({1, rf_latent * 2 + 1, rf_latent * 2 + 1, d});
    inputs[0].resize(dims);
    bak_input.resize(dims);
    decoder.init(inputs);
    once = true;
  }
  constexpr int deltas[] = {1, -1};
  static mutex mtx;

  // to see: why cropped on border not same result as original
  int cpt = 0;
  int gains = 0;

  const int cpt_tot = (t.dims()[1] - 2 * rf_latent) * (t.dims()[2] - 2 * rf_latent);
  for (int i0 = rf_latent; i0 < t.dims()[1] - rf_latent; ++i0)
    for (int j0 = rf_latent; j0 < t.dims()[2] - rf_latent; ++j0, cpt++) {
      if (verbose && full_ctx) {
        cout << "\rProcess channel " << c << ": " << (cpt * 100.) / (cpt_tot) << "%     " << gains << "   ";
        cout.flush();
      }
      // crop latent
      const Cdf &cdf = (full_ctx) ? getCdf(t, c, cprev, i0, j0) : kCdfs[c][0];
      const int most_probable = cdf.probable;
      int v_org = t(0, i0, j0, c); // nobody can write here: no mtx (but false sharing likely)
      int v_new = v_org;
      if (v_org == most_probable)
        continue; // early skip
      inputs[0].fill(0);
      if (mt)
        mtx.lock();
      for (int di = -rf_latent; di <= rf_latent; ++di)
        for (int dj = -rf_latent; dj <= rf_latent; ++dj) {
          if (t.in(0, i0 + di, j0 + dj, 0)) {
            for (int k = 0; k < d; ++k) {
              inputs[0](0, di + rf_latent, dj + rf_latent, k) = t(0, i0 + di, j0 + dj, k);
            }
          }
        }
      if (mt)
        mtx.unlock();
      offsetLatent(inputs[0]);
      bak_input = inputs[0];
      // compute initial distortion
      decoder.apply(inputs);
      swap(inputs[0], bak_input); // because sadl consume inputs
      const int i_org = i0 * scale + scale / 2;
      const int j_org = j0 * scale + scale / 2;
      Size s_output = {(uint16_t)decoder.result().dims()[1], (uint16_t)decoder.result().dims()[2]};
      R = cost(v_org, cdf);
      D = SE_part(im, i_org, j_org, prop.receiptive_field, toImage(decoder.result(), s_output));
      // const int sign=(most_probable-v_org)>0?1:-1; // prefer lower bitrate first
      for (auto delta : deltas) {
        //   int delta=sign*delta0;
        // if ((most_probable - v_org) * delta < 0) continue; // heuristic to lower the rate
        if (v_org + delta >= cdf.max_val)
          continue;
        if (v_org + delta <= cdf.min_val)
          continue;
        double cur_R = cost(v_org + delta, cdf);
        inputs[0](0, rf_latent, rf_latent, c) = v_org + delta;
        offsetValue(inputs[0](0, rf_latent, rf_latent, c),c);
        decoder.apply(inputs);
        double cur_D = SE_part(im, i_org, j_org, prop.receiptive_field, toImage(decoder.result(), s_output));
        if (cur_D * prop.lambda + cur_R < D * prop.lambda + R) {
          v_new = v_org + delta;

          if (verbose > 1 && full_ctx)
            cout << i0 << ' ' << j0 << ": " << v_org << "+" << delta << ": " << R << "+L*" << D << "=" << D * prop.lambda + R << " => "
                 << cur_R << "+L*" << cur_D << "=" << cur_D * prop.lambda + cur_R << endl;
          R = cur_R;
          D = cur_D;
          ++gains;
          //   break; // skip if found
        }
      }
      if (v_new != v_org) {
        if (mt)
          mtx.lock();
        t(0, i0, j0, c) = v_new;
        if (mt)
          mtx.unlock();
      }
    }
  if (verbose && full_ctx)
    cout << endl;
}

extern std::string compress(const sadl::Tensor<LatentType> &t, Size size);
extern sadl::Tensor<LatentType> asInt16(const sadl::Tensor<Type> &t);

sadl::Tensor<LatentType> rdoq(sadl::Model<Type> &decoder, const sadl::Tensor<LatentType> &latent, const Image &im,
                              const CodecProperties &prop, int nbPass) {
  sadl::Tensor<Type> t;
  t.resize(latent.dims());
  for (int k = 0; k < latent.size(); ++k)
    t[k] = latent[k];
  sadl::Tensor<LatentType> latent2;
  double totCost = compress(latent, im.size).size() * 8;
  for (int pass = 0; pass < nbPass; ++pass) {
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    int cprev = -1;
    for (int c0 = 0; c0 < kNbCdfs; ++c0) {
      int c = kOrder[c0];
      rdoqChannel(t, c, cprev, im, prop, totCost, true, false);
      if (verbose)
        cout << "Total: " << (c0 * 100.) / (kNbCdfs) << "%" << endl;
      cprev = c;
    }
    latent2 = asInt16(t);
    cout << "[INFO] PSNR: " << psnrReconstructed(decoder, latent2, im) << " dB" << endl;
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> cold = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "RDOQ: " << cold.count() << " s" << endl;
  }
  return latent2;
}

bool rdoqChannelMulti(sadl::Tensor<Type> &t, const vector<int> &ch, const vector<int> &chprev, const Image &im, const CodecProperties &prop, double totR) {

  for (int k=0;k<(int)ch.size();++k) {
    rdoqChannel(t, ch[k], chprev[k], im, prop, totR, true, true);
    counter++;
    cout << 100. * counter / kNbCdfs << "%     " << endl;
  }
  // cout<<"Thread finished"<<endl;
  return true;
}

void copy(const sadl::Tensor<Type> &src, int c, sadl::Tensor<Type> &tgt) {
  for (int i = 0; i < src.dims()[1]; ++i)
    for (int j = 0; j < src.dims()[2]; ++j)
      tgt(0, i, j, c) = src(0, i, j, c);
}

sadl::Tensor<LatentType> rdoq_mt(sadl::Model<Type> &decoder, const sadl::Tensor<LatentType> &latent, const Image &im,
                                 const CodecProperties &prop, int nbPass) {
  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  sadl::Tensor<Type> t;
  t.resize(latent.dims());
  for (int k = 0; k < latent.size(); ++k)
    t[k] = latent[k];

  const int nb = thread::hardware_concurrency() / 2;
  double totCost = compress(latent, im.size).size() * 8;
  vector<sadl::Tensor<Type>> ts;
  vector<vector<int>> channels;
  vector<vector<int>> channels_prev;
  vector<future<bool>> results;
  ts.resize(nb, t);
  channels.resize(nb);
  channels_prev.resize(nb);
  results.resize(nb);
  int idx = 0;
  for (int c0 = 0; c0 < kNbCdfs; ++c0) {
    int c = kOrder[c0];
    channels[idx].push_back(c);
    if (c0==0) channels_prev[idx].push_back(-1);
    else       channels_prev[idx].push_back(kOrder[c0-1]);
    idx = (idx + 1) % nb;
  }
  sadl::Tensor<LatentType> latent2;
  for (int pass = 0; pass < nbPass; ++pass) {
    counter = 0;
    for (int idx = 0; idx < nb; ++idx) {
      results[idx] = std::async(rdoqChannelMulti, ref(/*ts[idx]*/ t), ref(channels[idx]), ref(channels_prev[idx]), ref(im), prop, totCost);
    }
    for (int idx = 0; idx < nb; ++idx) {
      results[idx].wait();
      //    for (auto c : channels[idx]) {
      //      copy(ts[idx], c, t);
      //    }
    }
    latent2 = asInt16(t);
    totCost = compress(latent2, im.size).size() * 8;
    cout << "[INFO] R=" << totCost << " bits, PSNR: " << psnrReconstructed(decoder, latent2, im) << " dB" << endl;
  }
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> cold = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "RDOQ: " << cold.count() << " s" << endl;
  return latent2;
}
