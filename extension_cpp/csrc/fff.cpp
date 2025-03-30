// #include <torch/extension.h>
// #include <torch/torch.h>
// #include <cmath>
// #include <vector>
// #include <omp.h>

// namespace extension_cpp {

// constexpr float SQRT_2_PI = 0.7978845608f;
// constexpr float APPROX_COEF = 0.044715f;

// at::Tensor fff_cpu(
//     const at::Tensor &x,
//     int64_t input_width,
//     int64_t output_width,
//     int64_t depth,
//     const at::Tensor &weights_in,
//     const at::Tensor &weights_out
// ) {
//     auto x_data = x.data_ptr<float>();
//     auto weights_in_data = weights_in.data_ptr<float>();
//     auto weights_out_data = weights_out.data_ptr<float>();
//     int64_t batch_size = x.size(0);
//     int64_t seq_length = x.size(1);
//     int64_t num_nodes = weights_in.size(0);
//     auto current_nodes = torch::zeros({batch_size, seq_length}, torch::kInt64);
//     auto all_logits = torch::empty({batch_size, seq_length, depth + 1}, torch::kFloat32);
//     auto all_nodes = torch::empty({batch_size, seq_length, depth + 1}, torch::kInt64);
//     auto current_nodes_data = current_nodes.data_ptr<int64_t>();
//     auto all_logits_data = all_logits.data_ptr<float>();
//     auto all_nodes_data = all_nodes.data_ptr<int64_t>();

//     for (int64_t d = 0; d <= depth; ++d) {
// #pragma omp parallel for
//         for (int64_t b = 0; b < batch_size; ++b) {
//             for (int64_t s = 0; s < seq_length; ++s) {
//                 int64_t idx = b * seq_length + s;
//                 int64_t node_idx = current_nodes_data[idx];
//                 all_nodes_data[idx * (depth + 1) + d] = node_idx;
//                 const float* x_ptr = x_data + idx * input_width;
//                 const float* w_ptr = weights_in_data + node_idx * input_width;
//                 float score = 0.f;
//                 for (int64_t j = 0; j < input_width; ++j) {
//                     score += x_ptr[j] * w_ptr[j];
//                 }
//                 all_logits_data[idx * (depth + 1) + d] = score;
//                 current_nodes_data[idx] = node_idx * 2 + (score >= 0.f ? 2 : 1);
//             }
//         }
//     }

//     auto result = torch::zeros({batch_size, seq_length, output_width}, torch::kFloat32);
//     auto result_data = result.data_ptr<float>();
//     int64_t total = batch_size * seq_length * (depth + 1);
//     std::vector<float> gelu(total);
//     for (int64_t i = 0; i < total; ++i) {
//         float val = all_logits_data[i];
//         gelu[i] = 0.5f * val * (1.f + std::tanh(SQRT_2_PI * (val + APPROX_COEF * std::pow(val, 3))));
//     }

//     for (int64_t b = 0; b < batch_size; ++b) {
//         for (int64_t s = 0; s < seq_length; ++s) {
//             int64_t base = b * seq_length + s;
//             for (int64_t d = 0; d <= depth; ++d) {
//                 int64_t node_idx = all_nodes_data[base * (depth + 1) + d];
//                 float gelu_val = gelu[base * (depth + 1) + d];
//                 for (int64_t j = 0; j < output_width; ++j) {
//                     result_data[base * output_width + j] += gelu_val * weights_out_data[node_idx + j * num_nodes];
//                 }
//             }
//         }
//     }
//     return result;
// }

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

// TORCH_LIBRARY(extension_cpp, m) {
//     m.def("fff(Tensor x, int input_width, int output_width, int depth, Tensor weights_in, Tensor weights_out) -> Tensor");
// }

// TORCH_LIBRARY_IMPL(extension_cpp, CPU, m) {
//     m.impl("fff", &fff_cpu);
// }

// }

////////////////////////////
//// FFF L1
////////////////////////////

// #include <torch/extension.h>
// #include "mkl.h"
// #include <cmath>
// #include <cstddef>

// namespace extension_cpp
// {
//     at::Tensor fff_cpu(
//         const at::Tensor &x,
//         int64_t input_width,
//         int64_t output_width,
//         int64_t depth,
//         const at::Tensor &weights_in,
//         const at::Tensor &weights_out)
//     {
//         TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
//         TORCH_CHECK(weights_in.is_contiguous(), "weights_in must be contiguous");
//         TORCH_CHECK(weights_out.is_contiguous(), "weights_out must be contiguous");
//         TORCH_CHECK(x.dtype() == at::kFloat, "x must be float");
//         TORCH_CHECK(weights_in.dtype() == at::kFloat, "weights_in must be float");
//         TORCH_CHECK(weights_out.dtype() == at::kFloat, "weights_out must be float");
//         TORCH_CHECK(output_width == input_width, "input and output width must be equal");
        
//         auto original_size = x.sizes();
//         auto x_2d = x.view({-1, input_width});
//         int k = x_2d.size(0);
//         int m = input_width;
//         auto out_2d = torch::zeros({k, m}, x.options());
//         size_t *current_nodes = (size_t *)mkl_calloc(k, sizeof(size_t), 64);
//         float sqrt2 = std::sqrt(2);
//         float *IN = x_2d.data_ptr<float>();
//         float *W1_ptr = weights_in.data_ptr<float>();
//         float *W2_ptr = weights_out.data_ptr<float>();
//         float *OUT = out_2d.data_ptr<float>();
//         int n = weights_in.size(0);
       
//         for (int d = 0; d < depth; ++d)
//         {
//             float *mi = IN;
//             for (int i = 0; i < k; ++i)
//             {
//                 float val = cblas_sdot_64(m, mi, 1, W1_ptr + (current_nodes[i] * m), 1);
//                 val = val * std::erfc(-val / sqrt2) / 2;
//                 for (int j = 0; j < m; ++j)
//                 {
//                     OUT[i * m + j] += val * W2_ptr[current_nodes[i] + j * n];
//                 }
//                 current_nodes[i] = 2 * current_nodes[i] + 1 + (val > 0.f ? 1 : 0);
//                 mi += m;
//             }
//         }
//         mkl_free(current_nodes);
//         return out_2d.view(original_size);
//     }

//     PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

//     TORCH_LIBRARY(extension_cpp, m)
//     {
//         m.def("fff(Tensor x, int input_width, int output_width, int depth, Tensor weights_in, Tensor weights_out) -> Tensor");
//     }

//     TORCH_LIBRARY_IMPL(extension_cpp, CPU, m)
//     {
//         m.impl("fff", &fff_cpu);
//     }
// }


////////////////////////////
//// FFF L1 (Optimized)
////////////////////////////

// #include <torch/extension.h>
// #include "mkl.h"
// #include <cmath>
// #include <cstddef>
// #ifdef _OPENMP
// #include <omp.h>
// #endif

// namespace extension_cpp
// {
//     at::Tensor fff_cpu(const at::Tensor &x, int64_t input_width, int64_t output_width, int64_t depth, const at::Tensor &weights_in, const at::Tensor &weights_out)
//     {
//         TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
//         TORCH_CHECK(weights_in.is_contiguous(), "weights_in must be contiguous");
//         TORCH_CHECK(weights_out.is_contiguous(), "weights_out must be contiguous");
//         TORCH_CHECK(x.dtype() == at::kFloat, "x must be float");
//         TORCH_CHECK(weights_in.dtype() == at::kFloat, "weights_in must be float");
//         TORCH_CHECK(weights_out.dtype() == at::kFloat, "weights_out must be float");
//         TORCH_CHECK(output_width == input_width, "input and output width must be equal");

//         auto original_size = x.sizes();
//         auto x_2d = x.view({-1, input_width});
//         int k = x_2d.size(0);
//         int m = input_width;
//         auto out_2d = torch::zeros({k, m}, x.options());
//         size_t *current_nodes = (size_t *)mkl_calloc(k, sizeof(size_t), 64);
//         float sqrt2 = std::sqrt(2);
//         float *IN = x_2d.data_ptr<float>();
//         float *W1_ptr = weights_in.data_ptr<float>();
//         float *W2_ptr = weights_out.data_ptr<float>();
//         float *OUT = out_2d.data_ptr<float>();
//         int n = weights_in.size(0);

//         for (int d = 0; d < depth; ++d)
//         {
// #pragma omp parallel for
//             for (int i = 0; i < k; ++i)
//             {
//                 float *mi = IN + i * m;
//                 int current_index = current_nodes[i];
//                 float val = cblas_sdot_64(m, mi, 1, W1_ptr + (current_index * m), 1);
//                 val = val * std::erfc(-val / sqrt2) / 2;
//                 cblas_saxpy(m, val, W2_ptr + current_index, n, OUT + i * m, 1);
//                 current_nodes[i] = 2 * current_index + 1 + (val > 0.f ? 1 : 0);
//             }
//         }
//         mkl_free(current_nodes);
//         return out_2d.view(original_size);
//     }

//     PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

//     TORCH_LIBRARY(extension_cpp, m)
//     {
//         m.def("fff(Tensor x, int input_width, int output_width, int depth, Tensor weights_in, Tensor weights_out) -> Tensor");
//     }

//     TORCH_LIBRARY_IMPL(extension_cpp, CPU, m)
//     {
//         m.impl("fff", &fff_cpu);
//     }
// }



////////////////////////////
//// FFF L2
////////////////////////////

// #include <torch/extension.h>
// #include "mkl.h"
// #include <cmath>
// #include <cstddef>
// #include <algorithm>

// namespace extension_cpp
// {

//     at::Tensor fff_cpu(
//         const at::Tensor &x,
//         int64_t input_width,
//         int64_t output_width,
//         int64_t depth,
//         const at::Tensor &weights_in,
//         const at::Tensor &weights_out)
//     {
//         TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
//         TORCH_CHECK(weights_in.is_contiguous(), "weights_in must be contiguous");
//         TORCH_CHECK(weights_out.is_contiguous(), "weights_out must be contiguous");
//         TORCH_CHECK(x.dtype() == at::kFloat, "x must be float");
//         TORCH_CHECK(weights_in.dtype() == at::kFloat, "weights_in must be float");
//         TORCH_CHECK(weights_out.dtype() == at::kFloat, "weights_out must be float");
//         int k = x.size(0);
//         int m = input_width;
//         auto out = torch::zeros({k, output_width}, x.options());
//         size_t *current_nodes = (size_t *)mkl_calloc(k, sizeof(size_t), 64);
//         float *intermed = (float *)mkl_malloc(k * sizeof(float), 64);
//         float *intermed2 = (float *)mkl_malloc(k * sizeof(float), 64);
//         CBLAS_TRANSPOSE *transpose_instructions = (CBLAS_TRANSPOSE *)mkl_malloc(k * sizeof(CBLAS_TRANSPOSE), 64);
//         std::fill_n(transpose_instructions, k, CblasNoTrans);
//         MKL_INT64 *m_array = (MKL_INT64 *)mkl_malloc(k * sizeof(MKL_INT64), 64);
//         std::fill_n(m_array, k, 1LL);
//         MKL_INT64 *n_array = (MKL_INT64 *)mkl_malloc(k * sizeof(MKL_INT64), 64);
//         std::fill_n(n_array, k, (MKL_INT64)m);
//         MKL_INT64 *incs = (MKL_INT64 *)mkl_malloc(k * sizeof(MKL_INT64), 64);
//         std::fill_n(incs, k, 1LL);
//         float *alpha_array = (float *)mkl_malloc(k * sizeof(float), 64);
//         std::fill_n(alpha_array, k, 1.f);
//         float *beta_array = (float *)mkl_malloc(k * sizeof(float), 64);
//         std::fill_n(beta_array, k, 0.f);
//         float **w1_pointers = (float **)mkl_malloc(k * sizeof(float *), 64);
//         for (int i = 0; i < k; ++i)
//         {
//             w1_pointers[i] = weights_in.data_ptr<float>() + current_nodes[i] * m;
//         }
//         float **w2_pointers = (float **)mkl_malloc(k * sizeof(float *), 64);
//         float **in_pointers = (float **)mkl_malloc(k * sizeof(float *), 64);
//         float *IN = x.data_ptr<float>();
//         for (int i = 0; i < k; ++i)
//         {
//             in_pointers[i] = IN + i * m;
//         }
//         float **out_pointers = (float **)mkl_malloc(k * sizeof(float *), 64);
//         float *OUT = out.data_ptr<float>();
//         for (int i = 0; i < k; ++i)
//         {
//             out_pointers[i] = OUT + i * output_width;
//         }
//         float **intermed_pointers = (float **)mkl_malloc(k * sizeof(float *), 64);
//         for (int i = 0; i < k; ++i)
//         {
//             intermed_pointers[i] = intermed + i;
//         }
//         for (int d = 0; d < depth; ++d)
//         {
//             cblas_sgemv_batch_64(
//                 CblasRowMajor,
//                 transpose_instructions,
//                 m_array,
//                 n_array,
//                 alpha_array,
//                 (const float **)in_pointers,
//                 n_array,
//                 (const float **)w1_pointers,
//                 incs,
//                 beta_array,
//                 intermed_pointers,
//                 incs,
//                 k,
//                 m_array);
//             for (int i = 0; i < k; ++i)
//             {
//                 w2_pointers[i] = weights_out.data_ptr<float>() + current_nodes[i] * m;
//                 current_nodes[i] = 2 * current_nodes[i] + 1 + (intermed[i] > 0.f ? 1 : 0);
//                 w1_pointers[i] = weights_in.data_ptr<float>() + current_nodes[i] * m;
//             }
//             vsCdfNorm(k, intermed, intermed2);
//             vsMul(k, intermed, intermed2, intermed);
//             cblas_saxpy_batch_64(
//                 n_array,
//                 alpha_array,
//                 (const float **)w2_pointers,
//                 incs,
//                 out_pointers,
//                 incs,
//                 k,
//                 m_array);
//         }
//         mkl_free(current_nodes);
//         mkl_free(intermed);
//         mkl_free(intermed2);
//         mkl_free(transpose_instructions);
//         mkl_free(m_array);
//         mkl_free(n_array);
//         mkl_free(incs);
//         mkl_free(alpha_array);
//         mkl_free(beta_array);
//         mkl_free(w1_pointers);
//         mkl_free(w2_pointers);
//         mkl_free(in_pointers);
//         mkl_free(intermed_pointers);
//         return out;
//     }

//     PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

//     TORCH_LIBRARY(extension_cpp, m)
//     {
//         m.def("fff(Tensor x, int input_width, int output_width, int depth, Tensor weights_in, Tensor weights_out) -> Tensor");
//     }

//     TORCH_LIBRARY_IMPL(extension_cpp, CPU, m)
//     {
//         m.impl("fff", &fff_cpu);
//     }

// }

////////////////////////////
//// FFF L2 with Caching
////////////////////////////

// #include <torch/extension.h>
// #include "mkl.h"
// #include <cmath>
// #include <cstddef>
// #include <algorithm>
// #include <memory>

// namespace {

// // A helper class that caches MKL buffers for a given batch size (\(k\)) and input width (\(m\))
// struct BufferCache {
//     int batch_size;
//     int m;
//     size_t* current_nodes;
//     float* intermed;
//     float* intermed2;
//     CBLAS_TRANSPOSE* transpose_instructions;
//     MKL_INT64* m_array;
//     MKL_INT64* n_array;
//     MKL_INT64* incs;
//     float* alpha_array;
//     float* beta_array;
//     float** w1_pointers;
//     float** w2_pointers;
//     float** in_pointers;
//     float** out_pointers;
//     float** intermed_pointers;

//     BufferCache(int batch_size_, int m_) : batch_size(batch_size_), m(m_) {
//         current_nodes = (size_t*)mkl_calloc(batch_size, sizeof(size_t), 64);
//         intermed = (float*)mkl_malloc(batch_size * sizeof(float), 64);
//         intermed2 = (float*)mkl_malloc(batch_size * sizeof(float), 64);
//         transpose_instructions = (CBLAS_TRANSPOSE*)mkl_malloc(batch_size * sizeof(CBLAS_TRANSPOSE), 64);
//         std::fill_n(transpose_instructions, batch_size, CblasNoTrans);
//         m_array = (MKL_INT64*)mkl_malloc(batch_size * sizeof(MKL_INT64), 64);
//         std::fill_n(m_array, batch_size, 1LL);
//         n_array = (MKL_INT64*)mkl_malloc(batch_size * sizeof(MKL_INT64), 64);
//         std::fill_n(n_array, batch_size, (MKL_INT64)m);
//         incs = (MKL_INT64*)mkl_malloc(batch_size * sizeof(MKL_INT64), 64);
//         std::fill_n(incs, batch_size, 1LL);
//         alpha_array = (float*)mkl_malloc(batch_size * sizeof(float), 64);
//         std::fill_n(alpha_array, batch_size, 1.f);
//         beta_array = (float*)mkl_malloc(batch_size * sizeof(float), 64);
//         std::fill_n(beta_array, batch_size, 0.f);
//         w1_pointers = (float**)mkl_malloc(batch_size * sizeof(float*), 64);
//         w2_pointers = (float**)mkl_malloc(batch_size * sizeof(float*), 64);
//         in_pointers = (float**)mkl_malloc(batch_size * sizeof(float*), 64);
//         out_pointers = (float**)mkl_malloc(batch_size * sizeof(float*), 64);
//         intermed_pointers = (float**)mkl_malloc(batch_size * sizeof(float*), 64);
//     }

//     ~BufferCache() {
//         mkl_free(current_nodes);
//         mkl_free(intermed);
//         mkl_free(intermed2);
//         mkl_free(transpose_instructions);
//         mkl_free(m_array);
//         mkl_free(n_array);
//         mkl_free(incs);
//         mkl_free(alpha_array);
//         mkl_free(beta_array);
//         mkl_free(w1_pointers);
//         mkl_free(w2_pointers);
//         mkl_free(in_pointers);
//         mkl_free(out_pointers);
//         mkl_free(intermed_pointers);
//     }
// };

// // Get a cached buffer for the current dimensions. Reuse if possible.
// BufferCache& getBufferCache(int batch_size, int m) {
//     static thread_local std::unique_ptr<BufferCache> cache;
//     if (!cache || cache->batch_size != batch_size || cache->m != m) {
//         cache.reset(new BufferCache(batch_size, m));
//     }
//     return *cache;
// }

// } // anonymous namespace

// namespace extension_cpp {

// at::Tensor fff_cpu(
//     const at::Tensor &x,
//     int64_t input_width,
//     int64_t output_width,
//     int64_t depth,
//     const at::Tensor &weights_in,
//     const at::Tensor &weights_out)
// {
//     TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
//     TORCH_CHECK(weights_in.is_contiguous(), "weights_in must be contiguous");
//     TORCH_CHECK(weights_out.is_contiguous(), "weights_out must be contiguous");
//     TORCH_CHECK(x.dtype() == at::kFloat, "x must be float");
//     TORCH_CHECK(weights_in.dtype() == at::kFloat, "weights_in must be float");
//     TORCH_CHECK(weights_out.dtype() == at::kFloat, "weights_out must be float");

//     int k = x.size(0);
//     int m = input_width;
//     auto out = torch::zeros({k, output_width}, x.options());

//     // Retrieve the cached buffers (or allocate new ones if dimensions differ)
//     BufferCache& cache = getBufferCache(k, m);
//     // Reset current_nodes for each call
//     std::fill_n(cache.current_nodes, k, 0);

//     float* IN = x.data_ptr<float>();
//     float* OUT = out.data_ptr<float>();

//     // Set up pointers based on the input arrays
//     for (int i = 0; i < k; ++i) {
//         cache.in_pointers[i] = IN + i * m;
//         cache.out_pointers[i] = OUT + i * output_width;
//         cache.intermed_pointers[i] = cache.intermed + i;
//         // w1 pointers start at offset 0 (current_nodes[i] is zero)
//         cache.w1_pointers[i] = weights_in.data_ptr<float>() + cache.current_nodes[i] * m;
//     }

//     for (int d = 0; d < depth; ++d) {
//         cblas_sgemv_batch_64(
//             CblasRowMajor,
//             cache.transpose_instructions,
//             cache.m_array,
//             cache.n_array,
//             cache.alpha_array,
//             (const float**)cache.in_pointers,
//             cache.n_array,
//             (const float**)cache.w1_pointers,
//             cache.incs,
//             cache.beta_array,
//             cache.intermed_pointers,
//             cache.incs,
//             k,
//             cache.m_array);

//         for (int i = 0; i < k; ++i) {
//             cache.w2_pointers[i] = weights_out.data_ptr<float>() + cache.current_nodes[i] * m;
//             cache.current_nodes[i] = 2 * cache.current_nodes[i] + 1 + (cache.intermed[i] > 0.f ? 1 : 0);
//             cache.w1_pointers[i] = weights_in.data_ptr<float>() + cache.current_nodes[i] * m;
//         }

//         vsCdfNorm(k, cache.intermed, cache.intermed2);
//         vsMul(k, cache.intermed, cache.intermed2, cache.intermed);

//         cblas_saxpy_batch_64(
//             cache.n_array,
//             cache.alpha_array,
//             (const float**)cache.w2_pointers,
//             cache.incs,
//             cache.out_pointers,
//             cache.incs,
//             k,
//             cache.m_array);
//     }
//     return out;
// }

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//     m.def("fff", &fff_cpu, "Fast FFF CPU implementation");
// }

// TORCH_LIBRARY(extension_cpp, m) {
//     m.def("fff(Tensor x, int input_width, int output_width, int depth, Tensor weights_in, Tensor weights_out) -> Tensor");
// }

// TORCH_LIBRARY_IMPL(extension_cpp, CPU, m) {
//     m.impl("fff", &fff_cpu);
// }

// } // namespace extension_cpp




////////////////////////////
//// FFF L2 (O3-mini-high optimized)
////////////////////////////

#include <torch/extension.h>
#include "mkl.h"
#include <cmath>
#include <cstddef>
#include <algorithm>

namespace extension_cpp
{
    at::Tensor fff_cpu(const at::Tensor &x, int64_t input_width, int64_t output_width, int64_t depth, const at::Tensor &weights_in, const at::Tensor &weights_out)
    {
        TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
        TORCH_CHECK(weights_in.is_contiguous(), "weights_in must be contiguous");
        TORCH_CHECK(weights_out.is_contiguous(), "weights_out must be contiguous");
        TORCH_CHECK(x.dtype() == at::kFloat, "x must be float");
        TORCH_CHECK(weights_in.dtype() == at::kFloat, "weights_in must be float");
        TORCH_CHECK(weights_out.dtype() == at::kFloat, "weights_out must be float");
        TORCH_CHECK(output_width == input_width, "input and output width must be equal");

        auto original_size = x.sizes();
        auto x_2d = x.view({-1, input_width});
        int batch_size = x_2d.size(0);
        int hidden_dim = input_width;
        auto out_2d = torch::zeros({batch_size, hidden_dim}, x.options());

        float* IN = x_2d.data_ptr<float>();
        float* W1 = weights_in.data_ptr<float>();
        float* W2 = weights_out.data_ptr<float>();
        float* OUT = out_2d.data_ptr<float>();

        size_t* current_nodes = (size_t*)mkl_calloc(batch_size, sizeof(size_t), 64);
        float* intermed = (float*)mkl_malloc(batch_size * sizeof(float), 64);
        float* intermed2 = (float*)mkl_malloc(batch_size * sizeof(float), 64);
        CBLAS_TRANSPOSE* transpose_instructions = (CBLAS_TRANSPOSE*)mkl_malloc(batch_size * sizeof(CBLAS_TRANSPOSE), 64);
        std::fill_n(transpose_instructions, batch_size, CblasNoTrans);
        
        MKL_INT64* m_array = (MKL_INT64*)mkl_malloc(batch_size * sizeof(MKL_INT64), 64);
        std::fill_n(m_array, batch_size, 1);
        MKL_INT64* n_array = (MKL_INT64*)mkl_malloc(batch_size * sizeof(MKL_INT64), 64);
        std::fill_n(n_array, batch_size, (MKL_INT64)hidden_dim);
        MKL_INT64* incs = (MKL_INT64*)mkl_malloc(batch_size * sizeof(MKL_INT64), 64);
        std::fill_n(incs, batch_size, (MKL_INT64)1);
        
        float* alpha_array = (float*)mkl_malloc(batch_size * sizeof(float), 64);
        std::fill_n(alpha_array, batch_size, 1.f);
        float* beta_array = (float*)mkl_malloc(batch_size * sizeof(float), 64);
        std::fill_n(beta_array, batch_size, 0.f);
        
        float** w1_pointers = (float**)mkl_malloc(batch_size * sizeof(float*), 64);
        for (int i = 0; i < batch_size; i++) {
            w1_pointers[i] = W1;
        }
        float** w2_pointers = (float**)mkl_malloc(batch_size * sizeof(float*), 64);
        for (int i = 0; i < batch_size; i++) {
            w2_pointers[i] = W2;
        }
        float** in_pointers = (float**)mkl_malloc(batch_size * sizeof(float*), 64);
        for (int i = 0; i < batch_size; ++i) {
            in_pointers[i] = IN + i * hidden_dim;
        }
        float** out_pointers = (float**)mkl_malloc(batch_size * sizeof(float*), 64);
        for (int i = 0; i < batch_size; ++i) {
            out_pointers[i] = OUT + i * hidden_dim;
        }
        float** intermed_pointers = (float**)mkl_malloc(batch_size * sizeof(float*), 64);
        for (int i = 0; i < batch_size; ++i) {
            intermed_pointers[i] = intermed + i;
        }

        for (int d = 0; d < depth; ++d)
        {
            cblas_sgemv_batch_64(
                CblasRowMajor,
                transpose_instructions,
                m_array,
                n_array,
                alpha_array,
                (const float**)in_pointers,
                n_array,
                (const float**)w1_pointers,
                incs,
                beta_array,
                intermed_pointers,
                incs,
                batch_size,
                m_array
            );
            for (int k = 0; k < batch_size; ++k) {
                w2_pointers[k] = W2 + hidden_dim * current_nodes[k];
                current_nodes[k] = 2 * current_nodes[k] + 1 + (intermed[k] > 0.f ? 1 : 0);
                w1_pointers[k] = W1 + hidden_dim * current_nodes[k];
            }
            vsCdfNorm(batch_size, intermed, intermed2);
            vsMul(batch_size, intermed, intermed2, intermed);
            cblas_saxpy_batch_64(
                n_array,
                alpha_array,
                (const float**)w2_pointers,
                incs,
                out_pointers,
                incs,
                batch_size,
                m_array
            );
        }

        mkl_free(current_nodes);
        mkl_free(intermed);
        mkl_free(intermed2);
        mkl_free(transpose_instructions);
        mkl_free(m_array);
        mkl_free(n_array);
        mkl_free(incs);
        mkl_free(alpha_array);
        mkl_free(beta_array);
        mkl_free(w1_pointers);
        mkl_free(w2_pointers);
        mkl_free(in_pointers);
        mkl_free(intermed_pointers);

        return out_2d.view(original_size);
    }

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
    {
        m.def("fff", &fff_cpu, "Optimized fff with batched MKL operations");
    }

    TORCH_LIBRARY(extension_cpp, m)
    {
        m.def("fff(Tensor x, int input_width, int output_width, int depth, Tensor weights_in, Tensor weights_out) -> Tensor");
    }

    TORCH_LIBRARY_IMPL(extension_cpp, CPU, m)
    {
        m.impl("fff", &fff_cpu);
    }
}
