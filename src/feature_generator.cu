/*
 * Copyright 2018-2019 Autoware Foundation. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "feature_generator_cuda.h"
// #include <cuda_runtime_api.h>
// #include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>


static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

#if __CUDA_ARCH__ < 600
__device__ double atomicAdd(double* address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*) address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val +
    __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}

__device__  float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}
#endif

__device__ double atomic_exch(double* addr, double val) {
  unsigned long long int *m_addr = (unsigned long long int*) addr;
  unsigned long long int old_val = 0;
  old_val = atomicExch(m_addr, __double_as_longlong(val));
  return __longlong_as_double(old_val);
}

__device__ float atomic_exch(float* addr, float val) {
  return atomicExch(addr, (val));
}

template<typename Dtype>
__global__ void MapKernel(const int n, const pcl::PointXYZI* pc,
               Dtype* max_height_data, Dtype* mean_height_data,
               Dtype* mean_intensity_data, Dtype* count_data,
               int* map_idx) {
  CUDA_KERNEL_LOOP(i, n) {
    int idx = map_idx[i];
    if (idx == -1) {
      continue;
    }
    Dtype pz = pc[i].z;
    Dtype pi = pc[i].intensity / 255.0;
    atomicMax(&max_height_data[idx], pz);
    atomicAdd(&mean_height_data[idx], pz);
    if (mean_intensity_data != nullptr) {
      atomicAdd(&mean_intensity_data[idx], pi);
    }
    atomicAdd(&count_data[idx], (Dtype) 1);
  }
}

template<typename Dtype>
__global__ void AverageKernel(const int n, Dtype* count_data,
                 Dtype* max_height_data, Dtype* mean_height_data,
                 Dtype* mean_intensity_data, Dtype* nonempty_data,
                 Dtype* log_table, const int max_log_num) {
  CUDA_KERNEL_LOOP(i, n) {
    if (count_data[i] < 1e-6) {
      max_height_data[i] = 0;
    } else {
      mean_height_data[i] /= count_data[i];
      if (mean_intensity_data != nullptr) {
        mean_intensity_data[i] /= count_data[i];
      }
      nonempty_data[i] = Dtype(1.0);
    }
    int count = static_cast<int>(count_data[i]);
    if (count < max_log_num) {
      count_data[i] = log_table[count];
    } else {
      count_data[i] = log(1.0 + count);
    }
  }
}

template<typename Dtype>
__global__ void TopIntensityKernel(const int n, Dtype* top_intensity,
                                   pcl::PointXYZI* pc, Dtype* max_height_data,
                                   int* map_idx) {
  if (top_intensity == nullptr) {
    return;
  }
  CUDA_KERNEL_LOOP(i, n) {
    int idx = map_idx[i];
    if (idx == -1) {
      continue;
    }
    Dtype pz = pc[i].z;
    Dtype pi = pc[i].intensity / 255.0;
    if (pz == max_height_data[idx]) {
      top_intensity[idx] = pi;
    }
  }
}

template <typename Dtype>
__global__ void SetKernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(i, n) {
    y[i] = alpha;
  }
}

bool FeatureGeneratorCuda::init(int range, int width,
                                int height, const bool use_constant_feature,
                                const bool use_intensity_feature) {
  range_ = range;
  width_ = width;
  height_ = height;

  min_height_ = -5.0;
  max_height_ = 5.0;

  log_table_.resize(256);
  for (size_t i = 0; i < log_table_.size(); ++i) {
    log_table_[i] = std::log1p(static_cast<float>(i));
  }

  int map_size = height_ * width_;
  // float *direction_data, *distance_data;
  // if (use_constant_feature && use_intensity_feature) {
  //   direction_data = in_feature_ptr + siz * ;
  //   distance_data = in_feature_ptr + siz * 6;
  // } else if (use_constant_feature) {
  //   direction_data = in_feature_ptr + siz * 3;
  //   distance_data = in_feature_ptr + siz * 4;
  // }

  std::vector<float> direction_data(map_size);
  std::vector<float> distance_data(map_size);
  if (use_constant_feature) {
    for (int row = 0; row < height_; ++row) {
      for (int col = 0; col < width_; ++col) {
        int idx = row * width_ + col;
        // * row <-> x, column <-> y
        float center_x = Pixel2Pc(row, height_, range_);
        float center_y = Pixel2Pc(col, width_, range_);
        constexpr double K_CV_PI = 3.1415926535897932384626433832795;
        // normaliztion. -0.5~0.5
        direction_data[idx] = static_cast<float>(
            std::atan2(center_y, center_x) / (2.0 * K_CV_PI));
        distance_data[idx] =
            static_cast<float>(std::hypot(center_x, center_y) / 60.0 - 0.5);
      }
    }
  // memory copy direction and distance features
  cudaMemcpy(direction_data_, direction_data.data(),
             direction_data.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(distance_data_, distance_data.data(),
             distance_data.size() * sizeof(float), cudaMemcpyHostToDevice);
  }


  return true;
}

float FeatureGeneratorCuda::logCount(int count) {
  if (count < static_cast<int>(log_table_.size())) {
    return log_table_[count];
  }
  return std::log(static_cast<float>(1 + count));
}

void FeatureGeneratorCuda::generate(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr &pc_ptr, float *max_height_data,
    const bool use_constant_feature, const bool use_intensity_feature) {

  std::cout << "g1" << std::endl;
  const auto &points = pc_ptr->points;
  std::cout << "g1" << std::endl;
  int map_size = height_ * width_;
  std::cout << "g1" << std::endl;
  int block_size = (map_size + kGPUThreadSize - 1) / kGPUThreadSize;
  // initializing max_height_data_ with -5
  std::cout << "g2" << std::endl;
  SetKernel<float><<<block_size, kGPUThreadSize>>>(map_size, -5.f,
                                                max_height_data);
  std::cout << "g3" << std::endl;

  // HANDLE_ERROR(cudaMemset(max_height_data + map_size, 0.f,
  //                         sizeof(float) * map_size));
  // HANDLE_ERROR(cudaMemset(max_height_data + map_size * 2, 0.f,
  //                            sizeof(float) * map_size));
  // HANDLE_ERROR(cudaMemset(max_height_data + map_size * 3, 0.f,
  //                            sizeof(float) * map_size));
  // if (use_intensity_feature) {
  //   HANDLE_ERROR(cudaMemset(max_height_data + map_size * 4, 0.f,
  //                              sizeof(float) * map_size));
  //   HANDLE_ERROR(cudaMemset(max_height_data + map_size * 5, 0.f,
  //                              sizeof(float) * map_size));
  // }

  HANDLE_ERROR(cudaMalloc((void**)&mean_height_data_, sizeof(float) * map_size));
  HANDLE_ERROR(cudaMalloc((void**)&max_height_data_, sizeof(float) * map_size));
  HANDLE_ERROR(cudaMalloc((void**)&count_data_, sizeof(float) * map_size));
  HANDLE_ERROR(cudaMalloc((void**)&nonempty_data_, sizeof(float) * map_size));
  HANDLE_ERROR(cudaMalloc((void**)&top_intensity_data_, sizeof(float) * map_size));
  HANDLE_ERROR(cudaMalloc((void**)&mean_intensity_data_, sizeof(float) * map_size));

  HANDLE_ERROR(cudaMemset(mean_height_data_, 0.f,
                             sizeof(float) * map_size));
  std::cout << "g3" << std::endl;
  HANDLE_ERROR(cudaMemset(count_data_, 0.f,
                             sizeof(float) * map_size));
  HANDLE_ERROR(cudaMemset(nonempty_data_, 0.f,
                             sizeof(float) * map_size));
  std::cout << "g4" << std::endl;
  if (use_intensity_feature) {
    HANDLE_ERROR(cudaMemset(top_intensity_data_, 0.f,
                               sizeof(float) * map_size));
    HANDLE_ERROR(cudaMemset(mean_intensity_data_, 0.f,
                               sizeof(float) * map_size));
  }

  size_t cloud_size = pc_ptr->size();
  map_idx_.resize(points.size());
  float inv_res_x =
      0.5 * static_cast<float>(width_) / static_cast<float>(range_);
  float inv_res_y =
      0.5 * static_cast<float>(height_) / static_cast<float>(range_);

  std::cout << "g5" << std::endl;
  for (size_t i = 0; i < points.size(); ++i) {
    if (points[i].z <= min_height_ || points[i].z >= max_height_) {
      map_idx_[i] = -1;
      continue;
    }
    int pos_x = F2I(points[i].y, range_, inv_res_x);
    int pos_y = F2I(points[i].x, range_, inv_res_y);
    if (pos_x >= width_ || pos_x < 0 || pos_y >= height_ || pos_y < 0) {
      map_idx_[i] = -1;
      continue;
    }
    map_idx_[i] = pos_y * width_ + pos_x;
  }

  std::cout << "g6" << std::endl;
  // copy cloud data and map_idx from CPU to GPU memory
  if (cloud_size > pc_gpu_size_) {
    HANDLE_ERROR(cudaFree(pc_gpu_));
    HANDLE_ERROR(cudaMalloc(reinterpret_cast<void **>(&pc_gpu_),
                            int(cloud_size) * sizeof(pcl::PointXYZI)));
    std::cout << "g6" << std::endl;
    pc_gpu_size_ = cloud_size;
    HANDLE_ERROR(cudaFree(map_idx_gpu_));
    HANDLE_ERROR(cudaMalloc(reinterpret_cast<void **>(&map_idx_gpu_),
                            int(cloud_size) * sizeof(int)));
  }

  std::cout << pc_ptr->front() << std::endl;
  std::cout << &(pc_ptr->front()) << std::endl;
  std::cout << sizeof(pcl::PointXYZI) << std::endl;
  std::cout << int(cloud_size) << std::endl;



  HANDLE_ERROR(cudaMemcpy(map_idx_gpu_, map_idx_.data(),
                             sizeof(int) * cloud_size, cudaMemcpyHostToDevice));
  std::cout << "g7" << std::endl;
  HANDLE_ERROR(cudaMemcpy(pc_gpu_, &(pc_ptr->front()),
                          sizeof(pcl::PointXYZI) * int(cloud_size),
                             cudaMemcpyHostToDevice));


  std::cout << "g8" << std::endl;
  MapKernel<float><<<block_size, kGPUThreadSize>>>(cloud_size, pc_gpu_,
                                                   max_height_data_, mean_height_data_, mean_intensity_data_,
                                                   count_data_, map_idx_gpu_);
  std::cout << "g9" << std::endl;
  TopIntensityKernel<float><<<block_size, kGPUThreadSize>>>(cloud_size,
        top_intensity_data_, pc_gpu_, max_height_data_,
        map_idx_gpu_);

  std::cout << "g10" << std::endl;
  HANDLE_ERROR( cudaMemcpy(in_feature_, max_height_data_, sizeof(float) * map_size, cudaMemcpyDeviceToHost ) );
  HANDLE_ERROR( cudaMemcpy(in_feature_ + map_size, mean_height_data_, sizeof(float) * map_size , cudaMemcpyDeviceToHost ) );
  HANDLE_ERROR( cudaMemcpy(in_feature_ + map_size * 2, count_data_, sizeof(float) * map_size , cudaMemcpyDeviceToHost ) );
  HANDLE_ERROR( cudaMemcpy(in_feature_ + map_size * 3, direction_data_, sizeof(float) * map_size , cudaMemcpyDeviceToHost ) );
  HANDLE_ERROR( cudaMemcpy(in_feature_ + map_size * 4, top_intensity_data_, sizeof(float) * map_size , cudaMemcpyDeviceToHost ) );
  HANDLE_ERROR( cudaMemcpy(in_feature_ + map_size * 5, mean_intensity_data_, sizeof(float) * map_size , cudaMemcpyDeviceToHost ) );
  HANDLE_ERROR( cudaMemcpy(in_feature_ + map_size * 5, mean_intensity_data_, sizeof(float)* map_size, cudaMemcpyDeviceToHost ) );
  HANDLE_ERROR( cudaMemcpy(in_feature_ + map_size * 6, distance_data_, sizeof(float)* map_size, cudaMemcpyDeviceToHost ) );
  HANDLE_ERROR( cudaMemcpy(in_feature_ + map_size * 7, nonempty_data_, sizeof(float)* map_size, cudaMemcpyDeviceToHost ) );
  std::cout << "g11" << std::endl;
}
