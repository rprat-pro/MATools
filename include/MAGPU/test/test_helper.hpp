#pragma once 

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess) 
  {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

namespace test_helper
{
  template<typename T, MATools::MAGPU::MEM_MODE MM, MATools::MAGPU::GPU_TYPE GT>
    struct create
    {
      T* operator()(unsigned int N);
      T* operator()(T value, unsigned int N);
    };

  template<typename T, MATools::MAGPU::GPU_TYPE GT>
    struct create<T, MATools::MAGPU::MEM_MODE::CPU, GT>
    {
      T* operator()(unsigned int N)
      {
	T* ret;
	ret = new T[N];
	return ret;
      }

      T* operator()(T value, unsigned int N)
      {
	T* ret;
	ret = new T[N];
	for(int it = 0 ; it < N ; it++)
	{
	  ret[it] = value;
	}
	return ret;
      }
    };

  template<typename T>
    struct create<T, MATools::MAGPU::MEM_MODE::GPU, MATools::MAGPU::GPU_TYPE::SERIAL>
    {
      T* operator()(unsigned int N)
      {
	T* ret;
	ret = new T[N];
	return ret;
      }

      T* operator()(T value, unsigned int N)
      {
	T* ret;
	ret = new T[N];
	for(int it = 0 ; it < N ; it++)
	{
	  ret[it] = value;
	}
	return ret;
      }
    };

  template<typename T, MATools::MAGPU::MEM_MODE MM, MATools::MAGPU::GPU_TYPE GT>
    struct destroy
    {
      void operator()(T* ret);
    };

  template<typename T, MATools::MAGPU::GPU_TYPE GT>
    struct destroy<T, MATools::MAGPU::MEM_MODE::CPU, GT>
    {
      void operator()(T* ret)
      {
	delete ret;
      }
    };

  template<typename T>
    struct destroy<T, MATools::MAGPU::MEM_MODE::GPU, MATools::MAGPU::GPU_TYPE::SERIAL>
    {
      void operator()(T* ret)
      {
	std::cout << " debug destroy<T, MATools::MAGPU::MEM_MODE::GPU, MATools::MAGPU::GPU_TYPE::SERIAL> " << std::endl;
	delete ret;
      }
    };

  template<typename T, MATools::MAGPU::MEM_MODE MM, MATools::MAGPU::GPU_TYPE GT>
    struct copier
    {
      void operator()(T* dst, T* src, int N);
    };

  template<typename T, MATools::MAGPU::GPU_TYPE GT>
    struct copier<T, MATools::MAGPU::MEM_MODE::CPU, GT>
    {
      void operator()(T* dst, T* src, int N)
      {
	std::copy(src, src + N, dst);
      }
    };

  template<typename T>
    struct copier<T, MATools::MAGPU::MEM_MODE::GPU, MATools::MAGPU::GPU_TYPE::SERIAL>
    {
      void operator()(T* dst, T* src, int N)
      {
	std::copy(src, src + N, dst);
      }
    };

  template<MATools::MAGPU::MEM_MODE MM, MATools::MAGPU::GPU_TYPE GT>
    struct mini_runner
    {
      template<typename Func, typename... Args>
	void operator()(Func& fun, Args&&... args)
	{
	  fun(std::forward<Args>(args)...);
	}
    };

  template<MATools::MAGPU::GPU_TYPE GT>
    struct mini_runner<MATools::MAGPU::MEM_MODE::CPU, GT>
    {
      template<typename Func, typename... Args>
	void operator()(Func& fun, Args&&... args)
	{
	  fun(std::forward<Args>(args)...);
	}
    };

  template<>
    struct mini_runner<MATools::MAGPU::MEM_MODE::CPU, MATools::MAGPU::GPU_TYPE::SERIAL>
    {
      template<typename Func, typename... Args>
	void operator()(Func& fun, Args&&... args)
	{
	  fun(std::forward<Args>(args)...);
	}
    };
#ifdef __CUDA__
  template<typename T>
    __global__
    void fill(T* const ptr, T value, unsigned int N)
    {
      unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;
      if(idx < N)
	ptr[idx] = value;
    }
  template<typename T>
    struct create<T, MATools::MAGPU::MEM_MODE::GPU, MATools::MAGPU::GPU_TYPE::CUDA>
    {
      T* operator()(unsigned int N)
      {
	std::cout << " create<T, MATools::MAGPU::MEM_MODE::GPU, MATools::MAGPU::GPU_TYPE::CUDA> :: " << N << std::endl;
	T* ret;
	cudaMalloc((void**)&ret, N*sizeof(T));
	return ret;
      }


      T* operator()(T value, unsigned int N)
      {
	T* ret = nullptr;
	ret = operator()(N);
	assert(ret != nullptr);
	const int block_size = 256;
	const int number_of_blocks = (int)ceil((float)N/block_size);
	std::cout << " value : " << value << " N : " << N << " number_of_blocks : " << number_of_blocks << std::endl;
	fill<<<number_of_blocks, block_size>>>(ret, value, N);
	return ret;
      }
    };

  template<typename T>
    struct destroy<T, MATools::MAGPU::MEM_MODE::GPU, MATools::MAGPU::GPU_TYPE::CUDA>
    {
      void operator()(T* ret)
      {
	cudaFree(ret);
      }
    };

  template<typename T>
    struct copier<T, MATools::MAGPU::MEM_MODE::GPU, MATools::MAGPU::GPU_TYPE::CUDA>
    {
      void operator()(T* dst, T* const src, int N)
      {
	cudaMemcpy(dst, src, N*sizeof(T), cudaMemcpyDeviceToHost);
      }
    };

  template<typename Func, typename... Args>
    __global__
    void mini_launch(Func fun, unsigned int idx, Args... args)
    {
      fun(idx, args...);
    }
  template<>
    struct mini_runner<MATools::MAGPU::MEM_MODE::GPU, MATools::MAGPU::GPU_TYPE::CUDA>
    {
      template<typename Func, typename... Args>
	void operator()(Func& fun, unsigned int idx,  Args&&... args)
	{
	  std::cout << " run  " << fun.get_name() << std::endl;
	  mini_launch<<<1,1>>>(fun, idx, args...);
	  gpuErrchk(cudaPeekAtLastError());	
	  gpuErrchk(cudaDeviceSynchronize());	
	}
    };
#endif
}
