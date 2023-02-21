#pragma once 

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
			void operator()(T* ret)
			{
				delete ret;
			}
		};

	template<typename T, MATools::MAGPU::MEM_MODE MM, MATools::MAGPU::GPU_TYPE GT>
		struct copier
		{
			void operator()(T* dst, T* src, int N);
		};

	template<typename T, MATools::MAGPU::MEM_MODE MM>
		struct copier<T, MM, MATools::MAGPU::GPU_TYPE::SERIAL>
		{
			void operator()(T* dst, T* src, int N)
			{
				std::copy(src, src + N, dst);
			}
		};

	template<MATools::MAGPU::MEM_MODE, MATools::MAGPU::GPU_TYPE MM>
		struct mini_runner
		{
			template<typename Func, typename... Args>
			void operator()(Func& fun, Args&&... args)
			{
				fun(std::forward<Args>(args)...);
			}
		};

#ifdef __CUDA__
	template<typename T>
		struct create<T, MATools::MAGPU::MEM_MODE::GPU, MATools::MAGPU::GPU_TYPE::CUDA>
		{
			T* operator()(unsigned int N)
			{
				T* ret;
				cudaMalloc(&ret,N*sizeof(T));
				return ret;
			}

			template<typename T>
			__global__
			void fill(T* const ptr, T value, unsigned int N)
			{
				unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;
				if(idx < N)
					ptr[idx] = value;
			}

			T* operator()(T value, unsigned int N)
			{
				T* ret;
				cudaMalloc(&ret,N*sizeof(T));
				const int block_size = 256;
				const int number_of_blocks = (int)ceil((float)N/block_size);
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
			void operator()(T* dst, T* src, int N)
			{
				cudaMemcpy(dst, src, N*sizeof(T), cudaMemcpyDeviceToHost);
			}
		};

	template<>
		struct mini_runner<MATools::MAGPU::MEM_MODE::GPU, MATools::MAGPU::GPU_TYPE::CUDA>
		{

			template<typename Func, typename... Args>
			__global__
			void launch(Func fun, Args... args)
			{
				fun(args...);
			}

			template<typename Func, typename... Args>
			void operator()(Func& fun, Args&&... args)
			{
				
				launch<<<1,1>>>(fun,args...);
			}
		}
#endif
}
