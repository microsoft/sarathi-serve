#include <torch/torch.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <nccl.h>
#include <iostream>
#include <fstream>
#include <system_error>

ncclComm_t comm; // NCCL communicator


void init_nccl(int world_size, int rank, const std::string& file_path) {
    ncclUniqueId id;
    ncclResult_t nccl_res;

    // Rank 0 creates the unique ID and writes it to the file
    if (rank == 0) {
        nccl_res = ncclGetUniqueId(&id);
        if (nccl_res != ncclSuccess) {
            throw std::runtime_error("The ncclGetUniqueId operation failed: " + std::string(ncclGetErrorString(nccl_res)));
        }

        // Write the id to the file
        std::ofstream file(file_path, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Failed to open file file: " +  file_path);
        }

        file.write(reinterpret_cast<char*>(&id), sizeof(id));
        if (!file) {
            throw std::runtime_error("Failed to write to file: " +  file_path);
        }
        file.close();
    }

    // Synchronize all processes, ensuring the file is written before others continue
    // This may involve a barrier synchronization call depending on your environment (e.g., MPI_Barrier if using MPI)

    // Other ranks read the unique ID from the file
    if (rank != 0) {
        std::ifstream file;
        // Attempt to open the file multiple times if necessary, as the file might not be immediately available
        for (int i = 0; i < 10; ++i) { // retry 10 times
            file.open(file_path, std::ios::binary);
            if (file) break;
            if (i == 9) { // last attempt
                throw std::runtime_error("Failed to open file file: " +  file_path);
            }
            std::this_thread::sleep_for(std::chrono::seconds(1)); // sleep for a second before retrying
        }

        file.read(reinterpret_cast<char*>(&id), sizeof(id));
        if (!file) {
            throw std::runtime_error("Failed to read from file: " +  file_path);
        }
        file.close();
    }

    // Initialize the communicator
    nccl_res = ncclCommInitRank(&comm, world_size, id, rank);
    if (nccl_res != ncclSuccess) {
        throw std::runtime_error("The ncclCommInitRank operation failed: " + std::string(ncclGetErrorString(nccl_res)));
    }
}

void all_reduce(torch::Tensor& input_tensor) {
    // Check if tensor is empty
    if (input_tensor.numel() == 0) {
        throw std::runtime_error("The input tensor is empty.");
    }

    // Check if tensor is on GPU
    if (!input_tensor.device().is_cuda()) {
        throw std::runtime_error("The input tensor must be a CUDA tensor.");
    }

    // Check if the tensor's data type is supported
    ncclDataType_t dtype;
    if (input_tensor.scalar_type() == torch::kFloat16) {
        dtype = ncclHalf;
    } else if (input_tensor.scalar_type() == torch::kFloat32) {
        dtype = ncclFloat;
    } else if (input_tensor.scalar_type() == torch::kFloat64) {
        dtype = ncclDouble;
    } else {
        throw std::runtime_error("The data type of the input tensor is not supported.");
    }

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Perform the all-reduce operation
    ncclResult_t result = ncclAllReduce(input_tensor.data_ptr(), input_tensor.data_ptr(),
                                        input_tensor.numel(), dtype, ncclSum, comm, stream);

    // Check if the all-reduce operation was successful
    if (result != ncclSuccess) {
        throw std::runtime_error("The all-reduce operation failed: " + std::string(ncclGetErrorString(result)));
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("init_nccl", &init_nccl, "Initialize NCCL communicator");
    m.def("all_reduce", &all_reduce, "Perform NCCL all-reduce on a PyTorch tensor");
}
