pub const cudaError_t = c_uint;
pub const cudaSuccess: cudaError_t = 0;
pub const cudaErrorInvalidValue: cudaError_t = 1;
pub const cudaErrorMemoryAllocation: cudaError_t = 2;
pub const cudaErrorInitializationError: cudaError_t = 3;
pub const cudaErrorLaunchFailure: cudaError_t = 4;
pub const cudaErrorLaunchTimeout: cudaError_t = 6;
pub const cudaErrorLaunchOutOfResources: cudaError_t = 7;
pub const cudaErrorInvalidDeviceFunction: cudaError_t = 8;
pub const cudaErrorInvalidConfiguration: cudaError_t = 9;
pub const cudaErrorInvalidDevice: cudaError_t = 10;
pub const cudaErrorInvalidMemcpyDirection: cudaError_t = 21;

pub const cudaHostAllocDefault: c_uint = 0;
pub const cudaHostAllocPortable: c_uint = 1;
pub const cudaHostAllocMapped: c_uint = 2;
pub const cudaHostAllocWriteCombined: c_uint = 4;

pub const cudaMemcpyHostToHost: c_uint = 0;
pub const cudaMemcpyHostToDevice: c_uint = 1;
pub const cudaMemcpyDeviceToHost: c_uint = 2;
pub const cudaMemcpyDeviceToDevice: c_uint = 3;
pub const cudaMemcpyDefault: c_uint = 4;

pub const cudaStream_t = ?*anyopaque;

pub extern "c" fn cudaHostAlloc(ptr: *?*anyopaque, size: usize, flags: c_uint) cudaError_t;
pub extern "c" fn cudaFreeHost(ptr: ?*anyopaque) cudaError_t;
pub extern "c" fn cudaMalloc(devPtr: *?*anyopaque, size: usize) cudaError_t;
pub extern "c" fn cudaFree(devPtr: ?*anyopaque) cudaError_t;
pub extern "c" fn cudaMemcpy(dst: ?*anyopaque, src: ?*const anyopaque, count: usize, kind: c_uint) cudaError_t;
pub extern "c" fn cudaMemcpyAsync(dst: ?*anyopaque, src: ?*const anyopaque, count: usize, kind: c_uint, stream: cudaStream_t) cudaError_t;
pub extern "c" fn cudaMemset(devPtr: ?*anyopaque, value: c_int, count: usize) cudaError_t;
pub extern "c" fn cudaDeviceSynchronize() cudaError_t;
pub extern "c" fn cudaStreamSynchronize(stream: cudaStream_t) cudaError_t;
pub extern "c" fn cudaGetLastError() cudaError_t;
pub extern "c" fn cudaPeekAtLastError() cudaError_t;
pub extern "c" fn cudaGetErrorString(err: cudaError_t) [*:0]const u8;
pub extern "c" fn cudaGetErrorName(err: cudaError_t) [*:0]const u8;
pub extern "c" fn cudaStreamCreate(pStream: *cudaStream_t) cudaError_t;
pub extern "c" fn cudaStreamDestroy(stream: cudaStream_t) cudaError_t;
pub extern "c" fn cudaGetDeviceCount(count: *c_int) cudaError_t;
pub extern "c" fn cudaSetDevice(device: c_int) cudaError_t;
pub extern "c" fn cudaGetDevice(device: *c_int) cudaError_t;

pub const CudaError = error{
    InvalidValue,
    MemoryAllocation,
    InitializationError,
    LaunchFailure,
    LaunchTimeout,
    LaunchOutOfResources,
    InvalidDeviceFunction,
    InvalidConfiguration,
    InvalidDevice,
    InvalidMemcpyDirection,
    Unknown,
};

pub fn toError(err: cudaError_t) CudaError!void {
    return switch (err) {
        cudaSuccess => {},
        cudaErrorInvalidValue => CudaError.InvalidValue,
        cudaErrorMemoryAllocation => CudaError.MemoryAllocation,
        cudaErrorInitializationError => CudaError.InitializationError,
        cudaErrorLaunchFailure => CudaError.LaunchFailure,
        cudaErrorLaunchTimeout => CudaError.LaunchTimeout,
        cudaErrorLaunchOutOfResources => CudaError.LaunchOutOfResources,
        cudaErrorInvalidDeviceFunction => CudaError.InvalidDeviceFunction,
        cudaErrorInvalidConfiguration => CudaError.InvalidConfiguration,
        cudaErrorInvalidDevice => CudaError.InvalidDevice,
        cudaErrorInvalidMemcpyDirection => CudaError.InvalidMemcpyDirection,
        else => CudaError.Unknown,
    };
}
