// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Proto <-> CRD glue for the ModelMetadata types, which live in
//! modelexpress-types so external consumers can link them without the server.

pub use modelexpress_types::p2p::*;

use modelexpress_common::grpc::p2p::MxSourceType;

/// Convert an `MxSourceType` proto enum value (i32) to the CRD source type string.
pub fn source_type_name_from_proto(mx_source_type: i32) -> String {
    match MxSourceType::try_from(mx_source_type) {
        Ok(MxSourceType::Weights) => "weights",
        Ok(MxSourceType::Lora) => "lora",
        Ok(MxSourceType::CudaGraph) => "cuda_graph",
        Ok(MxSourceType::TorchCompileCache) => "torch_compile_cache",
        Ok(MxSourceType::TritonCache) => "triton_cache",
        Ok(MxSourceType::DeepGemmCache) => "deep_gemm_cache",
        Ok(MxSourceType::TilelangCache) => "tilelang_cache",
        Ok(MxSourceType::CuteDslCache) => "cute_dsl_cache",
        Ok(MxSourceType::FlashinferCache) => "flashinfer_cache",
        Ok(MxSourceType::TvmFfiCache) => "tvm_ffi_cache",
        Err(_) => "unknown",
    }
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_type_name_from_proto() {
        assert_eq!(source_type_name_from_proto(0), "weights");
        assert_eq!(source_type_name_from_proto(3), "torch_compile_cache");
        assert_eq!(source_type_name_from_proto(5), "deep_gemm_cache");
        assert_eq!(source_type_name_from_proto(6), "tilelang_cache");
        assert_eq!(source_type_name_from_proto(7), "cute_dsl_cache");
        assert_eq!(source_type_name_from_proto(8), "flashinfer_cache");
        assert_eq!(source_type_name_from_proto(9), "tvm_ffi_cache");
        assert_eq!(source_type_name_from_proto(99), "unknown");
    }
}
