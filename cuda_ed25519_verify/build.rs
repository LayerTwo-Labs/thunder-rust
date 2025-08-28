use std::env;
use std::path::PathBuf;

extern crate link_cplusplus;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let cuda_dir = manifest_dir.join("cuda");

    // Find CUDA installation
    let cuda_home = env::var("CUDA_HOME")
        .or_else(|_| env::var("CUDA_PATH"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());

    let cuda_include = format!("{}/include", cuda_home);
    let cuda_lib = format!("{}/lib64", cuda_home);

    build_cuda_library(
        &out_dir,
        &cuda_dir,
        &cuda_home,
        &cuda_include,
        &cuda_lib,
    );
    generate_bindings(&out_dir);
}

fn build_cuda_library(
    out_dir: &PathBuf,
    cuda_dir: &PathBuf,
    cuda_home: &str,
    cuda_include: &str,
    cuda_lib: &str,
) {
    // Compile only the files needed for simple threaded verification (NO MSM)
    let cuda_files = [
        "fe.cu",
        "ge.cu",
        "sc.cu",
        "sha512.cu",
        "util.cu",
        "verify.cu", // This contains the simple threading kernel
    ];

    // Create a simple wrapper that launches the threading kernel
    let cuda_wrapper = out_dir.join("wrapper.cu");
    let wrapper_content = format!(
        r#"
#include <stdio.h>
#include "{}/ed25519.cuh"

extern "C" {{
    // Simple threaded verification kernel launcher
    void launch_verify_kernel(
        const unsigned char *signature,
        const unsigned char *message,
        size_t *message_len,
        const unsigned char *public_key,
        int *verified,
        int *key_mapping,
        int limit,
        int blocks,
        int threads_per_block
    ) {{
        dim3 grid(blocks);
        dim3 block(threads_per_block);
        
        // Launch the simple threading kernel - each thread verifies one signature
        ed25519_kernel_verify_batch_multi_keypair<<<grid, block>>>(
            signature,
            message,
            message_len,
            public_key,
            verified,
            key_mapping,
            limit
        );
    }}
}}
"#,
        cuda_dir.display()
    );

    std::fs::write(&cuda_wrapper, wrapper_content)
        .expect("Failed to write CUDA wrapper");

    // Compile each file separately and collect object files
    let mut object_files = Vec::new();

    // Add the wrapper to sources list
    let mut all_sources = Vec::new();
    for file in &cuda_files {
        all_sources.push(cuda_dir.join(file));
    }
    all_sources.push(cuda_wrapper);

    for source in &all_sources {
        let source_name = source.file_stem().unwrap().to_str().unwrap();
        let obj_path = out_dir.join(source_name).with_extension("o");

        let output = std::process::Command::new("nvcc")
            .arg("--verbose")
            .arg("-c")
            .arg("-O3")
            .arg("--use_fast_math")
            .arg("-arch=sm_86")
            .arg("--ptxas-options=-O3")
            .arg("-DCUDA_ARCH=860")
            .arg("-Xcompiler")
            .arg("-fPIC")
            .arg("-rdc=true") // Enable separate compilation
            .arg("-I")
            .arg(cuda_include)
            .arg("-I")
            .arg(cuda_dir)
            .arg(source)
            .arg("-o")
            .arg(&obj_path)
            .output()
            .expect("Failed to execute nvcc");

        if !output.status.success() {
            panic!(
                "nvcc compilation failed for {}: {}",
                source.display(),
                String::from_utf8_lossy(&output.stderr)
            );
        }

        object_files.push(obj_path);
        println!("cargo:rerun-if-changed={}", source.display());
    }

    // Link all object files together
    let linked_obj = out_dir.join("cuda_ed25519_simple_linked.o");
    let mut link_cmd = std::process::Command::new("nvcc");
    link_cmd
        .arg("-dlink")
        .arg("-arch=sm_86")
        .arg("-o")
        .arg(&linked_obj);

    for obj in &object_files {
        link_cmd.arg(obj);
    }

    let output = link_cmd
        .output()
        .expect("Failed to execute nvcc device link");
    if !output.status.success() {
        panic!(
            "nvcc device linking failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    // Archive into a library
    let lib_path = out_dir.join("libcuda_ed25519_simple.a");
    let mut ar_cmd = std::process::Command::new("ar");
    ar_cmd.arg("rcs").arg(&lib_path);

    // Add the device-linked object first
    ar_cmd.arg(&linked_obj);

    // Then add all individual object files
    for obj in &object_files {
        ar_cmd.arg(obj);
    }

    let output = ar_cmd.output().expect("Failed to execute ar");
    if !output.status.success() {
        panic!(
            "ar archiving failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    // Link with CUDA libraries
    println!("cargo:rustc-link-lib=static=cuda_ed25519_simple");
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-search=native={}", cuda_lib);

    // Add C++ runtime as needed
    println!("cargo:rustc-link-arg=-Wl,-Bdynamic");
    println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
}

fn generate_bindings(out_dir: &PathBuf) {
    // Generate simple bindings for the threading kernel only
    let bindings_path = out_dir.join("bindings.rs");
    std::fs::write(
        &bindings_path,
        r#"
// Simple threading kernel bindings
extern "C" {
    pub fn launch_verify_kernel(
        signature: *const u8,
        message: *const u8,
        message_len: *mut usize,
        public_key: *const u8,
        verified: *mut i32,
        key_mapping: *mut i32,
        limit: i32,
        blocks: i32,
        threads_per_block: i32,
    );
}
"#,
    )
    .expect("Couldn't write bindings!");
}
