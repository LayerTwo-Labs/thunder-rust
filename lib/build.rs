use std::{env, path::PathBuf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let protos: &[&str] = &["../proto/proto/mainchain.proto"];
    let includes: &[&str] = &["../proto/proto"];
    let out_dir = PathBuf::from(
        env::var("OUT_DIR").expect("OUT_DIR environment variable not set"),
    )
    .join("file_descriptor_set.bin");
    let mut config = prost_build::Config::new();
    config.enable_type_names();
    tonic_build::configure()
        .build_server(false)
        .build_transport(false)
        .file_descriptor_set_path(out_dir)
        .compile_with_config(config, protos, includes)?;
    Ok(())
}
