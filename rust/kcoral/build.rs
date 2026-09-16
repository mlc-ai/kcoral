fn main() -> Result<(), Box<dyn std::error::Error>> {
    let protoc = protoc_bin_vendored::protoc_bin_path()?;
    std::env::set_var("PROTOC", protoc);
    tonic_build::configure()
        .build_client(true)
        .build_server(true)
        .bytes([".kcoral.gateway.v1.SlotFrame.data"])
        .compile_protos(&["proto/kcoral_gateway.proto"], &["proto"])?;
    println!("cargo:rerun-if-changed=proto/kcoral_gateway.proto");
    Ok(())
}
