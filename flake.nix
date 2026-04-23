{
  description = "Unsloth and Gemma4 Training Environment";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
          config.cudaSupport = true;
        };
        # PyPIの最新PyTorchに合わせて明示的にCUDA 12系を指定する
        cudaPkg = pkgs.cudaPackages_12; 
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            python313
            python313Packages.venvShellHook
            uv
            linuxPackages.nvidia_x11
            cudaPkg.cudatoolkit
            cudaPkg.cudnn
            cudaPkg.libcusparse_lt
            cudaPkg.nccl
            stdenv.cc.cc.lib
            zlib
            openssl
            glib # PythonのC拡張でしばしば要求されるため追加
          ];
          venvDir = "./.venv";
          postVenvCreation = ''
            unset SOURCE_DATE_EPOCH
          '';
          shellHook = ''
            # 手動の文字列結合を捨て、makeLibraryPathに任せる
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [
              pkgs.linuxPackages.nvidia_x11
              pkgs.stdenv.cc.cc.lib
              pkgs.zlib
              pkgs.glib
              cudaPkg.cudatoolkit
              cudaPkg.cudnn
              cudaPkg.libcusparse_lt
              cudaPkg.nccl
              cudaPkg.libnvshmem
            ]}:$LD_LIBRARY_PATH"
            export LD_LIBRARY_PATH=/run/opengl-driver/lib:$LD_LIBRARY_PATH
            export TRITON_LIBCUDA_PATH=/run/opengl-driver/lib
            
            export EXTRA_CCFLAGS="-I${cudaPkg.cudatoolkit}/include"
          '';
        };
      }
    );
}
