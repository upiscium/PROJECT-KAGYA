{
  description = "Python + uv Environment";

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
        };
        cudaLibs = with pkgs.cudaPackages; [ cudnn cudatoolkit ];
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            uv
            just
            ruff
            pre-commit
            gnused
          ] ++ cudaLibs;

          shellHook = ''
            # uvの仮想環境をプロジェクト直下の .venv に強制
            export UV_PROJECT_ENVIRONMENT=$PWD/.venv

            # CUDA/CuDNN runtime を torch から見えるようにする
            export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath cudaLibs}:$LD_LIBRARY_PATH
            
            # エージェントが誤ってグローバル環境を触らないための防御壁
            export PIP_REQUIRE_VIRTUALENV=1
          '';
        };
      }
    );
}
