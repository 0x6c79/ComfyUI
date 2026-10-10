{
  description = "ComfyUI - NixOS + ROCm 7.2";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  outputs =
    { self, nixpkgs }:
    let
      pkgs = nixpkgs.legacyPackages.x86_64-linux;
    in
    {
      devShells.x86_64-linux.default = pkgs.mkShell {
        packages = with pkgs; [
          python313 # 官方现在推荐 3.13，3.12 也可用
          git
          python313Packages.pip
          ruff
          ty

          stdenv.cc.cc.lib
          zlib
          zstd
          glib
          glibc
          numactl
        ];

        shellHook = ''
          # 让 .venv 里的 python 优先
          export UV_PYTHON=3.13
          # export HSA_OVERRIDE_GFX_VERSION=10.3.0
          # 完善动态链接库路径，加入 numactl 和 C 库
          # 注意：不要把 pkgs.glibc 放进来，否则会覆盖系统 glibc，
          # 导致系统工具报 GLIBC_x.xx not found
          export LD_LIBRARY_PATH="${
            pkgs.lib.makeLibraryPath [
              pkgs.stdenv.cc.cc.lib
              pkgs.zlib
              pkgs.zstd
              pkgs.glib
              pkgs.numactl
            ]
          }''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

        '';
      };
    };
}

# {
#   description = "ComfyUI development shell";

#   inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

#   outputs =
#     { self, nixpkgs }:
#     let
#       system = "x86_64-linux";
#       pkgs = nixpkgs.legacyPackages.${system};

#       runtimeLibs = with pkgs; [
#         stdenv.cc.cc.lib
#         zlib
#         zstd
#         glib
#         numactl
#       ];
#     in
#     {
#       devShells.${system}.default = pkgs.mkShell {
#         packages = with pkgs; [
#           python313
#           git
#           ruff
#           uv
#         ];

#         shellHook = ''
#           export UV_PYTHON=3.13

#           run-comfyui() {
#             LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath runtimeLibs}:$LD_LIBRARY_PATH" \
#               python main.py "$@"
#           }
#         '';
#       };
#     };
# }
