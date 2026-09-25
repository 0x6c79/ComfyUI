{
  description = "ComfyUI - NixOS + TheRock ROCm 10";
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
          stdenv.cc.cc.lib
          zlib
          glibc
          numactl
        ];
        # TheRock 的 pip 轮子自带 rocm，不需要 rocmPackages.clr
        shellHook = ''
          # 让 .venv 里的 python 优先
          export UV_PYTHON=3.13
          export HSA_OVERRIDE_GFX_VERSION=10.3.0
          # 完善动态链接库路径，加入 numactl 和系统 C 库
          export LD_LIBRARY_PATH=${
            pkgs.lib.makeLibraryPath [
              pkgs.stdenv.cc.cc.lib
              pkgs.zlib
              pkgs.numactl
              pkgs.glibc
            ]
          }:$LD_LIBRARY_PATH

        '';
      };
    };
}
