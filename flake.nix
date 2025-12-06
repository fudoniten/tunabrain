{
  description = "TunaBrain FastAPI service";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        pythonEnv = pkgs.python311.withPackages (ps: [
          ps.fastapi
          ps.uvicorn
          ps.pydantic
          ps.langchain
          ps.langchain-core
        ]);
      in {
        devShells.default = pkgs.mkShell {
          name = "tunabrain-dev";
          packages = [ pythonEnv pkgs.ruff pkgs.python311Packages.pytest ];
          shellHook = ''
            export PYTHONPATH=${builtins.getEnv "PWD"}/src:$PYTHONPATH
          '';
        };
      }
    );
}
