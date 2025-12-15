{
  description = "TunaBrain FastAPI service";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    nix-helpers = {
      url = "github:fudoniten/fudo-nix-helpers";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, nix-helpers, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        helpers = nix-helpers.packages."${system}";
        pythonEnv = pkgs.python311.withPackages (ps: [
          ps.fastapi
          ps.uvicorn
          ps.pydantic
          ps.langchain
          ps.langchain-core
          ps.langchain-openai
          ps.httpx
        ]);
        tunabrainServer = pkgs.writeShellApplication {
          name = "tunabrain-server";
          runtimeInputs = [ pythonEnv pkgs.python311 ];
          text = ''
            export PYTHONPATH=${./src}:${"PYTHONPATH:-"}
            exec python -m tunabrain "$@"
          '';
        };
      in {
        packages = rec {
          default = tunabrain;
          tunabrain = tunabrainServer;
          deployContainer = let tunabrain = self.packages."${system}".tunabrain;
          in helpers.deployContainers {
            name = "tunabrain";
            repo = "registry.kube.sea.fudo.link";
            tags = [ "latest" ];
            entrypoint = [ "${tunabrain}/bin/tunabrain" ];
            environmentPackages = [ tunabrain ];
            verbose = true;
          };
        };

        apps = rec {
          default = tunabrain;
          tunabrain = flake-utils.lib.mkApp { drv = tunabrainServer; };
          deployContainer = {
            type = "app";
            program =
              let deployContainer = self.packages."${system}".deployContainer;
              in "${deployContainer}/bin/deployContainers";
          };
        };

        devShells.default = pkgs.mkShell {
          name = "tunabrain-dev";
          packages = [ pythonEnv pkgs.ruff pkgs.python311Packages.pytest ];
          shellHook = ''
            export PYTHONPATH=${builtins.getEnv "PWD"}/src:$PYTHONPATH
          '';
        };

        checks.tests = pkgs.stdenv.mkDerivation {
          pname = "tunabrain-tests";
          version = "0.1.0";
          src = ./.;
          buildInputs = [ pythonEnv pkgs.python311Packages.pytest ];

          buildPhase = ''
            runHook preBuild
            export HOME=$TMPDIR
            export PYTHONPATH=$PWD/src:$PYTHONPATH
            pytest -q
            runHook postBuild
          '';

          installPhase = ''
            mkdir -p $out
          '';
        };
      });
}
