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
          ps.langchain-ollama
          ps.langgraph
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

        # Version information (git commit + timestamp)
        versionInfo = let
          gitCommit = self.rev or self.dirtyRev or "unknown";
          gitTimestamp = if self ? lastModified then
          # Format: YYYYMMDD-HHMMSS
            let
              ts = toString self.lastModified;
              # lastModified is Unix epoch, convert to readable format
              year = builtins.substring 0 4 ts;
              month = builtins.substring 4 2 ts;
              day = builtins.substring 6 2 ts;
            in "${year}${month}${day}"
          else
            "dev";
          versionTag = "${builtins.substring 0 7 gitCommit}-${gitTimestamp}";
        in { inherit gitCommit gitTimestamp versionTag; };

      in {
        packages = rec {
          default = tunabrain;
          tunabrain = tunabrainServer;
          deployContainer = let tunabrain = self.packages."${system}".tunabrain;
          in helpers.deployContainers {
            name = "tunabrain";
            repo = "registry.kube.sea.fudo.link";
            tags = [ "latest" versionInfo.versionTag ];
            entrypoint = [ "${tunabrain}/bin/tunabrain-server" ];
            verbose = true;
            env = {
              GIT_COMMIT = versionInfo.gitCommit;
              GIT_TIMESTAMP = versionInfo.gitTimestamp;
              VERSION = versionInfo.versionTag;
            };
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
          packages = [
            pythonEnv
            pkgs.ruff
            pkgs.python311Packages.pytest
            pkgs.python311Packages.pytest-asyncio
            pkgs.python311Packages.anyio
          ];
          shellHook = ''
            export PYTHONPATH=${builtins.getEnv "PWD"}/src:$PYTHONPATH
          '';
        };

        checks.tests = pkgs.stdenv.mkDerivation {
          pname = "tunabrain-tests";
          version = "0.1.0";
          src = ./.;
          buildInputs = [
            pythonEnv
            pkgs.python311Packages.pytest
            pkgs.python311Packages.pytest-asyncio
            pkgs.python311Packages.anyio
          ];

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
