{
  description = "StableHLO Extractor";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      uv2nix,
      pyproject-nix,
      pyproject-build-systems,
      ...
    }:

    flake-utils.lib.eachDefaultSystem (
      system:
      let
        inherit (nixpkgs) lib;
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
        python = pkgs.python311;

        # Load a uv workspace from a workspace root.
        # Uv2nix treats all uv projects as workspace projects.
        workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };

        # Create package overlay from workspace.
        overlay = workspace.mkPyprojectOverlay {
          # Prefer prebuilt binary wheels as a package source.
          # Sdists are less likely to "just work" because of the metadata missing from uv.lock.
          # Binary wheels are more likely to, but may still require overrides for library dependencies.
          sourcePreference = "wheel"; # or sourcePreference = "sdist";
          # Optionally customise PEP 508 environment
          # environ = {
          #   platform_release = "5.10.65";
          # };
        };

        cudaLibs = [
          pkgs.cudaPackages.cudnn
          pkgs.cudaPackages.nccl
          pkgs.cudaPackages.cutensor
          pkgs.cudaPackages.cusparselt
          pkgs.cudaPackages.libcublas
          pkgs.cudaPackages.libcusparse
          pkgs.cudaPackages.libcusolver
          pkgs.cudaPackages.libcurand
          pkgs.cudaPackages.cuda_cupti
          pkgs.cudaPackages.libcufft
          pkgs.cudaPackages.cuda_nvrtc
          pkgs.cudaPackages.cuda_cudart
          pkgs.rdma-core
        ];
        cudaLDLibraryPath = pkgs.lib.makeLibraryPath cudaLibs;

        # Extend generated overlay with build fixups
        #
        # Uv2nix can only work with what it has, and uv.lock is missing essential metadata to perform some builds.
        # This is an additional overlay implementing build fixups.
        # See:
        # - https://pyproject-nix.github.io/uv2nix/FAQ.html
        pyprojectOverrides = final: prev: {
          # Implement build fixups here.
          # Note that uv2nix is _not_ using Nixpkgs buildPythonPackage.
          # It's using https://pyproject-nix.github.io/pyproject.nix/build.html
          torch = prev.torch.overrideAttrs (old: {
            buildInputs = (old.buildInputs or [ ]) ++ cudaLibs;
          });
          torch-xla = prev.torch-xla.overrideAttrs (old: {
            buildInputs = (old.buildInputs or [ ]) ++ cudaLibs;
            postFixup = ''
              addAutoPatchelfSearchPath "${final.torch}"
            '';
          });
          nvidia-cusolver-cu12 = prev.nvidia-cusolver-cu12.overrideAttrs (old: {
            buildInputs = (old.buildInputs or [ ]) ++ cudaLibs;
          });
          nvidia-cusparse-cu12 = prev.nvidia-cusparse-cu12.overrideAttrs (old: {
            buildInputs = (old.buildInputs or [ ]) ++ cudaLibs;
          });
          nvidia-cufile-cu12 = prev.nvidia-cufile-cu12.overrideAttrs (old: {
            buildInputs = (old.buildInputs or [ ]) ++ cudaLibs;
          });
        };

        # Construct package set
        pythonSet =
          # Use base package set from pyproject.nix builders
          (pkgs.callPackage pyproject-nix.build.packages {
            inherit python;
          }).overrideScope
            (
              lib.composeManyExtensions [
                pyproject-build-systems.overlays.default
                overlay
                pyprojectOverrides
              ]
            );
        virtualenv = pythonSet.mkVirtualEnv "stablehlo-extracter-env" workspace.deps.default;
      in
      {
        # Package a virtual environment as our main application.
        #
        # Enable no optional dependencies for production build.
        packages.default = virtualenv;

        # Make extracter runnable with `nix run`
        apps.default = {
          type = "app";
          program = "${virtualenv}/bin/extract";
        };

        # This example provides two different modes of development:
        # - Impurely using uv to manage virtual environments
        # - Pure development using uv2nix to manage virtual environments
        devShells = {
          # It is of course perfectly OK to keep using an impure virtualenv workflow and only use uv2nix to build packages.
          # This devShell simply adds Python and undoes the dependency leakage done by Nixpkgs Python infrastructure.
          impure = pkgs.mkShell {
            packages = [
              virtualenv
              python
              pkgs.uv
              pkgs.ruff
              pkgs.basedpyright
            ];
            env = {
              # Prevent uv from managing Python downloads
              UV_PYTHON_DOWNLOADS = "never";
              # Force uv to use nixpkgs Python interpreter
              UV_PYTHON = python.interpreter;
            }
            // lib.optionalAttrs pkgs.stdenv.isLinux {
              # Python libraries often load native shared objects using dlopen(3).
              # Setting LD_LIBRARY_PATH makes the dynamic library loader aware of libraries without using RPATH for lookup.
              LD_LIBRARY_PATH = lib.makeLibraryPath pkgs.pythonManylinuxPackages.manylinux1;
            };
            shellHook = ''
              unset PYTHONPATH
            '';
          };

          # This devShell uses uv2nix to construct a virtual environment purely from Nix, using the same dependency specification as the application.
          # The notable difference is that we also apply another overlay here enabling editable mode ( https://setuptools.pypa.io/en/latest/userguide/development_mode.html ).
          #
          # This means that any changes done to your local files do not require a rebuild.
          #
          # Note: Editable package support is still unstable and subject to change.
          uv2nix =
            let
              # Create an overlay enabling editable mode for all local dependencies.
              editableOverlay = workspace.mkEditablePyprojectOverlay {
                # Use environment variable
                root = "$REPO_ROOT";
                # Optional: Only enable editable for these packages
                # members = [ "stablehlo-extracter" ];
              };

              # Override previous set with our overrideable overlay.
              editablePythonSet = pythonSet.overrideScope (
                lib.composeManyExtensions [
                  editableOverlay

                  # Apply fixups for building an editable package of your workspace packages
                  (final: prev: {
                    stablehlo-extracter = prev.stablehlo-extracter.overrideAttrs (old: {
                      # It's a good idea to filter the sources going into an editable build
                      # so the editable package doesn't have to be rebuilt on every change.
                      src = lib.fileset.toSource {
                        root = old.src;
                        fileset = lib.fileset.unions [
                          (old.src + "/pyproject.toml")
                          (old.src + "/README.md")
                          (old.src + "/src/stablehlo_extracter/__init__.py")
                        ];
                      };

                      # Hatchling (our build system) has a dependency on the `editables` package when building editables.
                      #
                      # In normal Python flows this dependency is dynamically handled, and doesn't need to be explicitly declared.
                      # This behaviour is documented in PEP-660.
                      #
                      # With Nix the dependency needs to be explicitly declared.
                      nativeBuildInputs =
                        old.nativeBuildInputs
                        ++ final.resolveBuildSystem {
                          editables = [ ];
                        };
                    });

                  })
                ]
              );

              # Build virtual environment, with local packages being editable.
              #
              # Enable all optional dependencies for development.
              virtualenv = editablePythonSet.mkVirtualEnv "stablehlo-extracter-dev-env" workspace.deps.all;

            in
            pkgs.mkShell {
              packages = [
                virtualenv
                pkgs.uv
              ];

              env = {
                # Don't create venv using uv
                UV_NO_SYNC = "1";

                # Force uv to use nixpkgs Python interpreter
                UV_PYTHON = python.interpreter;

                # Prevent uv from downloading managed Python's
                UV_PYTHON_DOWNLOADS = "never";
              };

              shellHook = ''
                # Undo dependency propagation by nixpkgs.
                unset PYTHONPATH

                # Get repository root using git. This is expanded at runtime by the editable `.pth` machinery.
                export REPO_ROOT=$(git rev-parse --show-toplevel)
              '';
            };
        };
      }
    );
}
