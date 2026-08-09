{
  description = "Receipt printer CLI";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    receipt-core.url = "git+ssh://git@github.com/receipt-blue/receipt-substrate.git";
    receipt-core.inputs.nixpkgs.follows = "nixpkgs";
    receipt-core.inputs.flake-utils.follows = "flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, receipt-core, ... }:
    let
      serveModule = import ./nix/modules/serve.nix { inherit self; };
    in
    {
      nixosModules.receipt-print-serve = serveModule;
      nixosModules.default = serveModule;
    }
    //
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python312;
        libusbPath = pkgs.lib.makeLibraryPath [ pkgs.libusb1 ];
        receiptCore = receipt-core.packages.${system}.receipt-core;
        receipt-print = python.pkgs.buildPythonApplication {
          pname = "receipt-print";
          version = "0.1.0";
          src = self;
          pyproject = true;
          nativeBuildInputs = [
            pkgs.makeWrapper
          ];
          build-system = with python.pkgs; [
            setuptools
            wheel
          ];
          propagatedBuildInputs = with python.pkgs; [
            click
            numpy
            pdf2image
            pillow
            python-escpos
            python-dotenv
            pyusb
            requests
          ];
          pythonRemoveDeps = [
            "markitdown"
            "receipt-core-renderer"
          ];
          pythonImportsCheck = [ "receipt_print" ];
          postInstall = ''
            wrapProgram "$out/bin/receipt-print" \
              --prefix LD_LIBRARY_PATH : ${libusbPath} \
              --set RECEIPT_CORE_BIN ${receiptCore}/bin/receipt-core
          '';
          meta = {
            description = "Receipt printer CLI";
            mainProgram = "receipt-print";
          };
        };
        receipt-print-tests = pkgs.runCommand "receipt-print-tests" {
          nativeBuildInputs = [
            python.pkgs.pytest
            receipt-print
          ];
        } ''
          cd ${self}
          pytest
          touch "$out"
        '';
      in
      {
        packages.default = receipt-print;
        packages.receipt-print = receipt-print;

        apps.default = {
          type = "app";
          program = "${receipt-print}/bin/receipt-print";
          meta.description = "Print files, links, and Are.na channels";
        };

        checks.default = receipt-print-tests;

        devShells.default = pkgs.mkShell {
          packages = [
            receipt-print
            pkgs.libusb1
            python.pkgs.pytest
          ];
          RP_VENDOR = "04b8";
          RP_PROFILE = "TM-T20II";
          LD_LIBRARY_PATH = libusbPath;
        };
      });
}
