{
  description = "Receipt printer CLI";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python312;
        libusbPath = pkgs.lib.makeLibraryPath [ pkgs.libusb1 ];
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
            pyusb
            requests
          ];
          pythonRemoveDeps = [
            "markitdown"
          ];
          pythonImportsCheck = [ "receipt_print" ];
          postInstall = ''
            wrapProgram "$out/bin/receipt-print" \
              --prefix LD_LIBRARY_PATH : ${libusbPath}
          '';
          meta = {
            description = "Receipt printer CLI";
            mainProgram = "receipt-print";
          };
        };
      in
      {
        packages.default = receipt-print;
        packages.receipt-print = receipt-print;

        apps.default = {
          type = "app";
          program = "${receipt-print}/bin/receipt-print";
        };

        checks.default = receipt-print;

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
