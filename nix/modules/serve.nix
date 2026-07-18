{ self }:
{ config, lib, pkgs, ... }:

let
  cfg = config.services.receipt-print-serve;
  printerEnvironment = {
    RP_HOST = "";
    RP_PRINT_MODE = "direct";
    RP_DEVICE_LOCK_PATH = "/run/receipt-print/device.lock";
    RP_SERVE_STATE_PATH = "/var/lib/receipt-print/jobs.sqlite3";
    RP_VENDOR = cfg.vendor;
    RP_PROFILE = cfg.profile;
  }
  // lib.optionalAttrs (cfg.device != null) {
    RP_DEVICE = cfg.device;
  }
  // lib.optionalAttrs (cfg.product != null) {
    RP_PRODUCT = cfg.product;
  }
  // cfg.environment;
  serveArgs = [
    "serve"
    "--host"
    cfg.host
    "--port"
    (toString cfg.port)
  ];
in
{
  options.services.receipt-print-serve = {
    enable = lib.mkEnableOption "standalone receipt-print raw ESC/POS HTTP serve";

    package = lib.mkOption {
      type = lib.types.package;
      default = self.packages.${pkgs.stdenv.hostPlatform.system}.default;
      defaultText = lib.literalExpression "receipt-print.packages.\${system}.default";
      description = "receipt-print package providing the `receipt-print serve` command.";
    };

    host = lib.mkOption {
      type = lib.types.str;
      default = "127.0.0.1";
      description = ''
        Bind address for the raw-print HTTP server. Keep this on loopback unless a
        non-kiosk client on another host is the intended consumer; the endpoint
        prints whatever bytes it receives to the local printer.
      '';
    };

    port = lib.mkOption {
      type = lib.types.port;
      default = 9100;
      description = ''
        Bind port for the raw-print HTTP server. Set this to the consumer's expected
        port (e.g. a Godot/API/dev client's configured print URL).
      '';
    };

    openFirewall = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = "Open `port` in the firewall. Only meaningful for a non-loopback host.";
    };

    user = lib.mkOption {
      type = lib.types.str;
      default = "receipt-print";
      description = "System user the serve daemon runs as. Must reach the printer device.";
    };

    group = lib.mkOption {
      type = lib.types.str;
      default = "lp";
      description = "Primary group for the serve user; `lp` reaches the printer device node.";
    };

    device = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = "/dev/receipt-printer";
      description = "Printer device path; set to null to use receipt-print's device discovery.";
    };

    vendor = lib.mkOption {
      type = lib.types.str;
      default = "04b8";
      description = "USB vendor ID and default udev match for the printer.";
    };

    product = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      description = "Optional USB product ID passed to receipt-print.";
    };

    profile = lib.mkOption {
      type = lib.types.str;
      default = "TM-T20II";
      description = "python-escpos printer profile passed to receipt-print.";
    };

    udev = {
      enable = lib.mkOption {
        type = lib.types.bool;
        default = true;
        description = "Install the USB printer permissions and receipt-printer symlink rule.";
      };
    };

    environment = lib.mkOption {
      type = lib.types.attrsOf lib.types.str;
      default = { };
      example = lib.literalExpression ''
        {
          RP_DEVICE = "/dev/receipt-printer";
          RP_VENDOR = "04b8";
          RP_PROFILE = "TM-T20II";
        }
      '';
      description = ''
        Extra environment for the serve daemon, threaded through to `connect_printer`
        (RP_DEVICE / RP_VENDOR / RP_PRODUCT / RP_PROFILE and the RP_SERVE_* tuning knobs).

        Do NOT set RP_HOST here: it would route prints to a Network(host=...) printer and
        hang on a device-not-found instead of using the local USB/File path. RP_SPEED and
        RP_SPEED_OVERRIDE are popped structurally at startup by `receipt-print serve`, so a
        speed value cannot inject GS ( K pre-stream bytes onto the raw wire even if set here.
      '';
    };

    configureClients = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = "Route receipt-print CLI commands through this service when it is healthy.";
    };
  };

  config = lib.mkIf cfg.enable {
    assertions = [
      {
        assertion = !(cfg.environment ? RP_HOST);
        message = ''
          services.receipt-print-serve.environment must not set RP_HOST: it routes prints
          to a Network printer and hangs on device-not-found. Leave it unset so the local
          USB/File path is used.
        '';
      }
    ];

    environment.sessionVariables = lib.mkIf cfg.configureClients {
      RP_SERVICE_URL = "http://${cfg.host}:${toString cfg.port}";
      RP_DEVICE_LOCK_PATH = "/run/receipt-print/device.lock";
    };

    users.groups.${cfg.group} = { };
    users.users.${cfg.user} = {
      isSystemUser = true;
      group = cfg.group;
    };

    services.udev.extraRules = lib.mkIf cfg.udev.enable ''
      SUBSYSTEM=="usb", ENV{DEVTYPE}=="usb_device", ATTR{idVendor}=="${cfg.vendor}", MODE="0660", GROUP="${cfg.group}", TAG+="uaccess"
      SUBSYSTEM=="usbmisc", KERNEL=="lp[0-9]*", ATTRS{idVendor}=="${cfg.vendor}", MODE="0660", GROUP="${cfg.group}", TAG+="uaccess", SYMLINK+="receipt-printer"
    '';

    networking.firewall.allowedTCPPorts =
      lib.mkIf cfg.openFirewall [ cfg.port ];

    systemd.services.receipt-print-serve = {
      description = "receipt-print raw ESC/POS HTTP serve (standalone)";
      wantedBy = [ "multi-user.target" ];
      after = [ "network-online.target" ];
      wants = [ "network-online.target" ];
      environment = printerEnvironment;
      serviceConfig = {
        ExecStart = "${lib.getExe cfg.package} ${lib.escapeShellArgs serveArgs}";
        User = cfg.user;
        Group = cfg.group;
        SupplementaryGroups = [ cfg.group ];
        Restart = "on-failure";
        RestartSec = "3s";
        TimeoutStopSec = "15s";
        StateDirectory = "receipt-print";
        StateDirectoryMode = "0750";
        RuntimeDirectory = "receipt-print";
        RuntimeDirectoryMode = "0770";
        UMask = "0007";
      };
    };
  };
}
