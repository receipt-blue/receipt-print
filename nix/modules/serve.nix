{ self }:
{ config, lib, pkgs, ... }:

let
  cfg = config.services.receipt-print-serve;
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

    networking.firewall.allowedTCPPorts =
      lib.mkIf cfg.openFirewall [ cfg.port ];

    systemd.services.receipt-print-serve = {
      description = "receipt-print raw ESC/POS HTTP serve (standalone)";
      wantedBy = [ "multi-user.target" ];
      after = [ "network.target" ];
      environment = cfg.environment;
      serviceConfig = {
        ExecStart = "${lib.getExe cfg.package} ${lib.escapeShellArgs serveArgs}";
        User = cfg.user;
        Group = cfg.group;
        Restart = "always";
        RestartSec = "3s";
        TimeoutStopSec = "15s";
      };
    };
  };
}
