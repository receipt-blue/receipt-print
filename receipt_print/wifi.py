import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

from escpos.escpos import QR_ECLEVEL_M

from .printer import CHAR_WIDTH, count_lines, enforce_line_limit, sanitize_output, wrap_text


class WifiError(RuntimeError):
    pass


@dataclass(frozen=True)
class WifiNetwork:
    ssid: str
    password: str = ""
    security: str = "WPA"
    hidden: bool = False
    profile_name: str = ""


@dataclass(frozen=True)
class WifiPanel:
    lines: List[str]
    bold_line_indices: set[int]
    width: int


@dataclass(frozen=True)
class NetworkManagerProfile:
    uuid: str
    name: str
    ssid: str
    key_mgmt: str
    hidden: bool


NmcliRunner = Callable[[Sequence[str], bool], str]

SECURITY_ALIASES = {
    "wpa": "WPA",
    "wpa2": "WPA",
    "wpa3": "WPA",
    "wep": "WEP",
    "nopass": "nopass",
    "none": "nopass",
    "open": "nopass",
}
ENTERPRISE_KEY_MGMT = {"wpa-eap", "ieee8021x"}
WPA_KEY_MGMT = {"wpa-psk", "sae"}
OPEN_KEY_MGMT = {"", "--", "owe"}
WEP_OR_OPEN_KEY_MGMT = {"none"}
WIFI_ESCAPE_CHARS = {"\\", ";", ",", ":", '"'}


def normalize_wifi_security(value: str) -> str:
    security = SECURITY_ALIASES.get((value or "").strip().lower())
    if not security:
        raise ValueError("security must be WPA, WEP, or nopass")
    return security


def escape_wifi_value(value: str) -> str:
    out = []
    for char in value:
        if char in WIFI_ESCAPE_CHARS:
            out.append("\\")
        out.append(char)
    return "".join(out)


def build_wifi_qr_payload(network: WifiNetwork) -> str:
    ssid = network.ssid
    if not ssid:
        raise ValueError("SSID is required")

    security = normalize_wifi_security(network.security)
    parts = ["WIFI:", f"T:{security};", f"S:{escape_wifi_value(ssid)};"]
    if security != "nopass":
        if not network.password:
            raise ValueError(f"{security} networks require a password")
        parts.append(f"P:{escape_wifi_value(network.password)};")
    if network.hidden:
        parts.append("H:true;")
    parts.append(";")
    return "".join(parts)


def build_wifi_panel(
    network: WifiNetwork,
    *,
    width: int = CHAR_WIDTH,
    wrap_mode: str = "hyphen",
    print_password: bool = True,
) -> WifiPanel:
    width = max(4, width)
    inner_width = max(1, width - 4)
    content: List[str] = []
    bold_indices: set[int] = set()

    def add_line(value: str, *, bold: bool = False, center: bool = False) -> None:
        idx = len(content)
        line = value.center(inner_width) if center else value
        content.append(line)
        if bold:
            bold_indices.add(idx)

    add_line("Wi-Fi", bold=True, center=True)
    content.append("")
    ssid_lines = _wrap_with_prefix(
        "ssid: ", _clean_panel_value(network.ssid), inner_width, wrap_mode
    )
    if ssid_lines:
        start_idx = len(content)
        content.extend(ssid_lines)
        bold_indices.update(range(start_idx, start_idx + len(ssid_lines)))
    security = normalize_wifi_security(network.security)
    if print_password and security != "nopass" and network.password:
        content.extend(
            _wrap_with_prefix(
                "password: ",
                _clean_panel_value(network.password),
                inner_width,
                wrap_mode,
            )
        )
    security_label = "open" if security == "nopass" else security
    content.extend(
        _wrap_with_prefix("security: ", security_label, inner_width, wrap_mode)
    )
    if network.hidden:
        content.append("hidden: yes")

    border = "+" + "-" * (width - 2) + "+"
    lines = [border]
    for line in content:
        lines.append("| " + line[:inner_width].ljust(inner_width) + " |")
    lines.append(border)
    return WifiPanel(
        lines=lines,
        bold_line_indices={idx + 1 for idx in bold_indices},
        width=width,
    )


def print_wifi_card(
    printer,
    network: WifiNetwork,
    *,
    wrap_mode: str,
    qr_size: int,
    ec: int = QR_ECLEVEL_M,
    print_password: bool = True,
) -> None:
    try:
        payload = build_wifi_qr_payload(network)
    except ValueError as exc:
        raise WifiError(str(exc)) from exc
    panel = build_wifi_panel(
        network,
        width=CHAR_WIDTH,
        wrap_mode=wrap_mode,
        print_password=print_password,
    )
    enforce_line_limit(count_lines("\n".join(panel.lines), panel.width))

    printer.set(align="left", font="a", bold=False, normal_textsize=True)
    for line_idx, line in enumerate(panel.lines):
        printer.set(bold=line_idx in panel.bold_line_indices)
        printer.text(_ensure_trailing_newline(line))
    printer.set(align="left", font="a", bold=False, normal_textsize=True)

    printer.set(align="right", font="a", bold=False)
    try:
        printer.qr(payload, size=qr_size, ec=ec)
    except Exception as exc:
        raise WifiError(f"Failed to print Wi-Fi QR: {exc}") from exc
    printer.set(align="left")


def select_stored_wifi_network() -> WifiNetwork:
    if sys.platform != "linux":
        raise WifiError("Stored Wi-Fi profile selection is only supported on Linux.")
    profiles = list_nmcli_wifi_profiles()
    if not profiles:
        raise WifiError("No stored NetworkManager Wi-Fi profiles were found.")
    profile = pick_nmcli_profile(profiles)
    return wifi_network_from_nmcli_profile(profile)


def list_nmcli_wifi_profiles(runner: NmcliRunner = None) -> List[NetworkManagerProfile]:
    run = runner or run_nmcli
    output = run(["-t", "-f", "UUID,TYPE,NAME", "connection", "show"], False)
    profiles: List[NetworkManagerProfile] = []
    for line in output.splitlines():
        parsed = parse_nmcli_connection_line(line)
        if parsed is None:
            continue
        uuid, connection_type, name = parsed
        if connection_type != "802-11-wireless":
            continue
        profiles.append(load_nmcli_profile(uuid, name, run))
    return profiles


def parse_nmcli_connection_line(line: str) -> Optional[tuple[str, str, str]]:
    if not line:
        return None
    parts = line.split(":", 2)
    if len(parts) != 3:
        return None
    return parts[0], parts[1], _unescape_nmcli(parts[2])


def load_nmcli_profile(
    uuid: str, name: str, runner: NmcliRunner = None
) -> NetworkManagerProfile:
    run = runner or run_nmcli
    fields = (
        "802-11-wireless.ssid,"
        "802-11-wireless-security.key-mgmt,"
        "802-11-wireless.hidden"
    )
    output = run(["-g", fields, "connection", "show", "uuid", uuid], False)
    values = output.splitlines()
    while len(values) < 3:
        values.append("")
    ssid = values[0] or name
    key_mgmt = values[1].strip().lower()
    hidden = values[2].strip().lower() in {"yes", "true", "1"}
    return NetworkManagerProfile(
        uuid=uuid,
        name=name,
        ssid=ssid,
        key_mgmt=key_mgmt,
        hidden=hidden,
    )


def wifi_network_from_nmcli_profile(
    profile: NetworkManagerProfile, runner: NmcliRunner = None
) -> WifiNetwork:
    run = runner or run_nmcli
    key_tokens = nmcli_key_mgmt_tokens(profile.key_mgmt)
    if key_tokens & ENTERPRISE_KEY_MGMT:
        raise WifiError(
            f"Profile '{profile.name}' uses enterprise Wi-Fi, which is not supported for QR export."
        )
    if key_tokens & WPA_KEY_MGMT:
        password = read_nmcli_password(
            profile.uuid, "802-11-wireless-security.psk", run
        )
        if not password:
            raise WifiError(f"Profile '{profile.name}' has no stored Wi-Fi password.")
        security = "WPA"
    elif key_tokens & WEP_OR_OPEN_KEY_MGMT:
        password = read_nmcli_password(
            profile.uuid, "802-11-wireless-security.wep-key0", run
        )
        security = "WEP" if password else "nopass"
    elif not key_tokens or key_tokens <= OPEN_KEY_MGMT:
        password = ""
        security = "nopass"
    else:
        raise WifiError(
            f"Profile '{profile.name}' uses unsupported Wi-Fi security '{profile.key_mgmt}'."
        )
    return WifiNetwork(
        ssid=profile.ssid,
        password=password,
        security=security,
        hidden=profile.hidden,
        profile_name=profile.name,
    )


def read_nmcli_password(uuid: str, field: str, runner: NmcliRunner = None) -> str:
    run = runner or run_nmcli
    output = run(["-s", "-g", field, "connection", "show", "uuid", uuid], True)
    values = output.splitlines()
    return values[0] if values else ""


def pick_nmcli_profile(profiles: Sequence[NetworkManagerProfile]) -> NetworkManagerProfile:
    if shutil.which("fzf"):
        picked = pick_nmcli_profile_fzf(profiles)
        if picked is not None:
            return picked
        raise WifiError("No Wi-Fi profile selected.")
    return pick_nmcli_profile_numbered(profiles)


def pick_nmcli_profile_fzf(
    profiles: Sequence[NetworkManagerProfile],
) -> Optional[NetworkManagerProfile]:
    labels = profile_labels(profiles)
    proc = subprocess.run(
        ["fzf", "--prompt=Wi-Fi profile> ", "--height=40%", "--layout=reverse"],
        input="\n".join(labels.keys()) + "\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=None,
        check=False,
    )
    selection = proc.stdout.strip()
    return labels.get(selection)


def pick_nmcli_profile_numbered(
    profiles: Sequence[NetworkManagerProfile],
) -> NetworkManagerProfile:
    labels = list(profile_labels(profiles).items())
    for idx, (label, _) in enumerate(labels, start=1):
        click_label = label.split("\t", 1)[1] if "\t" in label else label
        sys.stderr.write(f"{idx}. {click_label}\n")
    while True:
        raw = input("Wi-Fi profile: ").strip()
        try:
            choice = int(raw)
        except ValueError:
            sys.stderr.write("Enter a profile number.\n")
            continue
        if 1 <= choice <= len(labels):
            return labels[choice - 1][1]
        sys.stderr.write("Profile number out of range.\n")


def profile_labels(
    profiles: Sequence[NetworkManagerProfile],
) -> Dict[str, NetworkManagerProfile]:
    labels: Dict[str, NetworkManagerProfile] = {}
    for idx, profile in enumerate(profiles, start=1):
        ssid = profile.ssid or profile.name
        security = nmcli_security_label(profile.key_mgmt)
        detail = f"{ssid}\t{security}"
        if profile.hidden:
            detail += "\thidden"
        if profile.name != ssid:
            detail += f"\t{profile.name}"
        labels[f"{idx}\t{detail}"] = profile
    return labels


def nmcli_security_label(key_mgmt: str) -> str:
    tokens = nmcli_key_mgmt_tokens(key_mgmt)
    if tokens & WPA_KEY_MGMT:
        return "WPA"
    if tokens & ENTERPRISE_KEY_MGMT:
        return "enterprise"
    if tokens & WEP_OR_OPEN_KEY_MGMT:
        return "WEP/open"
    if not tokens or tokens <= OPEN_KEY_MGMT:
        return "open"
    return key_mgmt or "open"


def nmcli_key_mgmt_tokens(key_mgmt: str) -> set[str]:
    return set(key_mgmt.lower().replace(",", " ").split())


def run_nmcli(args: Sequence[str], include_passwords: bool = False) -> str:
    command = ["nmcli", *args]
    try:
        proc = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except FileNotFoundError as exc:
        raise WifiError("nmcli is required for stored Wi-Fi profile selection.") from exc
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or exc.stdout.strip() or "nmcli failed"
        if include_passwords:
            message = "nmcli failed while reading a stored Wi-Fi password"
        raise WifiError(message) from exc
    return proc.stdout


def _wrap_with_prefix(
    prefix: str, value: str, width: int, wrap_mode: str
) -> List[str]:
    if not value:
        return []
    if len(prefix) >= width:
        return _wrap_lines(prefix + value, width, wrap_mode)
    available = width - len(prefix)
    lines = _wrap_lines(value, available, wrap_mode)
    out = [prefix + lines[0]]
    pad = " " * len(prefix)
    for line in lines[1:]:
        out.append(pad + line)
    return out


def _wrap_lines(value: str, width: int, wrap_mode: str) -> List[str]:
    wrapped = wrap_text(value, width, wrap_mode)
    return wrapped.splitlines() if wrapped else [""]


def _clean_panel_value(value: str) -> str:
    return sanitize_output(value).encode("ascii", "ignore").decode("ascii").strip()


def _ensure_trailing_newline(text: str) -> str:
    return text if text.endswith("\n") else text + "\n"


def _unescape_nmcli(value: str) -> str:
    return value.replace(r"\:", ":").replace(r"\\", "\\")
