from click.testing import CliRunner
from escpos.escpos import QR_ECLEVEL_M

from receipt_print import cli as cli_module
from receipt_print.wifi import (
    MacosWifiProfile,
    NetworkManagerProfile,
    WifiNetwork,
    build_wifi_panel,
    build_wifi_qr_payload,
    list_macos_wifi_profiles,
    list_nmcli_wifi_profiles,
    macos_wifi_device,
    parse_nmcli_connection_line,
    read_macos_wifi_password,
    wifi_network_from_macos_profile,
    wifi_network_from_nmcli_profile,
)


TEST_SSID = "ExampleNet"
TEST_PASSWORD = "example-passphrase"
UNUSED_SSID = "unused-example-net"


class FakePrinter:
    def __init__(self):
        self.calls = []

    def set(self, **kwargs):
        self.calls.append(("set", kwargs))

    def text(self, value):
        self.calls.append(("text", value))

    def qr(self, value, size, ec):
        self.calls.append(("qr", value, size, ec))

    def cut(self):
        self.calls.append(("cut",))

    def close(self):
        self.calls.append(("close",))


def test_build_wifi_qr_payload_escapes_values():
    payload = build_wifi_qr_payload(
        WifiNetwork(
            ssid='Cafe;Guest:5G',
            password=r'p\,a:ss";word',
            security="WPA",
            hidden=True,
        )
    )

    assert payload == r'WIFI:T:WPA;S:Cafe\;Guest\:5G;P:p\\\,a\:ss\"\;word;H:true;;'


def test_build_wifi_qr_payload_open_network():
    payload = build_wifi_qr_payload(WifiNetwork(ssid="Guest", security="nopass"))

    assert payload == "WIFI:T:nopass;S:Guest;;"


def test_wifi_panel_prints_password_above_security():
    panel = build_wifi_panel(
        WifiNetwork(ssid=TEST_SSID, password=TEST_PASSWORD, security="WPA")
    )

    text = "\n".join(panel.lines)
    assert TEST_SSID in text
    assert f"password: {TEST_PASSWORD}" in text
    assert text.index(f"password: {TEST_PASSWORD}") < text.index("security: WPA")


def test_wifi_panel_omits_password_for_open_network():
    panel = build_wifi_panel(WifiNetwork(ssid="Guest", security="nopass"))

    text = "\n".join(panel.lines)
    assert "password:" not in text
    assert "security: open" in text


def test_wifi_panel_can_omit_password_text():
    panel = build_wifi_panel(
        WifiNetwork(ssid=TEST_SSID, password=TEST_PASSWORD, security="WPA"),
        print_password=False,
    )

    text = "\n".join(panel.lines)
    assert TEST_SSID in text
    assert "password:" not in text
    assert TEST_PASSWORD not in text
    assert "security: WPA" in text


def test_parse_nmcli_connection_line_keeps_colons_in_name():
    parsed = parse_nmcli_connection_line(
        r"abc:802-11-wireless:Kitchen\:Printer:5G"
    )

    assert parsed == ("abc", "802-11-wireless", "Kitchen:Printer:5G")


def test_list_nmcli_wifi_profiles_filters_and_loads_details():
    def runner(args, include_passwords=False):
        if args == ["-t", "-f", "UUID,TYPE,NAME", "connection", "show"]:
            return "\n".join(
                [
                    f"wifi-1:802-11-wireless:{TEST_SSID}",
                    "eth-1:802-3-ethernet:Wired",
                ]
            )
        assert args == [
            "-g",
            "802-11-wireless.ssid,802-11-wireless-security.key-mgmt,802-11-wireless.hidden",
            "connection",
            "show",
            "uuid",
            "wifi-1",
        ]
        return f"{TEST_SSID}\nwpa-psk\nno\n"

    profiles = list_nmcli_wifi_profiles(runner)

    assert profiles == [
        NetworkManagerProfile(
            uuid="wifi-1",
            name=TEST_SSID,
            ssid=TEST_SSID,
            key_mgmt="wpa-psk",
            hidden=False,
        )
    ]


def test_wifi_network_from_nmcli_profile_reads_wpa_password():
    profile = NetworkManagerProfile(
        uuid="wifi-1",
        name=TEST_SSID,
        ssid=TEST_SSID,
        key_mgmt="wpa-psk",
        hidden=False,
    )

    def runner(args, include_passwords=False):
        assert include_passwords is True
        assert args == [
            "-s",
            "-g",
            "802-11-wireless-security.psk",
            "connection",
            "show",
            "uuid",
            "wifi-1",
        ]
        return f"{TEST_PASSWORD}\n"

    network = wifi_network_from_nmcli_profile(profile, runner)

    assert network == WifiNetwork(
        ssid=TEST_SSID,
        password=TEST_PASSWORD,
        security="WPA",
        hidden=False,
        profile_name=TEST_SSID,
    )


def test_cli_wifi_manual_prints_password_by_default(monkeypatch):
    printer = FakePrinter()
    monkeypatch.setattr(cli_module, "connect_printer", lambda: printer)

    result = CliRunner().invoke(
        cli_module.cli,
        ["wifi", "--ssid", TEST_SSID, "--password", TEST_PASSWORD, "--no-cut"],
    )

    assert result.exit_code == 0
    assert (
        "qr",
        f"WIFI:T:WPA;S:{TEST_SSID};P:{TEST_PASSWORD};;",
        4,
        QR_ECLEVEL_M,
    ) in printer.calls
    printed_text = "".join(call[1] for call in printer.calls if call[0] == "text")
    assert TEST_SSID in printed_text
    assert f"password: {TEST_PASSWORD}" in printed_text
    assert ("cut",) not in printer.calls
    assert printer.calls[-1] == ("close",)


def test_cli_wifi_omit_password_keeps_password_in_qr_only(monkeypatch):
    printer = FakePrinter()
    monkeypatch.setattr(cli_module, "connect_printer", lambda: printer)

    result = CliRunner().invoke(
        cli_module.cli,
        [
            "wifi",
            "--ssid",
            TEST_SSID,
            "--password",
            TEST_PASSWORD,
            "--omit-password",
            "--no-cut",
        ],
    )

    assert result.exit_code == 0
    assert (
        "qr",
        f"WIFI:T:WPA;S:{TEST_SSID};P:{TEST_PASSWORD};;",
        4,
        QR_ECLEVEL_M,
    ) in printer.calls
    printed_text = "".join(call[1] for call in printer.calls if call[0] == "text")
    assert TEST_SSID in printed_text
    assert "password:" not in printed_text
    assert TEST_PASSWORD not in printed_text


def test_cli_wifi_manual_prompts_for_password(monkeypatch):
    printer = FakePrinter()
    monkeypatch.setattr(cli_module, "connect_printer", lambda: printer)

    result = CliRunner().invoke(
        cli_module.cli,
        ["wifi", "--ssid", TEST_SSID, "--no-cut"],
        input=f"{TEST_PASSWORD}\n",
    )

    assert result.exit_code == 0
    assert (
        "qr",
        f"WIFI:T:WPA;S:{TEST_SSID};P:{TEST_PASSWORD};;",
        4,
        QR_ECLEVEL_M,
    ) in printer.calls


def test_macos_wifi_device_parses_hardware_ports():
    def runner(args):
        assert args == ["-listallhardwareports"]
        return "\n".join(
            [
                "Hardware Port: Ethernet",
                "Device: en4",
                "",
                "Hardware Port: Wi-Fi",
                "Device: en0",
                "Ethernet Address: aa:bb:cc:dd:ee:ff",
            ]
        )

    assert macos_wifi_device(runner) == "en0"


def test_list_macos_wifi_profiles_parses_preferred_networks():
    def runner(args):
        if args == ["-listallhardwareports"]:
            return "Hardware Port: Wi-Fi\nDevice: en0\n"
        assert args == ["-listpreferredwirelessnetworks", "en0"]
        return "\n".join(
            [
                "Preferred networks on en0:",
                f"\t{TEST_SSID}",
                "\tGuest",
            ]
        )

    profiles = list_macos_wifi_profiles(runner)

    assert profiles == [
        MacosWifiProfile(ssid=TEST_SSID),
        MacosWifiProfile(ssid="Guest"),
    ]


def test_read_macos_wifi_password_returns_stored_password():
    def runner(args):
        assert args == [
            "find-generic-password",
            "-D",
            "AirPort network password",
            "-a",
            TEST_SSID,
            "-w",
        ]
        return f"{TEST_PASSWORD}\n"

    assert read_macos_wifi_password(TEST_SSID, runner) == TEST_PASSWORD


def test_wifi_network_from_macos_profile_with_password_is_wpa():
    profile = MacosWifiProfile(ssid=TEST_SSID)

    def runner(args):
        return TEST_PASSWORD

    network = wifi_network_from_macos_profile(profile, runner)

    assert network == WifiNetwork(
        ssid=TEST_SSID,
        password=TEST_PASSWORD,
        security="WPA",
        hidden=False,
        profile_name=TEST_SSID,
    )


def test_wifi_network_from_macos_profile_without_password_is_open():
    profile = MacosWifiProfile(ssid="Guest")

    def runner(args):
        return ""

    network = wifi_network_from_macos_profile(profile, runner)

    assert network == WifiNetwork(
        ssid="Guest",
        password="",
        security="nopass",
        hidden=False,
        profile_name="Guest",
    )


def test_cli_wifi_without_ssid_uses_stored_profile_selector(monkeypatch):
    printer = FakePrinter()
    monkeypatch.setattr(cli_module, "connect_printer", lambda: printer)
    monkeypatch.setattr(
        cli_module,
        "select_stored_wifi_network",
        lambda: WifiNetwork(ssid=TEST_SSID, password=TEST_PASSWORD),
    )

    result = CliRunner().invoke(cli_module.cli, ["wifi", "--no-cut"])

    assert result.exit_code == 0
    assert (
        "qr",
        f"WIFI:T:WPA;S:{TEST_SSID};P:{TEST_PASSWORD};;",
        4,
        QR_ECLEVEL_M,
    ) in printer.calls


def test_cli_wifi_rejects_empty_ssid(monkeypatch):
    monkeypatch.setattr(
        cli_module,
        "select_stored_wifi_network",
        lambda: WifiNetwork(ssid=UNUSED_SSID, password=TEST_PASSWORD),
    )

    result = CliRunner().invoke(cli_module.cli, ["wifi", "--ssid", ""])

    assert result.exit_code != 0
    assert "--ssid cannot be empty" in result.output
