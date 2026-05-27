# receipt-print

A Unix-inspired CLI for printing content on thermal receipt printers.

## Installation

A global installation is recommended, e.g. with [uv](https://docs.astral.sh/uv/):
```bash
uv tool install receipt-print
```

### Linux USB permissions

USB receipt printers usually need a udev rule before an unprivileged user can
open them through libusb. For Epson printers:

```bash
sudo modprobe usblp
sudo tee /etc/udev/rules.d/70-receipt-print.rules >/dev/null <<'EOF'
SUBSYSTEM=="usb", ENV{DEVTYPE}=="usb_device", ATTR{idVendor}=="04b8", MODE="0660", GROUP="lp", TAG+="uaccess"
SUBSYSTEM=="usbmisc", KERNEL=="lp[0-9]*", ATTRS{idVendor}=="04b8", MODE="0660", GROUP="lp", TAG+="uaccess", SYMLINK+="receipt-printer"
EOF
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=usb --attr-match=idVendor=04b8
sudo udevadm trigger --subsystem-match=usbmisc
```

On headless systems, or systems without logind seat access, add your user to
the selected group and start a new login session:

```bash
sudo usermod -aG lp "$USER"
```

## Usage

Print piped text:
```bash
echo "hello" | receipt-print
cat file.txt | receipt-print
```

Print text:
```bash
receipt-print text "hello" "world"
receipt-print text -l "line 1" "line 2"  # use newlines instead of spaces
receipt-print text --size 3 "big text"
```

Print file content:
```bash
receipt-print cat file1.txt file2.txt
```

Print images:
```bash
receipt-print image img1.png img2.png
receipt-print image https://example.com/photo.jpg
cat image_paths.txt | receipt-print image
pbpaste | receipt-print image  # clipboard file or path
receipt-print image --tile 5 mural.png  # print a set of sequential vertical segments
```

Print PDFs:
```bash
receipt-print pdf document.pdf
receipt-print pdf --pages 1,3,5 document.pdf

# print from clipboard (path or file)
pbpaste | receipt-print pdf  # or paste from `wl-paste` / `xclip -selection clipboard -o`

# convert to text first
receipt-print pdf --format text document.pdf
```

Run command(s) and print captured output:
```bash
receipt-print shell "ls -l" "git status"
receipt-print shell "hostname -I | awk '{print $1}'"
```

Skip cutting:
```bash
receipt-print text --no-cut "hello"
cat file.txt | receipt-print --no-cut
RP_NO_CUT=1 receipt-print image img1.png
```

## Configuration

Configure using environment variables ([see python-escpos documentation](https://python-escpos.readthedocs.io/en/latest/user/usage.html)):

- `RP_VENDOR`: USB vendor ID (default: 04b8)
- `RP_PRODUCT`: Optional exact USB product ID. If unset, `auto`, or `*`, the first USB printer matching `RP_VENDOR` is used.
- `RP_USB_AUTO_DISCOVER`: Set to `0` to disable product-ID fallback discovery after an exact `RP_PRODUCT` miss.
- `RP_DEVICE`: Exact device file path (e.g., `/dev/receipt-printer` or `/dev/usb/lp0`). Uses the kernel driver instead of libusb.
- `RP_DEVICE_AUTO_DISCOVER`: Set to `0` to skip automatic `/dev/receipt-printer*` and `/dev/usb/lp*` discovery.
- `RP_DEVICE_CANDIDATES`: Comma-separated device paths to try before libusb when `RP_DEVICE` is unset.
- `RP_PROFILE`: Printer profile (default: TM-T20II)
- `RP_HOST`: Network printer IP address (optional, fallback if USB fails)
- `RP_CHAR_WIDTH`: Character width per line (default: 72)
- `RP_MAX_LINES`: Maximum lines allowed without confirmation (default: 40)
- `RP_NO_CUT`: Set to `1` to disable automatic cutting
