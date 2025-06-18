# receipt-print

A Unix-inspired CLI for printing content on thermal receipt printers.

## Installation

A global installation is recommended, e.g. with [uv](https://docs.astral.sh/uv/):
```bash
uv tool install receipt-print
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
```

Print file content:
```bash
receipt-print cat file1.txt file2.txt
```

Print images:
```bash
receipt-print image img1.png img2.png
cat image_paths.txt | receipt-print image
pbpaste | receipt-print image  # clipboard file or path
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

## Configuration

Configure using environment variables ([see python-escpos documentation](https://python-escpos.readthedocs.io/en/latest/user/usage.html)):

- `RP_VENDOR`: USB vendor ID (default: 04b8)
- `RP_PRODUCT`: USB product ID (default: 0e2a)
- `RP_PROFILE`: Printer profile (default: TM-T20II)
- `RP_HOST`: Network printer IP address (optional, fallback if USB fails)
- `RP_CHAR_WIDTH`: Character width per line (default: 72)
- `RP_MAX_LINES`: Maximum lines allowed without confirmation (default: 40)

