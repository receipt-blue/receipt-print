# receipt-print

A simple CLI tool for printing text on thermal receipt printers.

## Installation

```bash
pip install receipt-print
```

## Usage

Print piped text directly:
```bash
echo "hello" | receipt-print
```

Print the contents of files:
```bash
receipt-print cat file1.txt file2.txt
```

Print text directly:
```bash
receipt-print echo "hello" "world"
receipt-print echo -l "line 1" "line 2"  # use newlines instead of spaces
```

Count how many lines would be printed:
```bash
receipt-print count file.txt
man ls | receipt-print count
```

## Configuration

Configure using environment variables ([see python-escpos documentation](https://readthedocs.org/projects/python-envconfig/en/latest/)):

- `RP_VENDOR`: USB vendor ID (default: 04b8)
- `RP_PRODUCT`: USB product ID (default: 0e2a)
- `RP_PROFILE`: Printer profile (default: TM-T20II)
- `RP_HOST`: Network printer IP address (optional, fallback if USB fails)
- `RP_CHAR_WIDTH`: Character width per line (default: 72)
- `RP_MAX_LINES`: Maximum lines allowed without confirmation (default: 40)

