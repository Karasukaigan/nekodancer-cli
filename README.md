# NekoDancer CLI

<img src="./resources/logo.png" alt="logo" width="300">

Automatically generate Funscript files from audio and video files.

## Features

- **Batch processing** — Process entire directories with progress display
- **Multi-axis** — Up to 4 synchronized axes (v1.0 separate files, v2.0 single file with channels)
- **Performance** — ~80× real-time (RTF ~0.012), a 2-minute track in ~1.5 seconds
- **Formats** — MP3, WAV, MP4, FLAC, OGG, M4A

## Installation & Usage

```bash
pip install -r requirements.txt

# Process a single file
python cli.py music.mp3

# Process all media in a directory
python cli.py ./videos/ --license Paid

# Multi-axis output
python cli.py music.mp3 -ma 2

# Custom creator and description
python cli.py music.mp3 -c "MyName" -d "My description"
```

### Arguments

| Argument | Description |
|----------|-------------|
| `path` | Audio/video file or directory (non-recursive) |
| `-v, --version` | Show version |
| `-i, --inverted` | Invert position mapping |
| `-l, --license {Free,Paid}` | License type in metadata |
| `-t, --title TITLE` | Custom title (ignored for directories) |
| `-r, --range [0-100]` | Max position range |
| `--raw` | Skip servo post-processing |
| `-ma, --multiaxis [VERSION]` | Multi-axis output (1.0 or 2.0) |
| `-c, --creator [CREATOR]` | Custom creator name |
| `-d, --description [DESCRIPTION]` | Custom description |

## Output

Generates `.funscript` files next to the input file:
- Single axis: `song.funscript`
- Multi-axis v1.0: `song.funscript` + `song.twist.funscript` + `song.roll.funscript` + `song.pitch.funscript`
- Multi-axis v2.0: Single file with bundled channels

### Support Development

NekoDancer is open source and free to use. If you find it useful, consider purchasing the [pre-built executable](https://beyondblackwall.com/nekodancer) to support ongoing development.

`NekoDancer.exe` is a ready-to-run Windows CLI tool — no Python environment required. Just download, unzip, and use it directly from any command prompt or integrate it into your workflow.

## Contributing

[Issues](https://github.com/Karasukaigan/nekodancer-cli/issues) and [Pull Requests](https://github.com/Karasukaigan/nekodancer-cli/pulls) are welcome to help improve this project.

## License

This project is licensed under the [MIT License](./LICENSE.txt).
