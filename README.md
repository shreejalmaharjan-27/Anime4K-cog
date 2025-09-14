# Anime4K Cog
This is a [Cog](https://cog.run/) for the Anime4K image upscaling model. It supports multiple presets and uses GPU acceleration for fast processing.

The code for this repository was used from the [Anime4K PyTorch implementation](https://colab.research.google.com/drive/11xAn4fyAUJPZOjrxwnL2ipl_1DGGegkB).

You can view the main Anime4K repository [here](https://github.com/bloc97/Anime4K).


## Requirements
- Python 3.10+
- [Cog](https://cog.run/getting-started)
- A GPU for best performance (NVIDIA recommended)

## Replicate 
This model is available to run instantly on [Replicate](https://replicate.com/shreejalmaharjan-27/anime4k).


## Usage
To use this Cog, you can run the following command:

```bash
cog build
```

```bash
cog run -i image=@path_to_your_image.jpg 
```

### Presets
You can specify the preset using the `-i preset="Preset_Name"` argument. Available presets are:
- A (Fast)
- B (Balanced)
- C (Quality)

```bash
cog run -i image=@path_to_your_image.jpg -i preset="B (Balanced)"
```

### Resolution
You can also specify the target resolution using the `-i resolution=RESOLUTION` argument. Available resolutions are:
- 1080p
- 2K
- 4K
- 8K

```bash
cog run -i image=@path_to_your_image.jpg -i resolution=1080p
```

## Development
To set up a development environment, you can use the following commands:
```bash
cog build
```

```bash
bash debug-run.sh
```

### Browser Intellisense

- `conda create -n anime4k python=3.10`
- `conda activate anime4k`
- `pip install -r requirements.txt`

Then select the `anime4k` interpreter in VSCode.


## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements
- [Cog](https://cog.run/)
- [Anime4K](https://github.com/bloc97/Anime4K)