# Frameloop

A CLI tool for generating videos and images using AI models from Replicate.

![frameloop](https://github.com/user-attachments/assets/986356c1-ce5f-4101-84ef-45509d030181)


```bash
# Generate a video from an image
frameloop video photo.jpg -p "camera slowly zooms in"

# Generate an image
frameloop image "a cat wearing a tiny hat" -a 16:9

# Upscale an image 4x
frameloop upscale photo.jpg --face-enhance
```

## Installation

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/YOUR_USERNAME/frameloop.git
cd frameloop
uv sync
```

Set your Replicate API token:

```bash
export REPLICATE_API_TOKEN="your_token_here"
```

## Usage

### Video Generation

```bash
# Basic image-to-video (uses wan model by default - fast & cheap)
frameloop video input.jpg -p "camera pans right slowly"

# Use a specific model
frameloop video input.jpg -p "dramatic zoom" -m seedance

# Google Veo with start and end frame interpolation
frameloop video start.jpg -p "smooth transition" -m veo -e end.jpg

# Minimax Hailuo for cinematic quality
frameloop video -p "ocean waves crashing" -m hailuo
```

**Available video models:**

| Model | Best for | Cost |
|-------|----------|------|
| `wan` | Fast & cheap | ~$0.09/video |
| `seedance` | Multi-resolution | ~$0.06/sec (1080p) |
| `veo` | Frame interpolation | - |
| `kling` | High quality | ~$0.09/sec (pro) |
| `minimax-live` | Smooth 2D animation | - |
| `hailuo` | Cinematic quality | ~$0.28-0.56/video |

### Image Generation

```bash
# Generate an image
frameloop image "a futuristic city at sunset" -a 16:9

# Use GPT Image for text rendering
frameloop image "a sign that says HELLO" -m gpt-image

# Transform existing images
frameloop image "make it look like a painting" -i photo.jpg -m nano-banana

# Generate multiple images
frameloop image "abstract art" -n 4 -m seedream
```

**Available image models:**

| Model | Best for | Cost |
|-------|----------|------|
| `seedream` | Fast, general purpose | ~$0.03/image |
| `gpt-image` | Text rendering, editing | ~$0.02/image |
| `nano-banana` | Image transformation | ~$0.15/image |
| `minimax-image` | Face reference support | ~$0.01/image |

### Upscaling

```bash
# 4x upscale
frameloop upscale photo.jpg

# 2x upscale with face enhancement
frameloop upscale portrait.jpg -s 2 --face-enhance
```

### Utility Commands

```bash
# Extract frames from video
frameloop extract-frame video.mp4              # First frame
frameloop extract-frame video.mp4 --last       # Last frame
frameloop extract-frame video.mp4 -f 30        # Frame 30

# Convert to social-media friendly MP4
frameloop convert video.mp4

# List all models
frameloop models

# Show model details
frameloop info seedance

# Check async prediction status
frameloop status prediction_id_here
```

### Common Options

```bash
-o, --output      # Custom output path
-a, --aspect-ratio # Aspect ratio (16:9, 9:16, 1:1, etc.)
-r, --resolution  # Resolution (480p, 720p, 1080p, 2K, 4K)
-d, --duration    # Video duration in seconds
--no-wait         # Submit and exit without waiting for result
```

## Requirements

- Python 3.11+
- [Replicate API token](https://replicate.com/account/api-tokens)
- FFmpeg (for `extract-frame` and `convert` commands)

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

## License

MIT
