"""CLI entry point for Frameloop."""

import os
import subprocess
import typer
from pathlib import Path
from rich.console import Console
from rich.table import Table

from .models import get_model, get_models_by_type, list_models
from .runner import run_prediction, get_prediction_status
from .utils import generate_output_filename
from .prompts import select_model, prompt_param, interactive_config

app = typer.Typer(name="frameloop", help="CLI tool for running Replicate AI models", no_args_is_help=True)
console = Console()


def _check_token():
    if not os.environ.get("REPLICATE_API_TOKEN"):
        console.print("[red]Error:[/] REPLICATE_API_TOKEN not set")
        raise typer.Exit(1)


def _run(model_config: dict, inputs: dict, output: str | None, no_wait: bool):
    try:
        run_prediction(model_config, inputs, output_path=output, wait=not no_wait)
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/] {e}")
        raise typer.Exit(1)


def _check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except FileNotFoundError:
        console.print("[red]Error:[/] ffmpeg not found. Install with: brew install ffmpeg")
        raise typer.Exit(1)


def _get_frame_count(video_path: str) -> int:
    """Get total frame count from video using ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", video_path],
        capture_output=True, text=True
    )
    if result.returncode != 0 or not result.stdout.strip():
        return -1
    # Handle trailing comma from csv output
    return int(result.stdout.strip().rstrip(','))


@app.command("extract-frame")
def extract_frame(
    video: str = typer.Argument(..., help="Path to input video"),
    last: bool = typer.Option(False, "--last", "-l", help="Extract last frame"),
    frame: int | None = typer.Option(None, "--frame", "-f", help="Extract specific frame number"),
    output: str | None = typer.Option(None, "-o", "--output", help="Output file path"),
):
    """Extract a frame from a video."""
    _check_ffmpeg()

    video_path = Path(video)
    if not video_path.exists():
        console.print(f"[red]Error:[/] Video not found: {video}")
        raise typer.Exit(1)

    output_path = output or generate_output_filename(video, "frame", "png")

    if last:
        # Get exact frame count and extract the last one
        console.print(f"[dim]Counting frames in {video_path.name}...[/]")
        frame_count = _get_frame_count(video)
        if frame_count < 1:
            console.print("[red]Error:[/] Could not determine frame count")
            raise typer.Exit(1)
        target_frame = frame_count - 1
        console.print(f"[dim]Extracting frame {target_frame + 1}/{frame_count}...[/]")
        cmd = ["ffmpeg", "-y", "-i", video, "-vf", f"select=eq(n\\,{target_frame})", "-vframes", "1", "-q:v", "2", output_path]
    elif frame is not None:
        console.print(f"[dim]Extracting frame {frame} from {video_path.name}...[/]")
        cmd = ["ffmpeg", "-y", "-i", video, "-vf", f"select=eq(n\\,{frame})", "-vframes", "1", "-q:v", "2", output_path]
    else:
        console.print(f"[dim]Extracting first frame from {video_path.name}...[/]")
        cmd = ["ffmpeg", "-y", "-i", video, "-vframes", "1", "-q:v", "2", output_path]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        console.print(f"[red]Error:[/] {result.stderr}")
        raise typer.Exit(1)

    console.print(f"[green]✓[/] Saved to {output_path}")


@app.command()
def convert(
    video: str = typer.Argument(..., help="Path to input video"),
    social: bool = typer.Option(True, "--social/--no-social", help="Optimize for social media"),
    quality: int = typer.Option(22, "-q", "--quality", help="CRF quality (lower=better, 18-28)"),
    output: str | None = typer.Option(None, "-o", "--output", help="Output file path"),
):
    """Convert video to social-friendly MP4."""
    _check_ffmpeg()

    video_path = Path(video)
    if not video_path.exists():
        console.print(f"[red]Error:[/] Video not found: {video}")
        raise typer.Exit(1)

    output_path = output or generate_output_filename(video, "converted", "mp4")

    cmd = [
        "ffmpeg", "-y", "-i", video,
        "-c:v", "libx264", "-preset", "slow", "-crf", str(quality),
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",  # Enable streaming
        "-pix_fmt", "yuv420p",  # Max compatibility
        output_path
    ]

    console.print(f"[dim]Converting {video_path.name} to social-friendly MP4...[/]")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        console.print(f"[red]Error:[/] {result.stderr}")
        raise typer.Exit(1)

    # Show file size comparison
    orig_size = video_path.stat().st_size / (1024 * 1024)
    new_size = Path(output_path).stat().st_size / (1024 * 1024)
    console.print(f"[green]✓[/] Saved to {output_path}")
    console.print(f"[dim]Size: {orig_size:.1f}MB → {new_size:.1f}MB[/]")


@app.command()
def video(
    image: str = typer.Argument(None, help="Path to input image or URL"),
    prompt: str = typer.Option(None, "-p", "--prompt", help="Text prompt for video generation"),
    model: str = typer.Option(None, "-m", "--model", help="Model: wan, seedance, veo, kling, kling-2.5-turbo, kling-2.6, minimax-live, hailuo, sora"),
    end_frame: str | None = typer.Option(None, "-e", "--end", help="End frame for interpolation"),
    duration: int = typer.Option(None, "-d", "--duration", help="Video length in seconds"),
    resolution: str = typer.Option(None, "-r", "--resolution", help="Output resolution"),
    aspect_ratio: str = typer.Option(None, "-a", "--aspect-ratio", help="Aspect ratio: 16:9, 9:16, 1:1"),
    fast_mode: str = typer.Option(None, "-f", "--fast-mode", help="Speed: Off, Balanced, Fast (wan only)"),
    optimize_prompt: bool = typer.Option(True, "--optimize/--no-optimize", help="Enable AI prompt optimization"),
    negative_prompt: str = typer.Option("", "-n", "--negative", help="Things to exclude from video"),
    audio: bool = typer.Option(True, "--audio/--no-audio", help="Generate audio (kling-2.6 only)"),
    output: str | None = typer.Option(None, "-o", "--output", help="Output file path"),
    no_wait: bool = typer.Option(False, "--no-wait", help="Submit and exit without waiting"),
    seed: int | None = typer.Option(None, "-s", "--seed", help="Random seed"),
):
    """Generate video from an image."""
    _check_token()

    # Auto-prompt for model if not provided
    if not model:
        model = select_model("video")
        if not model:
            raise typer.Exit(1)

    model_config = get_model(model)
    if not model_config or model_config["type"] != "video":
        console.print(f"[red]Error:[/] Unknown video model: {model}")
        raise typer.Exit(1)

    params = model_config["params"]

    # Auto-prompt for prompt if not provided
    if not prompt:
        import questionary
        prompt = questionary.text("Prompt (describe the video):").ask()
        if not prompt:
            console.print("[red]Error:[/] Prompt is required")
            raise typer.Exit(1)

    # Auto-prompt for image if required and not provided
    requires_image = any(params.get(p, {}).get("required") for p in ["image", "start_image", "first_frame_image", "input_reference"])
    if requires_image and not image:
        import questionary
        image = questionary.path("Input image (required):").ask()
        if not image:
            console.print(f"[red]Error:[/] {model} requires an input image")
            raise typer.Exit(1)

    # Auto-prompt for optional start/input image if supported and not provided
    if not image:
        import questionary
        # Check for any optional image input param
        for img_param in ["image", "start_image", "first_frame_image", "input_reference"]:
            if img_param in params and not params[img_param].get("required"):
                if questionary.confirm("Add start image?", default=False).ask():
                    image = questionary.path("Start image:").ask()
                break

    # Auto-prompt for optional end image if supported and not provided
    if not end_frame:
        import questionary
        # Check for any optional end frame param
        for end_param in ["end_image", "last_frame"]:
            if end_param in params and not params[end_param].get("required"):
                if questionary.confirm("Add end image?", default=False).ask():
                    end_frame = questionary.path("End image:").ask()
                break

    # Auto-prompt for params with choices when not provided via CLI
    for param_name, param_config in params.items():
        if param_name in ["prompt", "image", "start_image", "first_frame_image", "input_reference"]:
            continue
        if param_config.get("choices"):
            if param_name in ["duration", "seconds"] and not duration:
                duration = prompt_param(param_name, param_config)
            elif param_name == "resolution" and not resolution:
                resolution = prompt_param(param_name, param_config)
            elif param_name == "aspect_ratio" and not aspect_ratio:
                aspect_ratio = prompt_param(param_name, param_config)
            elif param_name == "fast_mode" and not fast_mode:
                fast_mode = prompt_param(param_name, param_config)

    inputs = {"prompt": prompt}

    if "aspect_ratio" in params:
        ar = aspect_ratio or params["aspect_ratio"].get("default", "16:9")
        inputs["aspect_ratio"] = ar
    if "image" in params and image:
        inputs["image"] = image
    if "start_image" in params and image:
        inputs["start_image"] = image
    if "first_frame_image" in params and image:
        inputs["first_frame_image"] = image
    if "input_reference" in params and image:
        inputs["input_reference"] = image
    if "last_frame" in params and end_frame:
        inputs["last_frame"] = end_frame
    if "end_image" in params and end_frame:
        inputs["end_image"] = end_frame
    if "duration" in params:
        dur = duration or params["duration"].get("default", 5)
        # Veo only accepts 4, 6, 8 - map default 5 to closest valid
        if model == "veo" and dur == 5:
            dur = 6
        # Hailuo: 1080p only supports 6s
        res = resolution or params.get("resolution", {}).get("default", "1080p")
        if model == "hailuo" and res == "1080p" and dur != 6:
            console.print("[yellow]Warning:[/] 1080p only supports 6s duration, adjusting")
            dur = 6
        inputs["duration"] = dur
    if "seconds" in params:
        # Sora uses "seconds" instead of "duration"
        sec = duration or params["seconds"].get("default", 4)
        inputs["seconds"] = sec
    if "resolution" in params:
        res = resolution or params["resolution"].get("default", "1080p")
        inputs["resolution"] = res
    if "mode" in params:
        res = resolution or params.get("resolution", {}).get("default", "1080p")
        inputs["mode"] = "pro" if res == "1080p" else "standard"
    if "fast_mode" in params:
        fm = fast_mode or params["fast_mode"].get("default", "Fast")
        inputs["fast_mode"] = fm
    if "prompt_optimizer" in params:
        inputs["prompt_optimizer"] = optimize_prompt
    if "negative_prompt" in params and negative_prompt:
        inputs["negative_prompt"] = negative_prompt
    if "generate_audio" in params:
        inputs["generate_audio"] = audio
    if seed is not None and "seed" in params:
        inputs["seed"] = seed

    # Handle optional API keys from env vars
    if "openai_api_key" in params:
        import os
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            inputs["openai_api_key"] = api_key

    _run(model_config, inputs, output, no_wait)


@app.command()
def image(
    prompt: str = typer.Argument(None, help="Text description of the image to generate"),
    images: list[str] | None = typer.Option(None, "-i", "--image", help="Input image(s) for transformation"),
    model: str = typer.Option(None, "-m", "--model", help="Model: seedream, nano-banana-pro, gpt-image, minimax-image, recraft-svg, recraft-20b"),
    aspect_ratio: str = typer.Option(None, "-a", "--aspect-ratio", help="Output aspect ratio"),
    resolution: str = typer.Option(None, "-r", "--resolution", help="Output resolution: 1K, 2K, 4K"),
    width: int | None = typer.Option(None, "-W", "--width", help="Custom width 1024-4096 (seedream only)"),
    height: int | None = typer.Option(None, "-H", "--height", help="Custom height 1024-4096 (seedream only)"),
    output_format: str = typer.Option(None, "-f", "--format", help="Output format: jpg, png, webp"),
    enhance: bool = typer.Option(True, "--enhance/--no-enhance", help="Enable prompt enhancement"),
    quality: str = typer.Option("auto", "-q", "--quality", help="Quality: low, medium, high, auto (gpt-image)"),
    background: str = typer.Option("auto", "-b", "--background", help="Background: auto, transparent, opaque (gpt-image)"),
    num_images: int = typer.Option(1, "-n", "--num-images", help="Number of images to generate"),
    face_ref: str | None = typer.Option(None, "--face-ref", help="Face reference image (minimax-image)"),
    style: str = typer.Option(None, "-s", "--style", help="Visual style (recraft models)"),
    output: str | None = typer.Option(None, "-o", "--output", help="Output file path"),
    no_wait: bool = typer.Option(False, "--no-wait", help="Submit and exit without waiting"),
):
    """Generate or transform images."""
    _check_token()

    # Auto-prompt for model if not provided
    if not model:
        model = select_model("image")
        if not model:
            raise typer.Exit(1)

    model_config = get_model(model)
    if not model_config or model_config["type"] != "image":
        console.print(f"[red]Error:[/] Unknown image model: {model}")
        raise typer.Exit(1)

    params = model_config["params"]

    # Auto-prompt for reference images first (so user knows what to reference in prompt)
    if not images and ("image_input" in params or "input_images" in params):
        import questionary
        if questionary.confirm("Add reference image(s)?", default=False).ask():
            images = []
            while True:
                img = questionary.path(f"Image {len(images) + 1} path:").ask()
                if not img:
                    break
                images.append(img)
                if not questionary.confirm("Add another image?", default=False).ask():
                    break
            if images:
                console.print(f"[dim]{len(images)} image(s) added — reference as Image 1, Image 2, etc. in your prompt[/]")

    # Auto-prompt for prompt if not provided
    if not prompt:
        import questionary
        prompt = questionary.text("Prompt (describe the image):").ask()
        if not prompt:
            console.print("[red]Error:[/] Prompt is required")
            raise typer.Exit(1)

    # Auto-prompt for params with choices when not provided via CLI
    for param_name, param_config in params.items():
        if param_name == "prompt":
            continue
        if param_config.get("choices"):
            # Check if user provided this param via CLI
            if param_name == "style" and not style:
                style = prompt_param(param_name, param_config)
            elif param_name == "aspect_ratio" and not aspect_ratio:
                aspect_ratio = prompt_param(param_name, param_config)
            elif param_name == "size" and not resolution:
                resolution = prompt_param(param_name, param_config)

    inputs = {"prompt": prompt}

    # Handle custom dimensions (seedream only)
    if width and height and "width" in params:
        inputs["size"] = "custom"
        inputs["width"] = width
        inputs["height"] = height
    elif width or height:
        if "width" not in params:
            console.print(f"[yellow]Warning:[/] Custom dimensions not supported by {model}, using aspect ratio")
        else:
            console.print("[red]Error:[/] Both --width and --height required for custom dimensions")
            raise typer.Exit(1)
    else:
        # Handle aspect ratio (only when not using custom dimensions)
        if "aspect_ratio" in params:
            ar = aspect_ratio or params["aspect_ratio"].get("default", "16:9")
            inputs["aspect_ratio"] = ar

        # Handle resolution/size (seedream uses 'size', nano-banana uses 'resolution')
        if "size" in params:
            res = resolution or params["size"].get("default", "2K")
            inputs["size"] = res
        elif "resolution" in params:
            res = resolution or params["resolution"].get("default", "2K")
            inputs["resolution"] = res

    # Handle output format
    if "output_format" in params:
        if output_format:
            inputs["output_format"] = output_format
        else:
            inputs["output_format"] = params["output_format"]["default"]

    # Handle prompt enhancement/optimization
    if "enhance_prompt" in params:
        inputs["enhance_prompt"] = enhance
    if "prompt_optimizer" in params:
        inputs["prompt_optimizer"] = enhance

    # Handle quality (gpt-image)
    if "quality" in params:
        inputs["quality"] = quality

    # Handle background (gpt-image)
    if "background" in params:
        inputs["background"] = background

    # Handle number of images
    if "number_of_images" in params:
        inputs["number_of_images"] = num_images
    if "max_images" in params and num_images > 1:
        inputs["max_images"] = num_images

    # Handle face reference (minimax-image)
    if face_ref and "subject_reference" in params:
        inputs["subject_reference"] = face_ref

    # Handle style (recraft-svg)
    if "style" in params:
        s = style or params["style"].get("default", "any")
        inputs["style"] = s

    # Handle input images (nano-banana-pro uses image_input, gpt-image uses input_images)
    if images:
        if "image_input" in params:
            inputs["image_input"] = images
        elif "input_images" in params:
            inputs["input_images"] = images

    _run(model_config, inputs, output, no_wait)


@app.command()
def upscale(
    image: str = typer.Argument(None, help="Path to input image or URL"),
    model: str = typer.Option("esrgan", "-m", "--model", help="Model to use"),
    scale: int = typer.Option(None, "-s", "--scale", help="Upscale factor: 2 or 4"),
    face_enhance: bool = typer.Option(False, "-f", "--face-enhance", help="Enhance faces"),
    output: str | None = typer.Option(None, "-o", "--output", help="Output file path"),
    no_wait: bool = typer.Option(False, "--no-wait", help="Submit and exit without waiting"),
):
    """Upscale an image."""
    _check_token()

    model_config = get_model(model)
    if not model_config or model_config["type"] != "upscale":
        console.print(f"[red]Error:[/] Unknown upscale model: {model}")
        raise typer.Exit(1)

    params = model_config["params"]

    # Auto-prompt for image if not provided
    if not image:
        import questionary
        image = questionary.path("Input image:").ask()
        if not image:
            console.print("[red]Error:[/] Image is required")
            raise typer.Exit(1)

    # Auto-prompt for scale if not provided
    if scale is None:
        scale = prompt_param("scale", params["scale"])

    inputs = {"image": image, "scale": scale, "face_enhance": face_enhance}
    _run(model_config, inputs, output, no_wait)


@app.command("remove-bg")
def remove_bg(
    image: str = typer.Argument(None, help="Path to input image or URL"),
    model: str = typer.Option("recraft-bg", "-m", "--model", help="Model: recraft-bg"),
    output: str | None = typer.Option(None, "-o", "--output", help="Output file path"),
    no_wait: bool = typer.Option(False, "--no-wait", help="Submit and exit without waiting"),
):
    """Remove background from an image."""
    _check_token()

    model_config = get_model(model)
    if not model_config or model_config["type"] != "transform":
        console.print(f"[red]Error:[/] Unknown transform model: {model}")
        raise typer.Exit(1)

    # Auto-prompt for image if not provided
    if not image:
        import questionary
        image = questionary.path("Input image:").ask()
        if not image:
            console.print("[red]Error:[/] Image is required")
            raise typer.Exit(1)

    inputs = {"image": image}
    _run(model_config, inputs, output, no_wait)


@app.command()
def mesh(
    image: str = typer.Argument(None, help="Path to input image or URL"),
    model: str = typer.Option("hunyuan3d", "-m", "--model", help="Model: hunyuan3d"),
    steps: int = typer.Option(None, "-s", "--steps", help="Inference steps (20-50)"),
    guidance_scale: float = typer.Option(None, "-g", "--guidance-scale", help="Guidance scale (1-20)"),
    resolution: int = typer.Option(None, "-r", "--resolution", help="Octree resolution: 256, 384, 512"),
    remove_background: bool = typer.Option(True, "--remove-bg/--no-remove-bg", help="Remove background from input"),
    seed: int | None = typer.Option(None, "--seed", help="Random seed"),
    output: str | None = typer.Option(None, "-o", "--output", help="Output file path"),
    no_wait: bool = typer.Option(False, "--no-wait", help="Submit and exit without waiting"),
):
    """Generate 3D mesh from an image."""
    _check_token()

    model_config = get_model(model)
    if not model_config or model_config["type"] != "3d":
        console.print(f"[red]Error:[/] Unknown 3D model: {model}")
        raise typer.Exit(1)

    params = model_config["params"]

    # Auto-prompt for image if not provided
    if not image:
        import questionary
        image = questionary.path("Input image:").ask()
        if not image:
            console.print("[red]Error:[/] Image is required")
            raise typer.Exit(1)

    # Auto-prompt for resolution if not provided
    if resolution is None:
        resolution = prompt_param("octree_resolution", params["octree_resolution"])

    inputs = {"image": image}
    inputs["steps"] = steps or params["steps"]["default"]
    inputs["guidance_scale"] = guidance_scale or params["guidance_scale"]["default"]
    inputs["octree_resolution"] = resolution
    inputs["remove_background"] = remove_background
    if seed is not None:
        inputs["seed"] = seed

    _run(model_config, inputs, output, no_wait)


@app.command()
def models():
    """List available models."""
    table = Table(title="Available Models")
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Description")

    for name, model_type, description in list_models():
        table.add_row(name, model_type, description)

    console.print(table)


@app.command()
def info(model_name: str = typer.Argument(..., help="Model name")):
    """Show model details."""
    model_config = get_model(model_name)
    if not model_config:
        console.print(f"[red]Error:[/] Unknown model: {model_name}")
        raise typer.Exit(1)

    console.print(f"\n[bold cyan]{model_name}[/] ({model_config['type']})")
    console.print(f"[dim]{model_config['description']}[/]")
    console.print(f"[bold]ID:[/] {model_config['id']}")

    console.print("\n[bold]Parameters:[/]")
    for name, info in model_config["params"].items():
        req = "[red]*[/]" if info.get("required") else " "
        default = f" [dim](default: {info['default']})[/]" if "default" in info else ""
        console.print(f"  {req} [cyan]{name}[/]: {info.get('help', '')}{default}")


@app.command()
def status(prediction_id: str = typer.Argument(..., help="Prediction ID")):
    """Check prediction status."""
    _check_token()

    try:
        result = get_prediction_status(prediction_id)
        color = {"starting": "yellow", "processing": "blue", "succeeded": "green", "failed": "red"}.get(result["status"], "white")

        console.print(f"\n[bold]Prediction:[/] {result['id']}")
        console.print(f"[bold]Status:[/] [{color}]{result['status']}[/]")

        if result["status"] == "succeeded" and result["output"]:
            out = result["output"][0] if isinstance(result["output"], list) else result["output"]
            console.print(f"[bold]Output:[/] {out}")

        if result["status"] == "failed" and result["error"]:
            console.print(f"[red]Error:[/] {result['error']}")

    except Exception as e:
        console.print(f"[red]Error:[/] {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
