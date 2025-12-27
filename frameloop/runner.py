"""Prediction execution and polling."""

import time

import replicate
from rich.console import Console
from rich.panel import Panel

from .utils import prepare_image_input, generate_output_filename, save_output, format_duration

console = Console()


def run_prediction(model_config: dict, inputs: dict, output_path: str | None = None, wait: bool = True) -> dict:
    """Run a prediction and return results."""
    model_id = model_config["id"]

    # Prepare file inputs
    if "image" in inputs:
        inputs["image"] = prepare_image_input(inputs["image"])
    if "start_image" in inputs:
        inputs["start_image"] = prepare_image_input(inputs["start_image"])
    if "last_frame" in inputs:
        inputs["last_frame"] = prepare_image_input(inputs["last_frame"])
    if "end_image" in inputs:
        inputs["end_image"] = prepare_image_input(inputs["end_image"])
    if "first_frame_image" in inputs and inputs["first_frame_image"]:
        inputs["first_frame_image"] = prepare_image_input(inputs["first_frame_image"])
    if "subject_reference" in inputs and inputs["subject_reference"]:
        inputs["subject_reference"] = prepare_image_input(inputs["subject_reference"])
    if "image_input" in inputs and inputs["image_input"]:
        inputs["image_input"] = [prepare_image_input(img) for img in inputs["image_input"]]
    if "input_images" in inputs and inputs["input_images"]:
        inputs["input_images"] = [prepare_image_input(img) for img in inputs["input_images"]]

    # Show info
    _show_start_info(model_id, inputs)

    # Create prediction
    start_time = time.time()
    prediction = replicate.predictions.create(model=model_id, input=inputs)

    if not wait:
        console.print(f"\n[dim]Prediction ID:[/] {prediction.id}")
        console.print("[dim]Use 'frameloop status <id>' to check progress[/]")
        return {"prediction_id": prediction.id, "status": prediction.status}

    # Poll until complete
    with console.status("[bold blue]Starting...", spinner="dots") as status:
        while prediction.status not in ("succeeded", "failed", "canceled"):
            time.sleep(2)
            prediction.reload()
            elapsed = format_duration(time.time() - start_time)
            status.update(f"[bold blue]{prediction.status.title()}... [dim]({elapsed})[/]")

    elapsed = time.time() - start_time

    if prediction.status == "failed":
        console.print(f"\n[red]Failed:[/] {prediction.error}")
        return {"prediction_id": prediction.id, "status": prediction.status, "elapsed_time": elapsed}

    # Get output URL
    output = prediction.output
    if isinstance(output, str):
        output_url = output
    elif isinstance(output, list) and output:
        output_url = output[0] if isinstance(output[0], str) else output[0].url
    else:
        output_url = getattr(output, "url", str(output))

    # Save output
    if not output_path:
        ext = inputs.get("output_format") or {"video": "mp4", "upscale": "png", "image": "jpg"}.get(model_config["type"], "png")
        input_ref = inputs.get("image") or (inputs.get("image_input") or [None])[0] or (inputs.get("input_images") or [None])[0] or "output"
        output_path = generate_output_filename(input_ref if isinstance(input_ref, str) else "output", model_config["type"], ext)

    save_output(output_url, output_path)

    console.print(f"\n[bold green]Complete![/]")
    console.print(f"[dim]Output:[/] {output_path}")
    console.print(f"[dim]Time:[/]   {format_duration(elapsed)}")

    return {"prediction_id": prediction.id, "status": prediction.status, "output_path": output_path, "elapsed_time": elapsed}


def _show_start_info(model_id: str, inputs: dict):
    """Display prediction start info."""
    lines = [f"[bold]Model:[/]  {model_id.split('/')[-1]}"]
    if "prompt" in inputs:
        prompt = inputs["prompt"][:60] + "..." if len(inputs["prompt"]) > 60 else inputs["prompt"]
        lines.append(f"[bold]Prompt:[/] {prompt}")
    console.print(Panel("\n".join(lines), title="[bold cyan]Frameloop[/]", border_style="cyan"))


def get_prediction_status(prediction_id: str) -> dict:
    """Get status of an existing prediction."""
    p = replicate.predictions.get(prediction_id)
    return {"id": p.id, "status": p.status, "model": p.model, "output": p.output, "error": p.error}
