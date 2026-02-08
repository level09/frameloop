"""Model registry for Replicate AI models."""

MODELS = {
    "veo": {
        "id": "google/veo-3.1",
        "type": "video",
        "description": "Google Veo 3.1 - supports start+end frame interpolation",
        "params": {
            "prompt": {"required": True, "type": "str", "help": "Text prompt for video generation"},
            "image": {"required": False, "type": "file", "help": "Start frame image"},
            "last_frame": {"required": False, "type": "file", "help": "End frame for interpolation"},
            "duration": {"default": 8, "choices": [4, 6, 8], "help": "Video length in seconds"},
            "resolution": {"default": "1080p", "choices": ["720p", "1080p"], "help": "Output resolution"},
            "aspect_ratio": {"default": "16:9", "choices": ["16:9", "9:16"], "help": "Video aspect ratio"},
            "seed": {"default": None, "type": "int", "help": "Random seed for reproducibility"},
        },
    },
    "wan": {
        "id": "wavespeedai/wan-2.1-i2v-480p",
        "type": "video",
        "description": "Fast & cheap image-to-video (~$0.09/video)",
        "params": {
            "prompt": {"required": True, "type": "str", "help": "Text prompt for video generation"},
            "image": {"required": True, "type": "file", "help": "Input image (first frame)"},
            "aspect_ratio": {"default": "16:9", "choices": ["16:9", "9:16"], "help": "Video aspect ratio"},
            "fast_mode": {"default": "Fast", "choices": ["Off", "Balanced", "Fast"], "help": "Speed optimization"},
            "seed": {"default": None, "type": "int", "help": "Random seed for reproducibility"},
        },
        "cost": {"flat": 0.09},  # $ per video
    },
    "seedance": {
        "id": "bytedance/seedance-1-pro-fast",
        "type": "video",
        "description": "Fast image-to-video generation by ByteDance",
        "params": {
            "prompt": {"required": True, "type": "str", "help": "Text prompt for video generation"},
            "image": {"required": True, "type": "file", "help": "Input image"},
            "duration": {"default": 5, "range": [2, 12], "help": "Video length in seconds"},
            "resolution": {"default": "1080p", "choices": ["480p", "720p", "1080p"], "help": "Output resolution"},
            "aspect_ratio": {"default": "16:9", "choices": ["16:9", "4:3", "1:1", "3:4", "9:16"], "help": "Video aspect ratio"},
            "seed": {"default": None, "type": "int", "help": "Random seed for reproducibility"},
        },
        "cost": {"480p": 0.015, "720p": 0.025, "1080p": 0.06},  # $ per second
    },
    "esrgan": {
        "id": "nightmareai/real-esrgan",
        "type": "upscale",
        "description": "Image upscaling with optional face enhancement",
        "params": {
            "image": {"required": True, "type": "file", "help": "Input image"},
            "scale": {"default": 4, "choices": [2, 4], "help": "Upscale factor"},
            "face_enhance": {"default": False, "type": "bool", "help": "Enhance faces in image"},
        },
    },
    "kling": {
        "id": "kwaivgi/kling-v2.1",
        "type": "video",
        "description": "Kling v2.1 by Kuaishou - high quality image-to-video",
        "params": {
            "prompt": {"required": True, "type": "str", "help": "Text prompt for video generation"},
            "start_image": {"required": True, "type": "file", "help": "Input image (first frame)"},
            "end_image": {"required": False, "type": "file", "help": "End frame for interpolation (pro mode)"},
            "duration": {"default": 5, "choices": [5, 10], "help": "Video length in seconds"},
            "mode": {"default": "standard", "choices": ["standard", "pro"], "help": "standard (720p) or pro (1080p)"},
            "negative_prompt": {"default": "", "type": "str", "help": "Things to exclude from video"},
        },
        "cost": {"standard": 0.05, "pro": 0.09},  # $ per second
    },
    "kling-2.5-turbo": {
        "id": "kwaivgi/kling-v2.5-turbo-pro",
        "type": "video",
        "description": "Kling v2.5 Turbo Pro - fast high-quality video (~$0.07/sec)",
        "params": {
            "prompt": {"required": True, "type": "str", "help": "Text prompt for video generation"},
            "start_image": {"required": False, "type": "file", "help": "Input image (first frame)"},
            "end_image": {"required": False, "type": "file", "help": "End frame for interpolation"},
            "duration": {"default": 5, "choices": [5, 10], "help": "Video length in seconds"},
            "aspect_ratio": {"default": "16:9", "choices": ["16:9", "9:16", "1:1"], "help": "Video aspect ratio (ignored if start_image provided)"},
            "negative_prompt": {"default": "", "type": "str", "help": "Things to exclude from video"},
        },
        "cost": {"per_second": 0.07},
    },
    "kling-2.6": {
        "id": "kwaivgi/kling-v2.6",
        "type": "video",
        "description": "Kling v2.6 - cinematic video with native audio generation (~$0.07-0.14/sec)",
        "params": {
            "prompt": {"required": True, "type": "str", "help": "Text prompt for video generation"},
            "start_image": {"required": False, "type": "file", "help": "Input image (first frame)"},
            "duration": {"default": 5, "choices": [5, 10], "help": "Video length in seconds"},
            "aspect_ratio": {"default": "16:9", "choices": ["16:9", "9:16", "1:1"], "help": "Video aspect ratio"},
            "generate_audio": {"default": True, "type": "bool", "help": "Generate synchronized audio (dialogue, effects, ambient)"},
            "negative_prompt": {"default": "", "type": "str", "help": "Things to exclude from video"},
        },
        "cost": {"without_audio": 0.07, "with_audio": 0.14},  # $ per second
    },
    "nano-banana-pro": {
        "id": "google/nano-banana-pro",
        "type": "image",
        "description": "Google's Nano Banana Pro - image generation & transformation with reference images",
        "params": {
            "prompt": {"required": True, "type": "str", "help": "Text description of the image to generate"},
            "image_input": {"required": False, "type": "file_list", "help": "Input images for transformation (up to 14)"},
            "aspect_ratio": {"default": "match_input_image", "choices": ["match_input_image", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"], "help": "Output aspect ratio"},
            "resolution": {"default": "2K", "choices": ["1K", "2K", "4K"], "help": "Output resolution"},
            "output_format": {"default": "jpg", "choices": ["jpg", "png"], "help": "Output image format"},
        },
        "cost": {"1K": 0.15, "2K": 0.15, "4K": 0.30},  # $ per image
    },
    "seedream": {
        "id": "bytedance/seedream-4",
        "type": "image",
        "description": "ByteDance Seedream 4.0 - fast, high-quality image generation (~$0.03/image)",
        "params": {
            "prompt": {"required": True, "type": "str", "help": "Text description of the image to generate"},
            "size": {"default": "2K", "choices": ["1K", "2K", "4K", "custom"], "help": "Output resolution"},
            "width": {"default": None, "range": [1024, 4096], "help": "Custom width (requires size=custom)"},
            "height": {"default": None, "range": [1024, 4096], "help": "Custom height (requires size=custom)"},
            "aspect_ratio": {"default": "16:9", "choices": ["1:1", "4:3", "3:4", "16:9", "9:16", "3:2", "2:3", "21:9"], "help": "Output aspect ratio"},
            "enhance_prompt": {"default": True, "type": "bool", "help": "Enable AI prompt enhancement"},
            "max_images": {"default": 1, "range": [1, 15], "help": "Number of images to generate"},
        },
        "cost": {"flat": 0.03},  # $ per image
    },
    "gpt-image": {
        "id": "openai/gpt-image-1.5",
        "type": "image",
        "description": "OpenAI GPT Image 1.5 - 4x faster, great text rendering & editing",
        "params": {
            "prompt": {"required": True, "type": "str", "help": "Text description of the image to generate"},
            "input_images": {"required": False, "type": "file_list", "help": "Input images for editing/compositing"},
            "aspect_ratio": {"default": "1:1", "choices": ["1:1", "3:2", "2:3"], "help": "Output aspect ratio"},
            "quality": {"default": "auto", "choices": ["low", "medium", "high", "auto"], "help": "Generation quality"},
            "input_fidelity": {"default": "low", "choices": ["low", "high"], "help": "Adherence to input image style"},
            "number_of_images": {"default": 1, "range": [1, 10], "help": "Number of images to generate"},
            "output_format": {"default": "webp", "choices": ["png", "jpeg", "webp"], "help": "Output image format"},
            "output_compression": {"default": 90, "range": [0, 100], "help": "Compression level (0-100)"},
            "background": {"default": "auto", "choices": ["auto", "transparent", "opaque"], "help": "Background type"},
        },
        "cost": {"low": 0.01, "medium": 0.02, "high": 0.04, "auto": 0.02},  # $ per image
    },
    "minimax-live": {
        "id": "minimax/video-01-live",
        "type": "video",
        "description": "Minimax Live2D - image-to-video with smooth animation",
        "params": {
            "prompt": {"required": True, "type": "str", "help": "Text prompt for video generation"},
            "first_frame_image": {"required": True, "type": "file", "help": "Input image (first frame)"},
            "prompt_optimizer": {"default": True, "type": "bool", "help": "Use AI prompt optimization"},
        },
    },
    "minimax-image": {
        "id": "minimax/image-01",
        "type": "image",
        "description": "Minimax Image-01 - high-fidelity image generation (~$0.01/image)",
        "params": {
            "prompt": {"required": True, "type": "str", "help": "Text description of the image to generate"},
            "aspect_ratio": {"default": "1:1", "choices": ["1:1", "16:9", "4:3", "3:2", "2:3", "3:4", "9:16", "21:9"], "help": "Output aspect ratio"},
            "number_of_images": {"default": 1, "range": [1, 9], "help": "Number of images to generate"},
            "prompt_optimizer": {"default": True, "type": "bool", "help": "Use AI prompt optimization"},
            "subject_reference": {"required": False, "type": "file", "help": "Face reference image for subject"},
        },
        "cost": {"flat": 0.01},  # $ per image
    },
    "hailuo": {
        "id": "minimax/hailuo-2.3",
        "type": "video",
        "description": "Minimax Hailuo 2.3 - cinematic video generation",
        "params": {
            "prompt": {"required": True, "type": "str", "help": "Text prompt for video generation"},
            "first_frame_image": {"required": False, "type": "file", "help": "Input image (optional first frame)"},
            "duration": {"default": 6, "choices": [6, 10], "help": "Video length (10s only at 768p)"},
            "resolution": {"default": "768p", "choices": ["768p", "1080p"], "help": "Output resolution (1080p only 6s)"},
            "prompt_optimizer": {"default": True, "type": "bool", "help": "Use AI prompt optimization"},
        },
        "cost": {"768p_6s": 0.28, "768p_10s": 0.56, "1080p_6s": 0.49},  # $ per video
    },
    "recraft-svg": {
        "id": "recraft-ai/recraft-v3-svg",
        "type": "image",
        "description": "Recraft V3 SVG - vector graphics generation from text prompts",
        "params": {
            "prompt": {"required": True, "type": "str", "help": "Text prompt for SVG generation"},
            "style": {"default": "any", "choices": ["any", "engraving", "line_art", "line_circuit", "linocut"], "help": "Visual style of the SVG"},
        },
    },
    "recraft-20b": {
        "id": "recraft-ai/recraft-20b-svg",
        "type": "image",
        "description": "Recraft 20B SVG - advanced vector graphics with icons & illustrations (~$0.044/image)",
        "params": {
            "prompt": {"required": True, "type": "str", "help": "Text prompt for SVG generation"},
            "style": {"default": "vector_illustration", "choices": ["vector_illustration", "vector_illustration/cartoon", "vector_illustration/doodle_line_art", "vector_illustration/engraving", "vector_illustration/flat_2", "vector_illustration/kawaii", "vector_illustration/line_art", "vector_illustration/line_circuit", "vector_illustration/linocut", "vector_illustration/seamless", "icon", "icon/broken_line", "icon/colored_outline", "icon/colored_shapes", "icon/colored_shapes_gradient", "icon/doodle_fill", "icon/doodle_offset_fill", "icon/offset_fill", "icon/outline", "icon/outline_gradient", "icon/uneven_fill"], "help": "Visual style (vector_illustration or icon variants)"},
        },
        "cost": {"flat": 0.044},
    },
    "recraft-bg": {
        "id": "recraft-ai/recraft-remove-background",
        "type": "transform",
        "description": "Recraft Remove Background - precise background removal (~$0.01/image)",
        "params": {
            "image": {"required": True, "type": "file", "help": "Image to remove background from (PNG, JPG, WEBP)"},
        },
        "cost": {"flat": 0.01},
    },
    "sora": {
        "id": "openai/sora-2",
        "type": "video",
        "description": "OpenAI Sora 2 - cinematic video with synchronized audio (~$0.10/sec)",
        "params": {
            "prompt": {"required": True, "type": "str", "help": "Text description of the video to generate"},
            "input_reference": {"required": False, "type": "file", "help": "Input image (first frame, must match aspect ratio)"},
            "seconds": {"default": 4, "choices": [4, 8, 12], "help": "Video length in seconds"},
            "aspect_ratio": {"default": "portrait", "choices": ["portrait", "landscape"], "help": "Portrait (720x1280) or landscape (1280x720)"},
            "openai_api_key": {"required": False, "type": "str", "env": "OPENAI_API_KEY", "help": "Optional: use own OpenAI key for direct billing"},
        },
        "cost": {"per_second": 0.10},
    },
    "hunyuan3d": {
        "id": "tencent/hunyuan3d-2:b1b9449a1277e10402781c5d41eb30c0a0683504fb23fab591ca9dfc2aabe1cb",
        "type": "3d",
        "description": "Tencent Hunyuan3D-2 - image to 3D mesh generation (~$0.17/run)",
        "params": {
            "image": {"required": True, "type": "file", "help": "Input image for generating 3D shape"},
            "steps": {"default": 50, "range": [20, 50], "help": "Number of inference steps"},
            "guidance_scale": {"default": 5.5, "range": [1, 20], "type": "float", "help": "Guidance scale for generation"},
            "octree_resolution": {"default": 256, "choices": [256, 384, 512], "help": "Octree resolution for mesh generation"},
            "remove_background": {"default": True, "type": "bool", "help": "Remove background from input image"},
            "seed": {"default": None, "type": "int", "help": "Random seed for reproducibility"},
        },
        "cost": {"flat": 0.17},
    },
}


def get_model(name: str) -> dict | None:
    """Get model config by name."""
    return MODELS.get(name.lower())


def get_models_by_type(model_type: str) -> dict:
    """Get all models of a specific type."""
    return {k: v for k, v in MODELS.items() if v["type"] == model_type}


def list_models() -> list[tuple[str, str, str]]:
    """Return list of (name, type, description) for all models."""
    return [(name, m["type"], m["description"]) for name, m in MODELS.items()]
