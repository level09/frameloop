"""Interactive prompts for CLI."""

import questionary
from questionary import Style

from .models import get_model, get_models_by_type

# Custom style matching rich output
style = Style([
    ("qmark", "fg:cyan bold"),
    ("question", "bold"),
    ("answer", "fg:cyan"),
    ("pointer", "fg:cyan bold"),
    ("highlighted", "fg:cyan bold"),
    ("selected", "fg:green"),
])


def select_model(model_type: str, current: str | None = None) -> str:
    """Prompt user to select a model."""
    models = get_models_by_type(model_type)
    choices = [
        questionary.Choice(
            title=f"{name} - {m['description'][:50]}...",
            value=name,
        )
        for name, m in models.items()
    ]

    default = current if current in models else list(models.keys())[0]

    return questionary.select(
        "Select model:",
        choices=choices,
        default=default,
        style=style,
        use_jk_keys=False,
    ).ask()


def prompt_param(name: str, param_config: dict, current_value=None) -> any:
    """Prompt for a single parameter based on its config."""
    help_text = param_config.get("help", "")
    choices = param_config.get("choices")
    param_type = param_config.get("type", "str")
    default = current_value or param_config.get("default")

    if choices:
        # Selection menu for params with choices (convert to strings for questionary)
        str_choices = [str(c) for c in choices]
        str_default = str(default) if default is not None else str_choices[0]
        result = questionary.select(
            f"{name} ({help_text}):",
            choices=str_choices,
            default=str_default if str_default in str_choices else str_choices[0],
            style=style,
        ).ask()
        # Convert back to original type if needed
        if result and param_type == "int":
            return int(result)
        if result and choices and isinstance(choices[0], int):
            return int(result)
        return result

    elif param_type == "bool":
        # Yes/No for booleans
        return questionary.confirm(
            f"{name}? ({help_text})",
            default=default if default is not None else True,
            style=style,
        ).ask()

    elif param_type == "int":
        # Number input
        result = questionary.text(
            f"{name} ({help_text}):",
            default=str(default) if default else "",
            style=style,
        ).ask()
        return int(result) if result else default

    else:
        # Text input
        return questionary.text(
            f"{name} ({help_text}):",
            default=str(default) if default else "",
            style=style,
        ).ask()


def prompt_missing(model_name: str, provided: dict) -> dict:
    """Prompt for any missing or customizable params."""
    model_config = get_model(model_name)
    if not model_config:
        return provided

    result = provided.copy()
    params = model_config["params"]

    # Check required params
    for name, config in params.items():
        if config.get("required") and not result.get(name):
            value = prompt_param(name, config)
            if value:
                result[name] = value

    return result


def interactive_config(model_name: str, provided: dict) -> dict:
    """Full interactive configuration for a model."""
    model_config = get_model(model_name)
    if not model_config:
        return provided

    result = provided.copy()
    params = model_config["params"]

    # Only prompt for params with choices that weren't explicitly set
    customizable = [
        (name, config) for name, config in params.items()
        if config.get("choices") and name not in provided
    ]

    if customizable:
        customize = questionary.confirm(
            "Customize options?",
            default=False,
            style=style,
        ).ask()

        if customize:
            for name, config in customizable:
                value = prompt_param(name, config, result.get(name))
                if value is not None:
                    result[name] = value

    return result
