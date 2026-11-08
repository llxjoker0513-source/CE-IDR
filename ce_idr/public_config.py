import yaml


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Expected a YAML configuration mapping.")
    if config.get("template_only", False):
        raise ValueError(
            "This is a schema-only template. Detailed settings are withheld; "
            "supply a complete private configuration."
        )
    return config


def build_model(model_config):
    from models.fusion_model import CEIDR

    return CEIDR(
        in_ch=model_config["in_ch"],
        base_ch=model_config["base_ch"],
        num_experts=model_config["num_experts"],
        semantic_expand=model_config["semantic_expand"],
        use_region_router=model_config["use_region_router"],
        use_source_router=model_config["use_source_router"],
        use_spatial_gate=model_config["use_spatial_gate"],
        expert_mode=model_config["expert_mode"],
        use_experts=model_config["use_experts"],
        router_mode=model_config["router_mode"],
    )
