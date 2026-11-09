def build_model(model_config):
    """Construct the network from an in-memory mapping of private model settings."""
    from .models.fusion_model import CEIDR

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
