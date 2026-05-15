  python src/inference_difix.py \
    --pretrained_model_name_or_path nvidia/difix \
    --input_image assets/example_input.png \
    --prompt "remove degradation" \
    --output_dir outputs/difix \
    --timestep 199