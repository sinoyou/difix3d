from pipeline_difix import DifixPipeline
from diffusers.utils import load_image

# 无参考图：必须用 nvidia/difix（单路 latent，batch=1）
pipe = DifixPipeline.from_pretrained("nvidia/difix", trust_remote_code=True)
# 有参考图：用 nvidia/difix_ref，并传入 ref_image=...
# pipe = DifixPipeline.from_pretrained("nvidia/difix_ref", trust_remote_code=True)

pipe.to("cuda")

input_image = load_image("assets/example_input.png")
prompt = "remove degradation"

output_image = pipe(
    prompt,
    image=input_image,
    num_inference_steps=1,
    timesteps=[199],
    guidance_scale=0.0,
).images[0]
output_image.save("example_output_no_ref.png")

# 参考图示例（需 difix_ref，且 ref_image 不能为 None，否则 UNet 内 new_forward 要求 batch 维为 2 的倍数）:
# pipe = DifixPipeline.from_pretrained("nvidia/difix_ref", trust_remote_code=True).to("cuda")
# ref_image = load_image("assets/example_ref.png")
# out = pipe(prompt, image=input_image, ref_image=ref_image, num_inference_steps=1, timesteps=[199], guidance_scale=0.0).images[0]
# out.save("example_output_ref.png")
