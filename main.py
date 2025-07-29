import os, replicate, gradio as gr

# Make sure to set your REPLICATE_API_TOKEN in the deployment environment.
# Do NOT hardcode it here for security reasons.

def generate_images(prompt):
    try:
        output = replicate.run(
            "stability-ai/sdxl",
            input={
                "prompt": prompt,
                "num_outputs": 3,
                "guidance_scale": 7.5,
                "num_inference_steps": 40
            }
        )
        return output + [None]*(3-len(output))  # Ensure 3 outputs
    except Exception as e:
        return f"Error: {e}", None, None

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 Nova AI")
    gr.Markdown("Generate **3 stunning AI images** from a text prompt using SDXL.")
    with gr.Row():
        prompt = gr.Textbox(label="Prompt", placeholder="e.g. A galaxy turtle surfing space waves")
        btn = gr.Button("Generate")
    with gr.Row():
        img1 = gr.Image(label="Image 1")
        img2 = gr.Image(label="Image 2")
        img3 = gr.Image(label="Image 3")
    btn.click(fn=generate_images, inputs=prompt, outputs=[img1, img2, img3])

demo.launch()
