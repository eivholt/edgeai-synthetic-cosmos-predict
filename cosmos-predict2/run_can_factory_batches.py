import json
import subprocess
import sys

import argparse

def main():
    parser = argparse.ArgumentParser(description="Run can factory batches.")
    parser.add_argument("--n", type=int, default=10, help="Number of iterations (default: 10)")
    args = parser.parse_args()

    NUM_ITERATIONS = args.n
    TEMPLATE = "batch_can_factory_LLM_enhanced_template.json"
    BATCH_JSON = "batch_can_factory_LLM_enhanced_template.json"

    # Load the template once to count prompts
    with open(TEMPLATE) as f:
        template_prompts = json.load(f)
    IMAGES_PER_BATCH = len(template_prompts)

    for i in range(NUM_ITERATIONS):
        # Reload fresh for each batch
        with open(TEMPLATE) as f:
            prompts = json.load(f)

        start_img_num = i * IMAGES_PER_BATCH + 1
        for j, entry in enumerate(prompts):
            img_num = start_img_num + j
            entry["output_image"] = f"outputs/can_factory_enhanced/{img_num:05d}.jpg"

        with open(BATCH_JSON, "w") as f:
            json.dump(prompts, f, indent=2)

        cmd = [
            "python", "-m", "examples.text2image",
            "--batch_input_json", BATCH_JSON,
            "--model_size", "2B",
            "--disable_guardrail",
            "--seed", str(i),
            #"--negative_prompt", "sharp focus, ultra‑high detail, DSLR, 4K, cinematic lighting, clean edges, HDR,no noise, no artefacts"
        ]
        subprocess.run(cmd)

if __name__ == "__main__":
    main()
