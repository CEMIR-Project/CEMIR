import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from tqdm import tqdm
from contextlib import redirect_stdout
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


MODEL_NAME = "Qwen/Qwen3-8B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.bfloat16,
).eval()

def extract_actions(task_description):
    return [action.strip() for action in re.findall(r'Human (.*?)\.', task_description) if action]

SYSTEM_PROMPT = (
    "You are creating visual scene descriptions for image generation.\n"
    "Follow these guidelines:\n"
    "1. Describe ONLY what's explicitly stated in the environment and actions\n"
    "2. Include positional relationships between objects (on, near, beside, etc.)\n"
    "3. Mention relevant nearby objects from the environment\n"
    "4. Keep human actions literal - don't add interpretive details like 'reaching for'\n"
    "5. Use 2-3 sentences to capture spatial relationships\n"
    "6. Modern indoor setting is implied - don't describe it separately\n"
    "7. Always use present continuous tense when describing human action\n"
    "8. In actions in which the human is moving from one place to another describe the human as walking from A to B and focus on the objects present in the path.\n"
    "9. Don't spend too long thinking (no more than 10 sentences of thinking)"
)

BATCH_SIZE = 6

def generate_scene_descriptions_batch(environment, actions_batch):
    texts = []
    for idx, action in enumerate(actions_batch):
        trajectory = "\n".join(f"- {a}" for a in actions_batch[:idx+1])
        current_action = action

        user_input = (
            f"### Environment Context:\n{environment}\n\n"
            f"### Action Sequence:\n{trajectory}\n\n"
            f"### Current Action to Visualize:\nHuman {current_action}\n\n"
            f"### Required Output Format:\n"
            f"- Start with the human performing the literal action including the primary object\n"
            f"- Then describe 2-3 secondary objects with their positions (using words like: on, beside, near, under)\n"
            f"- Optionally add a final phrase about key object states (open/closed/containing/...)\n\n"
            f"### Critical Guidelines:\n"
            f"- Don't spend too long thinking (no more than 10 sentences of thinking)\n"
            f"- Select only 3-4 total objects (most relevant to action)\n"
            f"- Always mention object positions explicitly\n"
            f"- Use literal action descriptions only (no interpretations)\n"
            f"- Always use present continuous tense when describing human action\n"
            f"- In actions in which the human is moving from one place to another describe the human as walking from A to B and focus on the objects present in the path.\n"
            f"- Avoid phrases like 'reaching for' or 'stepping over'\n"
            f"- Modern indoor setting is implied - don't describe it separately\n"
            f"- After thinking, provide the FINAL VISUAL DESCRIPTION as natural sentences:\n\n"
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        texts.append(text)

    inputs = tokenizer(texts, return_tensors="pt", padding=True).to(DEVICE)
    outputs = model.generate(
        **inputs,
        max_new_tokens=10000,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    return [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

def generate_prompt_json(episode_data, output_path_final):
    actions = extract_actions(episode_data["task_description"])
    environment = episode_data["demo_observation"]
    results = []

    print(f"Generating scene descriptions for {len(actions)} actions...", flush=True)
    for i in range(0, len(actions), BATCH_SIZE):
        batch = actions[i:i + BATCH_SIZE]
        responses = generate_scene_descriptions_batch(environment, batch)
        for j, full_response in enumerate(responses):
            idx = i + j
            if "</think>" in full_response:
                parts = full_response.split("</think>", 1)
                scene_description = parts[1].strip()
                scene_description = re.sub(r'<[^>]+>', '', scene_description)
                scene_description = re.sub(r'^\d+\.\s*', '', scene_description, flags=re.MULTILINE)
            else:
                scene_description = re.sub(r'<[^>]+>', '', "failed to complete thinking").strip()
                print(f"failed to complete thinking for action {idx+1} of episode {output_path_final}")

            results.append({
                "action_id": idx+1,
                "action_description": actions[idx],
                "scene_description": scene_description,
            })

    with open(output_path_final, 'w') as f:
        json.dump({"episode_scenes": results}, f, indent=2)

    print(f"\nSaved {len(results)} scene descriptions to {output_path_final}")
    return results

def new_extract_from_file(data):
    task_description = data.get("task_description", "")
    observations = data.get("demo_observations_fully", [])

    last_observation = observations[0] if observations else ""
    lines = last_observation.split('\n')
    final_last_observation = '\n'.join(lines[3:-2])

    lines = task_description.split('\n')
    final_task_description = '\n'.join(lines[2:-2])

    return [final_task_description, final_last_observation]


input_folder = "data/raw_episodes"
output_folder = "data/enhanced_text"
Path(output_folder).mkdir(parents=True, exist_ok=True)

with open("log_enhance.txt", "w") as log_file, redirect_stdout(log_file):
    input_files = list(Path(input_folder).glob('*.json'))
    print(f"Generating enhanced text for {len(input_files)} files")

    for json_file in tqdm(input_files, desc="Processing episodes"):
        with open(json_file, 'r') as f:
            data = json.load(f)

        task_description, observations = new_extract_from_file(data)

        episode = {
            "task_description": task_description,
            "demo_observation": observations
        }

        output_path_final = f"{output_folder}/{json_file.stem}.json"
        generate_prompt_json(episode, output_path_final)

    print(f"Processed {len(input_files)} files")
