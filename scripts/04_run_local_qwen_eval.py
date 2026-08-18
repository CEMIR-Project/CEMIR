import json
import os
import sys
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
import random
import yaml
from tqdm import tqdm
import time
import logging
logging.getLogger().setLevel(logging.ERROR)
import subprocess
import csv
import threading

def track_gpu_memory(interval=1.0):
    memory_samples = []

    def poll():
        while tracking_flag[0]:
            try:
                result = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"]
                )
                mem_usage = int(result.decode().split("\n")[0].strip())
                memory_samples.append(mem_usage)
            except Exception as e:
                print(f"[Warning] Could not read GPU usage: {e}")
            time.sleep(interval)

    tracking_flag = [True]
    thread = threading.Thread(target=poll)
    thread.start()
    return memory_samples, tracking_flag, thread

def log_experiment_to_csv(csv_path, config_idx, run_number, duration, avg_mem, max_mem):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["ConfigIndex", "RunNumber", "TimeSeconds", "AvgGPUMemMB", "MaxGPUMemMB"])
        writer.writerow([config_idx + 1, run_number + 1, f"{duration:.2f}", avg_mem, max_mem])


CONFIG_PATH = sys.argv[1] if len(sys.argv) > 1 else "config_local_qwen.yml"
INCLUDE_TITLE = "include_title"
INCLUDE_IMAGES = "include_images"
INCLUDE_ENHANCED = "include_enhanced"
BASE_FOLDER_PATH = "base_folder_path"
OUTPUT_FOLDER_PATH = "output_folder_path"
USE_IMAGE_EVERY_N_ACTION = "use_image_every_n_action"
USE_FISER_PROMPTING = "use_fiser_prompting"
SAMPLE_JSON_PATH = "sample_json_path"


SYSTEM_PROMPT = """You are a robot. Every time you will receive a description of the world and an instruction given by the human in the following templates:

[World Description]:

[Human Trajectory]:

[Human Instruction]:

For the [Human Trajectory] section you will be given the actions of the human towards achieving a goal. You will be provided the title of the action. a detailed description of the action which might sometimes have small inaccuracies. and a photo of the whole scene when the human is doing the action.
You are expected to generate a sequence of actions that are consistent with the human’s instruction and the world description. Please remember that each object has a number ID.

!!!!!THE LAST PART OF THE RESPONSE should be a sequence of actions starting with the string ”Actions:” (!!Only actions! No numbering for actions, no description, no extra words)!!!!!
!!!!! YOU CAN ONLY SELECT FROM THE ACTION FORMATS GIVEN TO YOU !!!!!
!!!!! AVOID AMBIGOUS WORDS LIKE THE THING or ... IN YOUR RESPONSES. USE PRECISE OBJECT NAMES AND IDs !!!!!

The available action templates contain:
move to XXX
pick up XXX
pick up XXX from XXX
put XXX into XXX
put XXX onto XXX
take XXX from XXX
give XXX to human
open XXX

A typical answer format is as follows, where the ”open” step is optional.

Actions:
move to XXX
open XXX
pick up XXX
move to human
give XXX to human

NOTE!!: Remember to open something like refrigerator if it is closed before pick up something. There are some openable objects: vessel, bag, box, package, cabinet, microwave, oven, dishwasher, refrigerator
Take an action like open refrigerator, open cabinet, open box, open bag, open vessel, open package, open microwave, open oven, open dishwasher
========================================================================================
To let you answer better, we give you the description of the goal space for the world as follows. The human has one of the following goals:
Goal 0: boxing books up for storage
Goal description: Put all the paper products of a certain kind into a box.
Goal 1: bring in wood
Goal description: Put all the building materials of a certain kind on top of the floor.
Goal 2: clearing the table after dinner
Goal description: Put all of one certain kind of cutlery inside some bucket, put all of another kind of cutlery into another bucket, and put all the flavorer of a certain kind into a bucket.
Goal 3: collect misplaced items
Goal description: Put all of a certain footwear, a certain decoration, and a certain paper product on top of the table.
Goal 4: collect aluminum cans
Goal description: Put all drinks into the ashcan.
Goal 5: installing alarms
Goal description: Put an electrical device on the table, on the countertop, and on the sofa.
Goal 6: laying tile floors
Goal description: Put all of a certain building material on top of the floor.
Goal 7: loading the dishwasher
Goal description: Put all the tablewares of two certain kinds and all of a certain vessel into the sink.
Goal 8: moving boxes to storage
Goal description: Put all the boxes on top of the floor.
Goal 9: oraganizing boxes in garage
Goal description: Put all of a certain plaything into a box, put all of a certain cutlery into another box, put all of a certain cleasing thing into a third box, and put all the boxes on top of the floor.
Goal 10: organize file cabinet
Goal description: Put all the writing tools of a certain kind on top of the table, and put all the paper products of a certain kind into the cabinet.
Goal 11: pick up trash
Goal description: Put all of a certain paper product and all of a certain drink into the ashcan.
Goal 12: put away Christmas decorations
Goal description: Put all the decorations of three certain kinds into the cabinet.
Goal 13: put away Halloween decorations
Goal description: Put all the vegetable of a certain kind and all the illumination tools of a certain kind into the cabinet, and put all of a certain vessels on top of the table.
Goal 14: put away toys
Goal description: Put all the playthings of a certain kind into a closed box.
Goal 15: put dishes away after cleaning
Goal description: Put all the tablewares of a certain kind into the cabinet.
Goal 16: put leftovers away
Goal description: Put all of a certain prepared food and all of a certain flavorer into the refrigerator.
Goal 17: put up Christmas decorations inside
Goal description: Put all of a certain illumination tool and all the decorations of two certain kinds on top of the table, and put all the decorations of another certain kind on top of the sofa.
Goal 18: re-shelve library books
Goal description: Put all of a certain paper product on top of the shelf.
Goal 19: serve hors d’oeuvres
Goal description: Put all of a certain baked food, a certain vegetable, a certain prepared food, and the trays on top of the table.
Goal 20: sort books
Goal description: Put all the paper products of two certain kinds on top of the shelf.
Goal 21: store food
Goal description: Put all of a certain prepared food, a certain snacks, and two certain kinds of flavorers into the cabinet.
Goal 22: store the groceries
Goal description: Put all of a certain fruit, a certain protein, and two certain kinds of vegetables into the refrigerator.
Goal 23: thaw frozen food
Goal description: Put all of a certain vegetable, a certain fruit, and a certain protein into the sink.
Goal 24: throw away leftovers
Goal description: Put all the snacks of a certain kind into the ashcan.
=========================================================================
"""


FISER_SYSTEM_PROMPT = """You are a robot. Every time you will receive a description of the world and an instruction given by the human in the following templates:

[World Description]:

[Human Trajectory]:

[Human Instruction]:

For the [Human Trajectory] section you will be given the actions of the human towards achieving a goal. You will be provided the title of the action. a detailed description of the action which might sometimes have small inaccuracies. and a photo of the whole scene when the human is doing the action.
You are expected to generate a sequence of actions that are consistent with the human’s instruction and the world description. Please remember that each object has a number ID.

Please first think step-by-step! 1. What is human doing? (Please find the most possible goal name and goal id if you are given the goal space, otherwise you should infer by yourself, but you STILL need output!)
2. Which object do you think the human is asking for?
3. What are your actions?
Answer the above questions one by one NO MATTER YOU KNOW THE GOAL SPACE OR NOT!

!!!!!THE LAST PART OF THE RESPONSE should be a sequence of actions starting with the string ”Actions:” (!!Only actions! No numbering for actions, no description, no extra words)!!!!!
!!!!! YOU CAN ONLY SELECT FROM THE ACTION FORMATS GIVEN TO YOU !!!!!
!!!!! AVOID AMBIGOUS WORDS LIKE THE THING or ... IN YOUR RESPONSES. USE PRECISE OBJECT NAMES AND IDs !!!!!

The available action templates contain:
move to XXX
pick up XXX
pick up XXX from XXX
put XXX into XXX
put XXX onto XXX
take XXX from XXX
give XXX to human
open XXX

A typical answer format is as follows, where the ”open” step is optional.

Actions:
move to XXX
open XXX
pick up XXX
move to human
give XXX to human

NOTE!!: Remember to open something like refrigerator if it is closed before pick up something. There are some openable objects: vessel, bag, box, package, cabinet, microwave, oven, dishwasher, refrigerator
Take an action like open refrigerator, open cabinet, open box, open bag, open vessel, open package, open microwave, open oven, open dishwasher
========================================================================================
To let you answer better, we give you the description of the goal space for the world as follows. The human has one of the following goals:
Goal 0: boxing books up for storage
Goal description: Put all the paper products of a certain kind into a box.
Goal 1: bring in wood
Goal description: Put all the building materials of a certain kind on top of the floor.
Goal 2: clearing the table after dinner
Goal description: Put all of one certain kind of cutlery inside some bucket, put all of another kind of cutlery into another bucket, and put all the flavorer of a certain kind into a bucket.
Goal 3: collect misplaced items
Goal description: Put all of a certain footwear, a certain decoration, and a certain paper product on top of the table.
Goal 4: collect aluminum cans
Goal description: Put all drinks into the ashcan.
Goal 5: installing alarms
Goal description: Put an electrical device on the table, on the countertop, and on the sofa.
Goal 6: laying tile floors
Goal description: Put all of a certain building material on top of the floor.
Goal 7: loading the dishwasher
Goal description: Put all the tablewares of two certain kinds and all of a certain vessel into the sink.
Goal 8: moving boxes to storage
Goal description: Put all the boxes on top of the floor.
Goal 9: oraganizing boxes in garage
Goal description: Put all of a certain plaything into a box, put all of a certain cutlery into another box, put all of a certain cleasing thing into a third box, and put all the boxes on top of the floor.
Goal 10: organize file cabinet
Goal description: Put all the writing tools of a certain kind on top of the table, and put all the paper products of a certain kind into the cabinet.
Goal 11: pick up trash
Goal description: Put all of a certain paper product and all of a certain drink into the ashcan.
Goal 12: put away Christmas decorations
Goal description: Put all the decorations of three certain kinds into the cabinet.
Goal 13: put away Halloween decorations
Goal description: Put all the vegetable of a certain kind and all the illumination tools of a certain kind into the cabinet, and put all of a certain vessels on top of the table.
Goal 14: put away toys
Goal description: Put all the playthings of a certain kind into a closed box.
Goal 15: put dishes away after cleaning
Goal description: Put all the tablewares of a certain kind into the cabinet.
Goal 16: put leftovers away
Goal description: Put all of a certain prepared food and all of a certain flavorer into the refrigerator.
Goal 17: put up Christmas decorations inside
Goal description: Put all of a certain illumination tool and all the decorations of two certain kinds on top of the table, and put all the decorations of another certain kind on top of the sofa.
Goal 18: re-shelve library books
Goal description: Put all of a certain paper product on top of the shelf.
Goal 19: serve hors d’oeuvres
Goal description: Put all of a certain baked food, a certain vegetable, a certain prepared food, and the trays on top of the table.
Goal 20: sort books
Goal description: Put all the paper products of two certain kinds on top of the shelf.
Goal 21: store food
Goal description: Put all of a certain prepared food, a certain snacks, and two certain kinds of flavorers into the cabinet.
Goal 22: store the groceries
Goal description: Put all of a certain fruit, a certain protein, and two certain kinds of vegetables into the refrigerator.
Goal 23: thaw frozen food
Goal description: Put all of a certain vegetable, a certain fruit, and a certain protein into the sink.
Goal 24: throw away leftovers
Goal description: Put all the snacks of a certain kind into the ashcan.
=========================================================================
"""


def extract_world_description(raw_json_path):
    with open(raw_json_path, 'r') as f:
        raw_data = json.load(f)
    return raw_data.get("demo_observations_fully", [""])[0]

def extract_instruction_from_task_description(raw_json_path):
    with open(raw_json_path, 'r') as f:
        raw_data = json.load(f)
    text = raw_data.get("task_description", "")
    lines = text.strip().split('\n')
    return lines[-1].strip()

def get_enhanced_steps(enhanced_json_path):
    with open(enhanced_json_path, 'r') as f:
        enhanced_data = json.load(f)
    return enhanced_data.get("episode_scenes", [])

def get_step_messages(step, pics_folder, config, should_include_image):
    messages = []
    action_id = step.get("action_id")
    action_desc = step.get("action_description", "").strip()
    scene_desc = step.get("scene_description", "").strip()
    image_path = os.path.join(pics_folder, f"{action_id}.png")
    action_text = ""
    if config.get(INCLUDE_TITLE, False):
        action_text += f"Scene Description: {scene_desc}\n"

    if config.get(INCLUDE_ENHANCED, False):
        action_text += f"Action Title: {action_desc}\n"

    messages.append({
        "role": "user",
        "content": [{"type": "text", "text": action_text}]
    })

    if config.get(INCLUDE_IMAGES, False) and should_include_image:
        if os.path.exists(image_path):
            messages.append({
                "role": "user",
                "content": [{"type": "image", "image": image_path}]
            })
        else:
            print(f"[Warning] Image not found: {image_path}")

    return messages


def create_the_final_user_prompt(episode_folder_path, config):
    raw_file_path = os.path.join(episode_folder_path, "raw_file.json")
    enhanced_file_path = os.path.join(episode_folder_path, "enhanced_text.json")
    pics_folder = os.path.join(episode_folder_path, "pics")

    world_description = extract_world_description(raw_file_path)
    instruction = extract_instruction_from_task_description(raw_file_path)
    enhanced_steps = get_enhanced_steps(enhanced_file_path)

    messages = []

    intro_text = (
        "======================== Now please begin to generate the output for given scenarios ========================\n"
        f"[World Description]: {world_description}\n"
        "[Human Trajectory]: You will now be given a sequence of actions that the human performed to reach a goal. "
        "Each step includes a title of the action, a scene description that might contain small inaccuracies, and an image of the scene when the action was done."
    )
    messages.append({
        "role": "user",
        "content": [{"type": "text", "text": intro_text}]
    })

    for i, step in enumerate(enhanced_steps):
        should_include_image = (i % config.get(USE_IMAGE_EVERY_N_ACTION)) == 0
        step_messages = get_step_messages(step, pics_folder, config, should_include_image)
        for message in step_messages:
            messages.append(message)

    instruction_text = (
        f"[Human Instruction]: {instruction}\n"
        "You are expected to generate a sequence of actions that are consistent with the human’s instruction and the world description.\n"
        "Please remember that each object has a number ID.\n"
        "Remember in the final output, the type of object should be specific!! you CAN NOT say something like e.g., note !!!!!"
        "!!!!!THE LAST PART OF THE RESPONSE should be a sequence of actions starting with the string \"Actions:\" \n"
        "(!!Only actions! No numbering for actions, no description, no extra words)!!!!!\n"
        "======================== Let’s start! ========================"
    )
    instruction_text_fiser = (
        f"[Human Instruction]: {instruction}\n"
        "You are expected to generate a sequence of actions that are consistent with the human’s instruction and the world description.\n"
        "Please remember that each object has a number ID.\n"
        "Remember in the final output, the type of object should be specific!! you CAN NOT say something like e.g., note !!!!!"
        "Please first think step-by-step!"
        "1. What is human doing? (Please find the most possible goal name and goal id if you are given the goal space, otherwise you should infer by yourself, but you STILL need output!)"
        "2. Which object do you think the human is asking for?"
        "3. What are your actions?"
        "Answer the above questions one by one NO MATTER YOU KNOW THE GOAL SPACE OR NOT!"
        "!!!!!THE LAST PART OF THE RESPONSE should be a sequence of actions starting with the string \"Actions:\" \n"
        "(!!Only actions! No numbering for actions, no description, no extra words)!!!!!\n"
        "======================== Let’s start! ========================"
    )

    final_instruction_text = instruction_text_fiser if config.get(USE_FISER_PROMPTING, False) else instruction_text
    messages.append({
        "role": "user",
        "content": [{"type": "text", "text": final_instruction_text}]
    })

    return messages

def get_ground_truth_simple(sample_path):
    raw_file_path = os.path.join(sample_path, "raw_file.json")
    with open(raw_file_path, 'r') as f:
        raw_data = json.load(f)
    the_actions_list = raw_data.get("demo_actions", [""])
    return "Actions:\n" + '\n'.join(the_actions_list)

def get_ground_truth(sample_path, config, sample_json_file):
    if config.get(USE_FISER_PROMPTING, False):
        final_episode_id = os.path.basename(sample_path).lower()
        return sample_json_file.get(final_episode_id)

    raw_file_path = os.path.join(sample_path, "raw_file.json")
    with open(raw_file_path, 'r') as f:
        raw_data = json.load(f)
    the_actions_list = raw_data.get("demo_actions", [""])
    return "Actions:\n" + '\n'.join(the_actions_list)


def create_prompt_messages(current_path, sample_path, sample_path_2, config, sample_json_file):
    final_messages = []

    system_prompt = FISER_SYSTEM_PROMPT if config.get(USE_FISER_PROMPTING, False) else SYSTEM_PROMPT

    final_messages.append({
        "role": "system",
        "content": [{"type": "text", "text": system_prompt}]
    })

    the_sample_query_messages = create_the_final_user_prompt(sample_path, config)
    for message in the_sample_query_messages:
        final_messages.append(message)
    final_messages.append({
        "role": "assistant",
        "content": [{"type": "text", "text": get_ground_truth(sample_path, config, sample_json_file)}]
    })

    the_sample_query_messages_2 = create_the_final_user_prompt(sample_path_2, config)
    for message in the_sample_query_messages_2:
        final_messages.append(message)
    final_messages.append({
        "role": "assistant",
        "content": [{"type": "text", "text": get_ground_truth(sample_path_2, config, sample_json_file)}]
    })

    the_current_query_messages = create_the_final_user_prompt(current_path, config)
    for message in the_current_query_messages:
        final_messages.append(message)

    return final_messages


def extract_assistant_response(text_block):
    if isinstance(text_block, list):
        text_block = text_block[0]
    parts = text_block.split("assistant\n")
    if len(parts) > 1:
        return parts[-1].strip()
    else:
        return ""


def list_folders(path):
    return [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]


def select_random_example(folder_list, all_samples_list, current_list, config):
    fiser_state = config.get(USE_FISER_PROMPTING, False)
    if fiser_state:
        the_sample = None
        while the_sample is None or the_sample in current_list:
            the_sample = random.choice(all_samples_list)
    else:
        the_sample = None
        while the_sample is None or the_sample in current_list:
            the_sample = random.choice(folder_list)

    return the_sample


model_name = "Qwen/Qwen2.5-Omni-3B"
device_map = "auto"

model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device_map,
    attn_implementation="flash_attention_2"
)
model.disable_talker()

processor = Qwen2_5OmniProcessor.from_pretrained(model_name)


def perform_one_experiment(base_folder_path, sample_json_file, output_path, config):
    failed_episodes_path = os.path.join(output_path, "failed_episodes.txt")
    folder_list = list_folders(base_folder_path)
    all_samples_list = list(sample_json_file.keys())

    os.makedirs(output_path, exist_ok=True)

    for episode in tqdm(folder_list, desc=f"Processing episodes in {base_folder_path}"):
        output_file_path = os.path.join(output_path, f"{episode}.json")
        if os.path.exists(output_file_path):
            print(f"[SKIP] Episode {episode} already processed.")
            continue

        done = False
        attempts = 0
        max_attempts = 10
        while not done and attempts < max_attempts:
            attempts += 1
            try:
                current_list = [episode]
                the_one_shot_sample = select_random_example(folder_list, all_samples_list, current_list, config)
                current_list.append(the_one_shot_sample)
                the_one_shot_sample_2 = select_random_example(folder_list, all_samples_list, current_list, config)

                episode_path = os.path.join(base_folder_path, episode)
                sample_path = os.path.join(base_folder_path, the_one_shot_sample)
                sample_path_2 = os.path.join(base_folder_path, the_one_shot_sample_2)

                final_prompt = create_prompt_messages(episode_path, sample_path, sample_path_2, config, sample_json_file)

                USE_AUDIO_IN_VIDEO = False
                with torch.no_grad():
                    text = processor.apply_chat_template(final_prompt, add_generation_prompt=True, tokenize=False)
                    audios, images, videos = process_mm_info(final_prompt, use_audio_in_video=USE_AUDIO_IN_VIDEO)
                    inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True)
                    inputs = inputs.to(model.device).to(model.dtype)
                    text_ids = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO)
                    text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    assistant_message = extract_assistant_response(text)

                output_data = {
                    "demo_answer": get_ground_truth_simple(episode_path),
                    "sample_1_path": the_one_shot_sample,
                    "sample_1": get_ground_truth(sample_path, config, sample_json_file),
                    "sample_2_path": the_one_shot_sample_2,
                    "sample_2": get_ground_truth(sample_path_2, config, sample_json_file),
                    "assistant": assistant_message
                }

                with open(output_file_path, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)

                done = True

            except Exception as e:
                print(f"[ERROR] Failed processing episode: {episode} (attempt {attempts}/{max_attempts}) — {str(e)}")
                if attempts >= max_attempts:
                    with open(failed_episodes_path, 'a', encoding='utf-8') as fail_log:
                        fail_log.write(f"{episode}\n")

def read_config(path):
    with open(path, "r") as f:
        config_data = yaml.safe_load(f)

    configs_list = config_data.get("configurations", [])
    number_of_runs = config_data.get("number_of_runs", 1)
    sample_json_path = config_data.get(SAMPLE_JSON_PATH, "")
    return configs_list, number_of_runs, sample_json_path

def get_base_path_from_config(config):
    return config.get(BASE_FOLDER_PATH)

def get_output_path_from_config(config, experiment_number):
    return os.path.join(config.get(OUTPUT_FOLDER_PATH), str(experiment_number))


def validate_input_folders(configs_list):
    for idx, config in enumerate(configs_list):
        base_path = config.get(BASE_FOLDER_PATH)
        if not base_path or not os.path.isdir(base_path):
            raise ValueError(f"[CONFIG #{idx + 1}] Invalid base folder path: {base_path}")

        folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
        if len(folders) == 0:
            raise ValueError(f"[CONFIG #{idx + 1}] Base folder '{base_path}' contains no episode folders.")

        print(f"[OK] Config #{idx + 1}: Found {len(folders)} folders in '{base_path}'")


def perform_all_experiments():
    configs_list, number_of_runs, sample_json_path = read_config(CONFIG_PATH)
    validate_input_folders(configs_list)

    needs_fiser_samples = any(c.get(USE_FISER_PROMPTING, False) for c in configs_list)
    sample_json_file = {}
    if needs_fiser_samples:
        if not sample_json_path:
            raise ValueError(
                "At least one configuration has use_fiser_prompting: true, "
                "but 'sample_json_path' is not set in the config file."
            )
        with open(sample_json_path, 'r') as f:
            sample_json_file = json.load(f)

    log_csv_path = "experiment_metrics.csv"

    for i in tqdm(range(number_of_runs), desc="Experiment repetitions"):
        for config_idx, config in enumerate(tqdm(configs_list, desc="Configuration loop", leave=False)):
            base_path = get_base_path_from_config(config)
            output_path = get_output_path_from_config(config, i)

            print(f"\n[INFO] Starting config #{config_idx + 1}, run #{i + 1}")

            mem_samples, flag, thread = track_gpu_memory(interval=1.0)
            exp_start_time = time.time()

            try:
                perform_one_experiment(base_path, sample_json_file, output_path, config)
            finally:
                flag[0] = False
                thread.join()

            exp_duration = time.time() - exp_start_time
            avg_mem = int(sum(mem_samples) / len(mem_samples)) if mem_samples else 0
            max_mem = max(mem_samples) if mem_samples else 0

            log_experiment_to_csv(log_csv_path, config_idx, i, exp_duration, avg_mem, max_mem)

            print(f"[DONE] Config #{config_idx + 1}, run #{i + 1} completed in {exp_duration:.2f} seconds — AvgMem: {avg_mem} MB, MaxMem: {max_mem} MB\n")


perform_all_experiments()
