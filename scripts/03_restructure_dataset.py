import os
import shutil


def restructure_dataset(images_dir, raw_dir, enhanced_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    episode_files = [f for f in os.listdir(raw_dir) if f.endswith(".json")]
    episode_names = [os.path.splitext(f)[0] for f in episode_files]

    for episode in episode_names:
        episode_folder = os.path.join(output_dir, episode)
        pics_dest = os.path.join(episode_folder, "pics")

        raw_file_src = os.path.join(raw_dir, f"{episode}.json")
        enhanced_file_src = os.path.join(enhanced_dir, f"{episode}.json")
        images_src_folder = os.path.join(images_dir, episode)

        os.makedirs(episode_folder, exist_ok=True)

        if os.path.exists(raw_file_src):
            shutil.copy(raw_file_src, os.path.join(episode_folder, "raw_file.json"))
        else:
            print(f" Missing raw file for episode: {episode}")

        if os.path.exists(enhanced_file_src):
            shutil.copy(enhanced_file_src, os.path.join(episode_folder, "enhanced_text.json"))
        else:
            print(f" Missing enhanced file for episode: {episode}")

        if os.path.exists(images_src_folder):
            shutil.copytree(images_src_folder, pics_dest, dirs_exist_ok=True)
        else:
            print(f" Missing image folder for episode: {episode}")

    print(f" Dataset restructuring complete: {output_dir}")


restructure_dataset(
    images_dir="data/images",
    raw_dir="data/raw_episodes",
    enhanced_dir="data/enhanced_text",
    output_dir="data/dataset"
)
