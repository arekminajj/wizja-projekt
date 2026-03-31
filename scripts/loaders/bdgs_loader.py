import os
import random
from pathlib import Path

from scripts.loaders.base_loader import BaseDatasetLoader
from scripts.vars import TRAINING_IMAGES_PATH


class BDGSDatasetLoader(BaseDatasetLoader):
    def get_learning_files(skip_empty=True, shuffle=True, limit=None, offset=0,
                           limit_recordings_of_single_person_single_gesture=None,
                           limit_images_in_single_person_single_recording=None,
                           limit_people=None, base_path=TRAINING_IMAGES_PATH):
        image_files = []
        classify_file = None

        visited_paths = {}
        visited_people = {}

        for root, _, files in os.walk(base_path):
            if ".git" in root:
                continue
            for file in files:
                if file.lower().endswith(".txt"):
                    classify_file = os.path.join(root, file)
                    break
            if classify_file is None or len(files) == 0: continue

            root = Path(root).resolve()
            parent_path = root.parent
            people_path = parent_path.parents[1].name
            visited_paths[parent_path] = visited_paths.get(parent_path, 0) + 1
            visited_people[people_path] = visited_people.get(people_path, 0) + 1

            if limit_recordings_of_single_person_single_gesture is not None and visited_paths.get(parent_path,
                                                                                                  0) > limit_recordings_of_single_person_single_gesture:
                continue
            if limit_people is not None and len(visited_people) > limit_people:
                break

            with open(classify_file, "r") as f:
                classify_row = [line.split("\n")[0] for line in f]
            files = sorted(files)
            files.pop(0)
            bg_image = files[0]

            added = 0
            for index in range(len(files) - 1):
                if files[index].lower().endswith(('.png', '.jpg', '.jpeg')):
                    if limit_images_in_single_person_single_recording is not None and added >= limit_images_in_single_person_single_recording:
                        break
                    if skip_empty:
                        if classify_row[index].split(" ")[0] != "0":
                            image_files.append(
                                (os.path.join(root, files[index]), classify_row[index], (os.path.join(root, bg_image))))
                            added += 1
                    else:
                        image_files.append(
                            (os.path.join(root, files[index]), classify_row[index], (os.path.join(root, bg_image))))
                        added += 1

        if shuffle: random.shuffle(image_files)
        return image_files[offset:(limit + offset if limit is not None else None)]
