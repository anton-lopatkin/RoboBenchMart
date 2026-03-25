SYSTEM_PROMPT="""
You will receive the following input:
1. **Image input**: Two images
   - The first image shows the original observation from the robot’s right shoulder camera.
   - The second image contains the same observation, with each object labeled with its numeric identifier and outlined by a bounding box.
2. **Language input**:
    - A task instruction describing the specific requirement. Based on this instruction, you need to generate a sequence of skill calls to fulfill the task.
    - A structured list of detected objects in the scene. Each object is represented as a dictionary with:
        - "product_id" — a unique identifier of the object (matches the ID shown on the annotated image),
        - "product_name" — the name of the product,
        - "bbox" — a list of four numbers [x_min, y_min, x_max, y_max] representing the bounding box of the object in image coordinates.

Available Skills:
1. `pick_item(item_id: int)` 
    - picks up the specified object using its identifier
    - item_id corresponds to the object identifier shown in the labeled image and in the bounding box description
2. `place_to_basket()`
    - places the currently held object into the basket

Task Requirements:
Based on the image and language inputs, generate a sequence of skill calls. 
Each skill call sequence should contain the skill name and the skill operation parameters (if the skill requires parameters).

Output Format: 
Generate a skill call sequence in the following structure:
[
    {
        "name": "Skill Name 1",
        "params": {
            "parameter": "value"
        }
    },
    {
        "name": "Skill Name 2",
        "params": {
            "parameter": "value"
        }
    }
]

You cannot pick a new item until the previously picked item has been placed into the basket.
"""

USER_PROMPT="""
Task Description: {task_description}

Scene Description: {scene_description}

Generate a sequence of skill calls and return nothing except the sequence in the specified format.
"""
