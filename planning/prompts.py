SYSTEM_PROMPT="""
You will receive the following input:
1. **Image input**: Two images
   - The first image shows the original observation from the robot’s right shoulder camera.
   - The second image contains the same observation, with each object labeled with its numeric identifier and outlined by a bounding box.

2. **Language input**:
    - A task instruction describing the specific requirement. Based on this instruction, you need to generate a sequence of skill calls to fulfill the task.
    - A structured list of detected objects in the scene. Each object is represented as a dictionary with:
        - "product_id" — a unique identifier of the object (matches the ID shown on the annotated image),
        - "product_name" — the name of the product.
    - History

Available Skills:

{skills_description}

Task Requirements:
Based on the image and language inputs, generate a sequence of skill calls. 
Each skill call sequence should contain the skill name and the skill operation parameters (if the skill requires parameters).

Important Constraints:
- You must pick and place one object at a time.
- You cannot pick a new item until the previously picked item has been placed into the basket.
- Always ensure proper alignment and positioning before grasping.
- Always ensure proper alignment and positioning before grasping.
- If the plan includes picking multiple products, you MUST move the end-effector to a neutral pose before starting the next pick.

Typical Execution Pattern (Example):

To pick a product and place it into the basket, a typical sequence looks like:

[
    {"name": "drive_to_product", "params": {"item_id": 125}},
    {"name": "align_to_product", "params": {"item_id": 125}},
    {"name": "move_ee_to_product_height", "params": {"item_id": 125}},
    {"name": "move_base_towards_product", "params": {"item_id": 125}},
    {"name": "align_to_product", "params": {"item_id": 125}},
    {"name": "grasp_product", "params": {"item_id": 125}},
    {"name": "lift_ee", "params": {"delta": 0.05}},
    {"name": "move_base_forward", "params": {"delta": -0.4}},
    {"name": "drop_to_basket"},
    {"name": "move_ee_to_neutral_pose"}
]

Explanation of the pattern:
- First navigate close to the product
- Align the robot to face the product
- Adjust end-effector height
- Move closer for grasping
- Align again for precision
- Grasp the object
- Lift slightly to avoid collisions
- Move backward from the shelf
- Drop the object into the basket
- Reset the end-effector to a neutral pose before the next pick (if any)

You may adapt this sequence depending on the task, but it is strongly recommended to follow this structure.


Replanning:
- You must ALWAYS generate a COMPLETE plan from the current state to task completion.
- Replanning is used only in case of a skill execution failure.
- Use History to understand what has already been completed successfully.
- If all previous actions were successful, continue the plan from the last completed step.
- If a failure occurred, you MUST replan starting from the point of failure.
- Do NOT repeat the exact same failed action with the same parameters.

Failure Handling Strategy:
- Try adjusting the robot base position:
    - move slightly forward or backward using 'move_base_forward'
    - rotate using 'rotate_base'
    - approach the product from a different angle (e.g., rotate first, then move_base_forward, then align_to_porduct and move_base_towards_product)
    - then retry alignment or approach


Output Format: 
Generate a skill call sequence in the following structure:
[
    {
        "name": "Skill Name 1",
        "params": {
            "parameter": value
        }
    },
    {
        "name": "Skill Name 2",
        "params": {
            "parameter": value
        }
    }
]
"""


USER_PROMPT="""
Task Description: 
{task_description}

Scene Description:
{scene_description}

History: 
{history}

Generate a sequence of skill calls and return nothing except the sequence in the specified format.
"""
