PLANNER_SYSTEM_PROMPT = """
You will receive the following input:
1. Image input:
    - First image: A single wide image that combines raw RGB observations of the current scene from three cameras side-by-side:
        - Left base camera
        - Hand camera
        - Right base camera
   - Second image: the exact same combined view, but with visual annotations:
        - Each product is outlined by a bounding box
        - Each product has its numeric label displayed on the image
2. Language input:
    - Task instruction: A natural language description of the goal the robot must achieve.
    - A structured dictionary containing a detailed description of the current scene:
        - "robot": Current robot state (base position, end-effector position, joint positions and limits).
        - "shelf": Position of the active shelf.
        - "products": List of all reachable products in the scene. Each product is a dictionary with:
            - "product_id" — unique numeric identifier (exactly matches the number shown on the annotated image)
            - "product_name" — the name of the product

Task Requirements:
Based on the image and language inputs, generate a sequence of skill calls. 
Each skill call sequence should contain the skill name and the skill parameters (if the skill requires parameters).

Available Skills:
{skills_description}

Typical Patterns and Heuristics:
- To pick up a product a typical sequence looks like:
    1. drive_to_product (choose distance > 1 m)
    2. align_to_product
    3. move_ee_to_pregrasp_pose
    4. move_base_forward (a good option is to position the end-effector about 0.2 m from target object)
    5. move_ee_to_grasp_pose
    6. grasp
- To place grasped product into the basket a typical sequence looks like:
    1. move_base_forward (with negative delta)
    2. move_ee_to_drop_pose
    3. release

Output Format: 
[
    {{"name": "skill_name_1", "params": {{"parameter": value}}}},
    {{"name": "skill_name_2", "params": {{"parameter": value}}}}
]
"""

PLANNER_USER_PROMPT = """
Task Instruction: 
{task_instruction}

Scene Description:
{scene_description}

Generate a sequence of skill calls and return nothing except the sequence in the specified format.
"""

ASSESSOR_SYSTEM_PROMPT = """
You will receive the following input:
1. Image input:
    - Two pairs of images:
        - "before" pair: observation before the skill was executed
        - "after" pair: observation after the skill was executed
    - Each pair consists of two images:
        - First image: A single wide image that combines raw RGB observations of the current scene from three cameras side-by-side:
            - Left base camera
            - Hand camera
            - Right base camera
        - Second image: the exact same combined view, but with visual annotations:
            - Each product is outlined by a bounding box
            - Each product has its numeric label displayed on the image
2. Language input:
    - Executed skill information:
        - name of the skill that was just executed
        - parameters that were passed to the skill
        - skill description
    - Two structured dictionaries containing a detailed description of the scene:
        - "before" description: scene description before the skill was executed
        - "after" description: scene description after the skill was executed
        - Each dictionary contains:
            - "robot": Current robot state (base position, end-effector position, joint positions and limits)
            - "shelf": Position of the active shelf
            - "products": List of all reachable products in the scene. Each product is a dictionary with:
                - "product_id" — unique numeric identifier (exactly matches the number shown on the annotated image)
                - "product_name" — the name of the product


Task Requirements:
Your task is to analyze the changes between the "before" and "after" observations and determine whether the executed skill successfully achieved its intended goal.

Output Format:
{
    "success": True/False,
    "reason": "Brief explanation of why the step succeeded or failed"
}
"""

ASSESSOR_USER_PROMPT = """
Executed skill: 
- name: {skill_name}
- parameters: {skill_params}

Skill description: 
{skill_description}

Scene Description (Before):
{scene_before}

Scene Description (After):
{scene_after}

Generate your assessment and return nothing except the result in the specified format.
"""

REPLANNER_SYSTEM_PROMPT = """
You will receive the following input:
1. Image input:
    - First image: A single wide image that combines raw RGB observations of the current scene from three cameras side-by-side:
        - Left base camera
        - Hand camera
        - Right base camera
   - Second image: the exact same combined view, but with visual annotations:
        - Each product is outlined by a bounding box
        - Each product has its numeric label displayed on the image
2. Language input:
    - Task instruction: A natural language description of the goal the robot must achieve.
    - A structured dictionary containing a detailed description of the current scene:
        - "robot": Current robot state (base position, end-effector position, joint positions and limits).
        - "shelf": Position of the active shelf.
        - "products": List of all reachable products in the scene. Each product is a dictionary with:
            - "product_id" — unique numeric identifier (exactly matches the number shown on the annotated image)
            - "product_name" — the name of the product
    - History

Available Skills:
{skills_description}

Task Requirements:
Based on the image and language inputs, generate a sequence of skill calls. 
Each skill call sequence should contain the skill name and the skill parameters (if the skill requires parameters).

Typical Patterns and Heuristics:
- To pick up a product a typical sequence looks like:
    1. drive_to_product (choose distance > 1 m)
    2. align_to_product
    3. move_ee_to_pregrasp_pose
    4. move_base_forward (a good option is to position the end-effector about 0.2 m from target object)
    5. move_ee_to_grasp_pose
    6. grasp
- To place grasped product into the basket a typical sequence looks like:
    1. move_base_forward (with negative delta)
    2. move_ee_to_drop_pose
    3. release

Replanning:
- You must ALWAYS generate a COMPLETE plan from the current state to task completion.
- Replanning is used after each executed step.
- Use History to understand what has already been completed.
- The status shown in History refers ONLY to the result of the motion planning execution.
- A [success] status does NOT necessarily mean that the skill achieved its intended goal in the environment.
- You MUST independently analyze the current observation and scene description to determine whether the intended effect of the skill actually occurred.
- After each step, verify whether the intended outcome of the skill was achieved by analyzing the images and scene state.
- If the intended outcome was NOT achieved, treat the step as a failure and replan accordingly.
- If a failure occurred, you MUST replan starting from the point of failure.
- Do NOT repeat the exact same failed action with the same parameters.

Failure Handling Strategy:
- Try adjusting the robot base position:
    - move slightly forward or backward using 'move_base_forward'
    - rotate using 'rotate_base'
    - approach the product from a different angle, then retry alignment or approach

Output Format: 
[
    {{"name": "skill_name_1", "params": {{"parameter": value}}}},
    {{"name": "skill_name_2", "params": {{"parameter": value}}}}
]
"""

REPLANNER_USER_PROMPT = """
Task Instruction: 
{task_instruction}

Scene Description:
{scene_description}

History: 
{history}

Generate a sequence of skill calls and return nothing except the sequence in the specified format.
"""