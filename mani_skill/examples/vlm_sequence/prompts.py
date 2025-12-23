
vlm_sequence_prompt = '''
Target:
You are a professional robot operator. You should analyze the provided frame, ground truth of the object and end effector, and the task description. Given the frame at these timestamps, the robot will do a motion primitive action at the end of all these frames to continue the task. You need to provide the correct action primitives that, if executed correctly, would have led to successful task completion.


Analysis requirements:
1. Watch the frame carefully, the given ground truth information and the task description.
2. The ground truth contains the position of the objects and the end effector in the world frame.
3. You can only use the combination of the action primitives provided below, using json list format.
4. Your output must **strictly follow the Return Format below** and **only output** four lines (field name and value). Do not add any additional commentary.
5. You should consider the collision and the task description to decide the action primitives. For example, when you want to move the end effector to a certain position, you should consider whether the end effector will collide with the objects or the environment. If so, you should choose the action primitive that can avoid the collision, such as using two action primitives instead of moving the end effector directly to the target position.
6. When the primitives are executed, the end effector should be able to complete the task. You should output the correct action primitives until the end of the task. For example, when you move gripper, you should also try to make the gripper higher than all the objects to avoid collision with the objects.
7. When you calculate some coordinates, you should round the coordinates to 6 decimal places. When you want to release an object on another object, you should move a little higher (+z axis) to avoid collision with the other object.
8. When you decide the quaternion, you should use the format (qx, qy, qz, qw), and the values should be rounded to 6 decimal places. When you want to grasp an object, you should use the quaternion of the object to have the same orientation as the object. You should only use rotation at the pre-grasp stage, and you should output (-1, -1, -1, -1) as (qx, qy, qz, qw) when you want to move the end effector without rotation.
9. You should watch carefully to see if the end effector is closing or not. If you want to grasp something, you should output "Open Gripper" if the end effector is closing now.


Action Primitives (choose from the list below):
1. Move to (x, y, z, qx, qy, qz, qw) in the world frame
2. Close Gripper
3. Open Gripper

Return Format (use json format):
{{
    "primitives": [sequence of action primitives that would have led to successful task completion, e.g., ["Move to (0.5, 0.5, 0.5)", "Close Gripper", "Move to (1.0, 1.0, 1.0)", "Open Gripper"], using json list format]
    "evidence": [Explanation of the key evidence that led you to this sequence of action primitives]
}}

Task Description:
{task_desc}
'''




'''
Action Primitives (choose from the list below):
1. Move Left 
2. Move Right
3. Move Forward
4. Move Backward
5. Move Up
6. Move Down
7. Close Gripper
8. Open Gripper
'''


check_stage_prompt = '''
Target:
You are a professional task-stage detector for robotic manipulation. A task consists of a fixed, deterministic sequence of action primitives. Given a single frame at the current timestamp, your goal is to determine the correct stage ID from which the robot should continue, so that executing the remaining actions in order will successfully complete the task.

Analysis requirements:
1. Watch the frame carefully, and the task description.
2. Determine which actions have already been completed based on visible evidence.
3. Identify the earliest action stage that has NOT yet been completed.
4. Output the corresponding stage ID so that starting execution from this stage will allow the robot to finish the task.
5. Your output must **strictly follow the Return Format below** and **only output** four lines (field name and value). Do not add any additional commentary.


Action Primitives (Deterministic Order):
1. Pick red cube
2. Place red cube on green cube
3. Pick blue cube
4. Place blue cube on red cube

Example:
If the frame shows that the red cube is already placed on top of the green cube, then actions 1 and 2 are completed. The robot should continue from stage 3. So the output should be:
{{
    "stage": 3,
    "evidence": "The robot has already picked the red cube and placed it on the green cube."
}}


Return Format (use json format):
{{
    "stage": integer stage id, e.g., 1
    "evidence": Explanation of the key evidence that led you to this stage id
}}

Task Description:
{task_desc}
'''