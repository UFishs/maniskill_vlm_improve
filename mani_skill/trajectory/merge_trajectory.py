import argparse
from pathlib import Path
import h5py
from mani_skill.utils.logging_utils import logger

from mani_skill.utils.io_utils import dump_json, load_json


def merge_trajectories(output_path: str, traj_paths: list, num_demos: list, recompute_id: bool = True):
    """
    Merges multiple JSON and H5 files into a single JSON and H5 file.

    This function combines the contents of multiple JSON and H5 files. It keeps the first value for all keys
    (other than "episodes") and logs a warning for any differences. The "episodes" from each JSON file are merged
    into a single list, and the corresponding H5 data is copied to the output H5 file.

    Args:
        output_path (str): The path to the output H5 file. The corresponding JSON file will be saved with the same
                           name but with a .json extension.
        traj_paths (list): A list of paths to the input trajectory files (H5 files). The corresponding JSON files
                           should have the same name but with a .json extension.
        recompute_id (bool): If True, recompute the episode IDs to ensure they are unique. If False, keep the original
                             episode IDs.

    Raises:
        AssertionError: If there is a conflict in the episode IDs when recompute_id is False.
    """
    if num_demos is None:
        num_demos = [None] * len(traj_paths)

    logger.info(f"Merging {output_path}")

    merged_h5_file = h5py.File(output_path, "w")
    merged_json_path = output_path.replace(".h5", ".json")
    merged_json_data = {"episodes": []}
    cnt = 0

    for file_idx, traj_path in enumerate(traj_paths):
        traj_path = str(traj_path)
        logger.info(f"Merging{traj_path}")

        with h5py.File(traj_path, "r") as h5_file:
            json_data = load_json(traj_path.replace(".h5", ".json"))
            
            # For keys other than episodes, keep the first data
            # and check if there is any conflict with other data.
            for key, value in json_data.items():
                if key == "episodes":
                    continue
                if key not in merged_json_data:
                    merged_json_data[key] = value
                else:
                    if merged_json_data[key] != value:
                        logger.warning(f"Conflict detected for key {key} in {traj_path}: {merged_json_data[key]} != {value}")

            # Merge episodes

            if num_demos[file_idx] is not None:
                used_episodes = json_data["episodes"][:num_demos[file_idx]]
            else:
                used_episodes = json_data["episodes"]


            for ep in used_episodes:
                episode_id = ep["episode_id"]
                traj_id = f"traj_{episode_id}"

                # Copy h5 data
                if recompute_id:
                    new_traj_id = f"traj_{cnt}"
                else:
                    new_traj_id = traj_id

                assert new_traj_id not in merged_h5_file, new_traj_id
                h5_file.copy(traj_id, merged_h5_file, new_traj_id)

                # Copy json data
                if recompute_id:
                    ep["episode_id"] = cnt
                merged_json_data["episodes"].append(ep)

                cnt += 1

    merged_h5_file.close()
    dump_json(merged_json_path, merged_json_data, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dirs", nargs="+")
    parser.add_argument("-o", "--output-path", type=str)
    parser.add_argument("-p", "--pattern", type=str, default="trajectory.h5")
    args = parser.parse_args()


    traj_paths = []
    if args.input_dirs:
        for input_dir in args.input_dirs:
            input_dir = Path(input_dir)
            traj_paths.extend(sorted(input_dir.rglob(args.pattern)))
    

    
    # level_name = 'base'
    # level_name = 'stage_1'
    # level_name = 'stage_2'
    level_name = 'stage_3'
    # level_name = 'stage_4'
    merged_iter_idx = 0
    # env_name = 'mugcleanup'
    env_name = 'stackpyramid'
    # seeds = [1000, 2000, 3000, 4000, 5000, 6000]
    seeds = [2000, 3000, 4000]


    args.output_path = f'/cephfs/gyshare/ruizihang/maniskill_vlm_improve/diffusion_policy/data/eval/{env_name}/system_collect/iter_{merged_iter_idx+1}_merge/iter_{merged_iter_idx+1}_{level_name}_merge.h5'
    base_traj_path = f'/cephfs/gyshare/ruizihang/maniskill_vlm_improve/diffusion_policy/data/eval/{env_name}/system_collect/iter_{merged_iter_idx}_merge/iter_{merged_iter_idx}_{level_name}_merge.h5'
    if merged_iter_idx == 0:
        # base_traj_path = '/cephfs/gyshare/ruizihang/maniskill_vlm_improve/demos/MugCleanup-v1/motionplanning/base_traj_200.rgb.pd_joint_delta_pos.physx_cpu.h5'
        # base_traj_path = '/cephfs/gyshare/ruizihang/maniskill_vlm_improve/demos/MugCleanup-v1/motionplanning/split_200_pd_joint_delta_pos/stage_2.h5'
        # base_traj_path = '/cephfs/gyshare/ruizihang/maniskill_vlm_improve/demos/MugCleanup-v1/motionplanning/split_200_pd_joint_delta_pos/stage_3.h5'
        
        base_traj_path = '/cephfs/gyshare/ruizihang/maniskill_vlm_improve/demos/StackPyramid-v1/motionplanning/base_traj_200.rgb.pd_joint_delta_pos.physx_cpu.h5'
        base_traj_path = '/cephfs/gyshare/ruizihang/maniskill_vlm_improve/demos/StackPyramid-v1/motionplanning/split_200_pd_joint_delta_pos/stage_1.h5'
        base_traj_path = '/cephfs/gyshare/ruizihang/maniskill_vlm_improve/demos/StackPyramid-v1/motionplanning/split_200_pd_joint_delta_pos/stage_2.h5'
        base_traj_path = '/cephfs/gyshare/ruizihang/maniskill_vlm_improve/demos/StackPyramid-v1/motionplanning/split_200_pd_joint_delta_pos/stage_3.h5'

        

    traj_paths = [base_traj_path]
    for seed in seeds:
        new_path = f'/cephfs/gyshare/ruizihang/maniskill_vlm_improve/diffusion_policy/data/eval/{env_name}/system_collect/iter_{merged_iter_idx+1}_seed_{seed}/{level_name}_record/iter_{merged_iter_idx+1}_{level_name}.h5'
        traj_paths.append(new_path)

    print(traj_paths)

    num_demos = None



    # args.output_path = '/cephfs/gyshare/ruizihang/maniskill_vlm_improve/diffusion_policy/data/eval/system_collect/iter_1_merge/iter_1_base_merge_vlm.h5'
    # traj_paths = [
    #     '/cephfs/gyshare/ruizihang/maniskill_vlm_improve/demos/StackThree-v1/motionplanning/base_traj_200.rgb.pd_joint_delta_pos.physx_cpu.h5',
    #     '/cephfs/gyshare/ruizihang/maniskill_vlm_improve/diffusion_policy/data/eval/system_collect/iter_1_seed_1000/base_record/vlm_succ_base_record.h5',
    #     '/cephfs/gyshare/ruizihang/maniskill_vlm_improve/diffusion_policy/data/eval/system_collect/iter_1_seed_2000/base_record/vlm_succ_base_record.h5',
    # ]
    # num_demos = None

    # args.output_path = '/cephfs/gyshare/ruizihang/maniskill_vlm_improve/diffusion_policy/data/eval/system_collect/iter_1_merge/iter_1_base_merge_primitive.h5'
    # traj_paths = [
    #     '/cephfs/gyshare/ruizihang/maniskill_vlm_improve/demos/StackThree-v1/motionplanning/base_traj_200.rgb.pd_joint_delta_pos.physx_cpu.h5',
    #     '/cephfs/gyshare/ruizihang/maniskill_vlm_improve/diffusion_policy/data/eval/system_collect/iter_1_seed_1000/base_record/primitive_succ_base_record.h5',
    #     '/cephfs/gyshare/ruizihang/maniskill_vlm_improve/diffusion_policy/data/eval/system_collect/iter_1_seed_2000/base_record/primitive_succ_base_record.h5',
    # ]
    # num_demos = None

    # args.output_path = '/cephfs/gyshare/ruizihang/maniskill_vlm_improve/diffusion_policy/data/eval/system_collect/iter_1_merge/iter_1_base_merge_primitive_364.h5'
    # traj_paths = [
    #     '/cephfs/gyshare/ruizihang/maniskill_vlm_improve/demos/StackThree-v1/motionplanning/base_traj_200.rgb.pd_joint_delta_pos.physx_cpu.h5',
    #     '/cephfs/gyshare/ruizihang/maniskill_vlm_improve/diffusion_policy/data/eval/system_collect/iter_1_seed_1000/base_record/primitive_succ_base_record.h5',
    #     '/cephfs/gyshare/ruizihang/maniskill_vlm_improve/diffusion_policy/data/eval/system_collect/iter_1_seed_2000/base_record/primitive_succ_base_record.h5',
    #     '/cephfs/gyshare/ruizihang/maniskill_vlm_improve/diffusion_policy/data/eval/system_collect/iter_1_seed_3000/base_record/primitive_succ_base_record.h5',
    #     '/cephfs/gyshare/ruizihang/maniskill_vlm_improve/diffusion_policy/data/eval/system_collect/iter_1_seed_4000/base_record/primitive_succ_base_record.h5',
    #     '/cephfs/gyshare/ruizihang/maniskill_vlm_improve/diffusion_policy/data/eval/system_collect/iter_1_seed_5000/base_record/primitive_succ_base_record.h5',
    #     '/cephfs/gyshare/ruizihang/maniskill_vlm_improve/diffusion_policy/data/eval/system_collect/iter_1_seed_6000/base_record/primitive_succ_base_record.h5'
    # ]
    # num_demos = [
    #     200,
    #     36,
    #     29,
    #     32,
    #     20,
    #     32,
    #     15
    # ]

    # args.output_path = '/cephfs/gyshare/ruizihang/maniskill_vlm_improve/diffusion_policy/data/eval/mugcleanup/bootstrapped/iter_1_merge/bootstrapped_iter_1_merge.h5'
    # traj_paths = [
    #     '/cephfs/gyshare/ruizihang/maniskill_vlm_improve/demos/MugCleanup-v1/motionplanning/base_traj_200.rgb.pd_joint_delta_pos.physx_cpu.h5',
    #     '/cephfs/gyshare/ruizihang/maniskill_vlm_improve/diffusion_policy/data/eval/mugcleanup/bootstrapped/iter_1_seed_3000/bootstrapped_record/iter_1_bootstrapped.h5',
    # ]
    # num_demos = None

    # args.output_path = '/cephfs/gyshare/ruizihang/maniskill_vlm_improve/diffusion_policy/data/eval/stackthree/bootstrapped/iter_1_merge/bootstrapped_iter_1_merge.h5'
    # traj_paths = [
    #     '/cephfs/gyshare/ruizihang/maniskill_vlm_improve/demos/StackThree-v1/motionplanning/base_traj_200.rgb.pd_joint_delta_pos.physx_cpu.h5',
    #     '/cephfs/gyshare/ruizihang/maniskill_vlm_improve/diffusion_policy/data/eval/stackthree/bootstrapped/iter_1_seed_1000/bootstrapped_record/iter_1_bootstrapped.h5',
    #     '/cephfs/gyshare/ruizihang/maniskill_vlm_improve/diffusion_policy/data/eval/stackthree/bootstrapped/iter_1_seed_2000/bootstrapped_record/iter_1_bootstrapped.h5',
    #     '/cephfs/gyshare/ruizihang/maniskill_vlm_improve/diffusion_policy/data/eval/stackthree/bootstrapped/iter_1_seed_3000/bootstrapped_record/iter_1_bootstrapped.h5',
    #     '/cephfs/gyshare/ruizihang/maniskill_vlm_improve/diffusion_policy/data/eval/stackthree/bootstrapped/iter_1_seed_4000/bootstrapped_record/iter_1_bootstrapped.h5'

    # ]
    # num_demos = None



    output_dir = Path(args.output_path).parent
    output_dir.mkdir(exist_ok=True, parents=True)

    merge_trajectories(args.output_path, traj_paths, num_demos)


if __name__ == "__main__":
    main()
