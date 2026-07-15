import argparse
import time
from pathlib import Path

import mujoco
import numpy as np


WHEELS = {
    "fl": {
        "body": "fl_wheel",
        "site": "fl_wheel_center",
    },
    "fr": {
        "body": "fr_wheel",
        "site": "fr_wheel_center",
    },
    "hl": {
        "body": "hl_wheel",
        "site": "hl_wheel_center",
    },
    "hr": {
        "body": "hr_wheel",
        "site": "hr_wheel_center",
    },
}


def get_body_id(model, name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise RuntimeError(f"Body not found: {name}")
    return body_id


def get_site_id(model, name):
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if site_id < 0:
        raise RuntimeError(f"Site not found: {name}")
    return site_id


def get_fk_wheel_center_from_body(model, data, body_id, site_id):
    """
    用 wheel body 的世界位姿 + site 在 wheel body 下的局部位置，
    手动计算 site 的世界坐标。

    公式：
        p_site_world = p_body_world + R_body_world @ p_site_local
    """

    # wheel body 原点在世界坐标系下的位置
    body_pos_world = data.xpos[body_id].copy()

    # wheel body 在世界坐标系下的旋转矩阵
    body_rot_world = data.xmat[body_id].reshape(3, 3).copy()

    # site 在 wheel body 局部坐标系下的位置
    site_local_pos = model.site_pos[site_id].copy()

    # 手动 FK 计算 site 世界坐标
    fk_pos_world = body_pos_world + body_rot_world @ site_local_pos

    return fk_pos_world


def build_ids(model):
    ids = {}

    for leg, names in WHEELS.items():
        body_id = get_body_id(model, names["body"])
        site_id = get_site_id(model, names["site"])

        ids[leg] = {
            "body_name": names["body"],
            "site_name": names["site"],
            "body_id": body_id,
            "site_id": site_id,
        }

    return ids


def check_wheel_error(model, data, ids, step_id, eps, log_file):
    """
    只在误差大于 eps 时写 log。
    正常 error 接近 0 时，不输出任何东西。
    """

    for leg, item in ids.items():
        body_id = item["body_id"]
        site_id = item["site_id"]

        # 1. 手动 FK 计算 site 世界坐标
        fk_pos = get_fk_wheel_center_from_body(model, data, body_id, site_id)

        # 2. MuJoCo 直接给出的 site 世界坐标
        sim_site_pos = data.site_xpos[site_id].copy()

        # 3. error
        error_vec = fk_pos - sim_site_pos
        pos_error = np.linalg.norm(error_vec)
        z_error = error_vec[2]

        # 只有误差大于 eps 才记录
        if abs(z_error) > eps or pos_error > eps:
            log_file.write(
                f"step={step_id}, "
                f"leg={leg}, "
                f"z_error={z_error:.9e}, "
                f"pos_error={pos_error:.9e}, "
                f"fk_z={fk_pos[2]:.9f}, "
                f"sim_z={sim_site_pos[2]:.9f}, "
                f"fk_pos=[{fk_pos[0]:.9f}, {fk_pos[1]:.9f}, {fk_pos[2]:.9f}], "
                f"sim_pos=[{sim_site_pos[0]:.9f}, {sim_site_pos[1]:.9f}, {sim_site_pos[2]:.9f}]\n"
            )
            log_file.flush()


def write_header(log_file, xml_path, ids, eps, print_every):
    log_file.write("========== Wheel FK Error Checker Started ==========\n")
    log_file.write(f"xml_path={xml_path}\n")
    log_file.write(f"eps={eps}\n")
    log_file.write(f"print_every={print_every}\n")
    log_file.write("wheel mapping:\n")

    for leg, item in ids.items():
        log_file.write(
            f"  {leg}: "
            f"body={item['body_name']} body_id={item['body_id']}, "
            f"site={item['site_name']} site_id={item['site_id']}\n"
        )

    log_file.write("Only errors larger than eps will be printed.\n")
    log_file.write("====================================================\n")
    log_file.flush()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--xml",
        type=str,
        required=True,
        help="Path to MuJoCo XML file",
    )

    parser.add_argument(
        "--log",
        type=str,
        default="logs/wheel_fk_error.log",
        help="Path to output log file",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=1000000,
        help="Number of simulation steps",
    )

    parser.add_argument(
        "--print_every",
        type=int,
        default=1000,
        help="Check interval. Larger value means less overhead.",
    )

    parser.add_argument(
        "--eps",
        type=float,
        default=1e-6,
        help="Only log errors larger than this threshold.",
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional sleep time after each checked step.",
    )

    args = parser.parse_args()

    xml_path = Path(args.xml)
    if not xml_path.exists():
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    ids = build_ids(model)

    with open(log_path, "w", buffering=1) as log_file:
        write_header(
            log_file=log_file,
            xml_path=xml_path,
            ids=ids,
            eps=args.eps,
            print_every=args.print_every,
        )

        # 初始 forward
        mujoco.mj_forward(model, data)

        # 第 0 步检查一次
        check_wheel_error(
            model=model,
            data=data,
            ids=ids,
            step_id=0,
            eps=args.eps,
            log_file=log_file,
        )

        for step_id in range(1, args.steps + 1):
            mujoco.mj_step(model, data)

            if step_id % args.print_every == 0:
                check_wheel_error(
                    model=model,
                    data=data,
                    ids=ids,
                    step_id=step_id,
                    eps=args.eps,
                    log_file=log_file,
                )

                if args.sleep > 0.0:
                    time.sleep(args.sleep)

        log_file.write("========== Wheel FK Error Checker Finished ==========\n")
        log_file.flush()


if __name__ == "__main__":
    main()