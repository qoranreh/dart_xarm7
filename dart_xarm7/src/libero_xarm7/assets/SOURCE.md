# 출처

robosuite v1.5.2 (https://github.com/ARISE-Initiative/robosuite, MIT License) 에서 그대로 복사 (2026-10-08).
xArm7 은 robosuite 1.4.1 / 1.5.0 / 1.5.1 에는 없고 v1.5.2 에 처음 포함됨 (커밋 4b5c5d3 "added xarm7 (revised) (#667)", 2025-04-07).

| 여기 | 원본 (robosuite/models/assets/...) |
|---|---|
| robots/xarm7/robot.xml | robots/xarm7/robot.xml |
| robots/xarm7/meshes/{link_base,link1..7,end_tool}.stl | robots/xarm7/meshes/ (robot.xml 이 참조하는 9개만) |
| grippers/xarm7_gripper.xml | grippers/xarm7_gripper.xml |
| grippers/meshes/xarm7_gripper/*.stl | grippers/meshes/xarm7_gripper/ (gripper xml 이 참조하는 7개만) |

XML 수정 사항은 아래에 기록한다 (없으면 원본과 동일).
