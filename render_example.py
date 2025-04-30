
import sys
import render

for scene in ["031","002"]:
    sys.argv[1] = "--config"
    sys.argv[2] = f"configs/example/waymo_train_{scene}.yaml"
    sys.argv[3] = "mode"
    sys.argv[4] = "trajectory"
    render.start()


