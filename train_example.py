
import sys
import train

for scene in ["031","002"]:
    sys.argv[1] = "--config"
    sys.argv[2] = f"configs/example/waymo_train_{scene}.yaml"
    train.start()


