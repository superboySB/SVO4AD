from highway_env.data.trajectory import DataLoader
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--path", help="the path to the NGSIM csv file",
                    default='highway_env/data/Next_Generation_Simulation__NGSIM__Vehicle_Trajectories_and_Supporting_Data.csv')
parser.add_argument("--scene", help="location", default='i-80')  # us-101, i-80
args = parser.parse_args()

# vehicle trajectory data on southbound US 101
# and Lankershim Boulevard in Los Angeles, CA,
# eastbound I-80 in Emeryville, CA and Peachtree Street in Atlanta, Georgia.
path = args.path
scene = args.scene
data_loader = DataLoader(scene)
data_loader.read_from_csv(path)
data_loader.clean()

save_path = 'highway_env/data/processed/' + scene
if not os.path.exists(save_path):
    os.makedirs(save_path)
data_loader.dump(folder=save_path)
print(f"vehicle-{scene}=")
print([key for key, _ in data_loader.veh_dict.items()])
