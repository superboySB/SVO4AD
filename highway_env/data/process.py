from .settings import *
from .data import VehicleRecord, Snapshot, VehicleRecordList
import os

class DataLoader:
    def __init__(self, name):
        """
        NGSIM Data Loader
        params:
            name: scene in dataset
        """
        self.name = name
        self.vr_dict = dict()
        self.snap_dict = dict()
        self.veh_dict = dict()
        self.snap_ordered_list = list()
        self.veh_ordered_list = list()

    def read_from_csv(self, filename):
        """
        Columns in this Dataset (See About NGSIM)
        """
        f = open(filename, 'r')
        line = f.readline()  # 把title那行跳过去
        print('Processing raw data...')
        counter = 0

        # e.g. Vehicle record: 0, vehicle ID: 515, unixtime: 1118848075000, time: 2005-06-15 08:07:55, lane: 3, y: 188.062, x: 30.034
        self.vr_dict = dict()

        # e.g. Snapshot: unixtime: 1118848075000, number of vehs: 324
        self.snap_dict = dict()

        # e.g. Vehicle: veh_ID: 515, number of unixtimes: 3340
        self.veh_dict = dict()

        while (line):
            if counter % 10000 == 0:
                print(counter)
                print(line)
            if counter > 10000 and GLB_DEBUG:
                break
            line = f.readline().strip('\n').strip('\r').strip('\t')
            if line == "":
                continue

            words = line.split(',')
            assert (len(words) == NUM_COLS)

            if words[GLB_loc_colidx] == self.name:  # 筛选这个loc下的轨迹
                tmp_vr = VehicleRecord()
                tmp_vr.build_from_raw(counter, line)
                self.vr_dict[tmp_vr.ID] = tmp_vr
                counter += 1

                if tmp_vr.unixtime not in self.snap_dict.keys():
                    self.snap_dict[tmp_vr.unixtime] = Snapshot(tmp_vr.unixtime)
                self.snap_dict[tmp_vr.unixtime].add_vr(tmp_vr)

                if tmp_vr.veh_ID not in self.veh_dict.keys():
                    self.veh_dict[tmp_vr.veh_ID] = VehicleRecordList(tmp_vr.veh_ID)
                self.veh_dict[tmp_vr.veh_ID].add_vr(tmp_vr)

        self.snap_ordered_list = list(self.snap_dict.keys())
        self.veh_ordered_list = list(self.veh_dict.keys())
        self.snap_ordered_list.sort()
        self.veh_ordered_list.sort()

        for tmp_unixtime, tmp_snap in self.snap_dict.items():
            tmp_snap.sort_vehs()

        for tmp_vehID, tmp_veh in self.veh_dict.items():
            tmp_veh.sort_time()

        f.close()

    def dump(self, folder, vr_filename='vehicle_record_file.csv', v_filename='vehicle_file.csv',
             snapshot_filename='snapshot_file.csv'):
        """
            Dump vehicle record, vehicle and snapshot to Local
        """
        print('Dumping processed data...')
        f_vr = open(os.path.join(folder, vr_filename), 'w')
        for vr_ID, vr in self.vr_dict.items():
            f_vr.write(vr.to_string() + '\n')
        f_vr.close()

        f_v = open(os.path.join(folder, v_filename), 'w')
        for _, v in self.veh_dict.items():
            f_v.write(v.to_string() + '\n')
        f_v.close()

        f_ss = open(os.path.join(folder, snapshot_filename), 'w')
        for _, ss in self.snap_dict.items():
            f_ss.write(ss.to_string() + '\n')
        f_ss.close()

    def load(self, folder, vr_filename='vehicle_record_file.csv', v_filename='vehicle_file.csv',
             snapshot_filename='snapshot_file.csv'):
        self.vr_dict = dict()
        self.snap_dict = dict()
        self.veh_dict = dict()
        print("Loading trajectories of chosen veh-ids from NGSIM...")

        # records
        f_vr = open(os.path.join(folder, vr_filename), 'r')
        for line in f_vr:
            if line == '':
                continue
            words = line.rstrip('\n').rstrip('\r').split(',')
            assert (len(words) == 17)
            tmp_vr = VehicleRecord()
            tmp_vr.build_from_processed(self.name, words)
            self.vr_dict[tmp_vr.ID] = tmp_vr
        f_vr.close()

        # vehicle
        f_v = open(os.path.join(folder, v_filename), 'r')
        for line in f_v:
            if line == '':
                continue
            words = line.rstrip('\n').rstrip('\r').split(',')
            assert (len(words) > 1)
            tmp_v = VehicleRecordList()
            tmp_v.build_from_processed(words, self.vr_dict)
            self.veh_dict[tmp_v.veh_ID] = tmp_v
        f_v.close()

        # snapshot
        f_ss = open(os.path.join(folder, snapshot_filename), 'r')
        for line in f_ss:
            if line == '':
                continue
            words = line.rstrip('\n').rstrip('\r').split(',')
            assert (len(words) > 1)
            tmp_ss = Snapshot()
            tmp_ss.build_from_processed(words, self.vr_dict)
            self.snap_dict[tmp_ss.unixtime] = tmp_ss
        f_ss.close()

        # ordered list
        self.snap_ordered_list = list(self.snap_dict.keys())
        self.veh_ordered_list = list(self.veh_dict.keys())
        self.snap_ordered_list.sort()
        self.veh_ordered_list.sort()

        for tmp_unixtime, tmp_snap in self.snap_dict.items():
            tmp_snap.sort_vehs()
        for tmp_vehID, tmp_veh in self.veh_dict.items():
            tmp_veh.sort_time()

    # Especially used for us-101, clean duplicate record
    def clean(self):
        for unixtime, snap in self.snap_dict.items():
            veh_ID_list = list(map(lambda x: x.veh_ID, snap.vr_list))
            veh_ID_set = set(veh_ID_list)

            if len(veh_ID_list) > len(veh_ID_set):
                new_vr_list = list()
                new_vr_ID_set = set()
                for vr in snap.vr_list:
                    if vr.veh_ID not in new_vr_ID_set:
                        new_vr_list.append(vr)
                        new_vr_ID_set.add(vr.veh_ID)
                    self.snap_dict[unixtime].vr_list = new_vr_list

    def down_sample(self, sample_rate=3000):
        self.vr_dict = {k: v for (k, v) in self.vr_dict.items() if v.unixtime % sample_rate == 0}
        self.snap_dict = {k: v for (k, v) in self.snap_dict.items() if k % sample_rate == 0}
        for veh in self.veh_dict.values():
            veh.down_sample(sample_rate)
        self.snap_ordered_list = list(filter(lambda x: x % sample_rate == 0, self.snap_ordered_list))
