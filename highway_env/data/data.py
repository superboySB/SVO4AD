import numpy as np
import pytz
from .settings import *

class VehicleRecord:
    def __init__(self):
        self.ID = None
        self.veh_ID = None
        self.frame_ID = None
        self.unixtime = None

    def build_from_raw(self, ID, s1):
        self.ID = ID
        words = s1.split(',')
        assert (len(words) == NUM_COLS)

        tz = pytz.timezone(timezone_dict[words[GLB_loc_colidx]])
        self.veh_ID = np.int(words[GLB_vehID_colidx])
        # self.frame_ID = np.int(words[GLB_frmID_colidx])
        self.unixtime = np.int(words[GLB_glbtime_colidx])
        self.time = datetime.datetime.fromtimestamp(np.float(self.unixtime) / 1000, tz)
        self.x = np.float(words[GLB_locx_colidx])
        self.y = np.float(words[GLB_locy_colidx])
        self.lat = np.float(words[GLB_glbx_colidx])
        self.lon = np.float(words[GLB_glby_colidx])
        self.len = np.float(words[GLB_vehlen_colidx])
        self.wid = np.float(words[GLB_vehwid_colidx])
        self.cls = np.int(words[GLB_vehcls_colidx])
        self.spd = np.float(words[GLB_vehspd_colidx])
        self.acc = np.float(words[GLB_vehacc_colidx])
        self.lane_ID = np.int(words[GLB_laneID_colidx])
        # self.intersection_ID = np.int(words[GLB_interID_colidx])
        self.pred_veh_ID = np.int(words[GLB_pred_colidx])
        self.follow_veh_ID = np.int(words[GLB_follow_colidx])
        self.shead = np.float(words[GLB_shead_colidx])
        self.thead = np.float(words[GLB_thead_colidx])

    def build_from_processed(self, name, words):
        assert (len(words) == 17)
        self.ID = np.int(words[0])
        self.veh_ID = np.int(words[1])
        self.unixtime = np.int(words[2])
        tz = pytz.timezone(timezone_dict[name])
        self.time = datetime.datetime.fromtimestamp(np.float(self.unixtime) / 1000, tz)
        self.x = np.float(words[3])
        self.y = np.float(words[4])
        self.lat = np.float(words[5])
        self.lon = np.float(words[6])
        self.len = np.float(words[7])
        self.wid = np.float(words[8])
        self.cls = np.int(words[9])
        self.spd = np.float(words[10])
        self.acc = np.float(words[11])
        self.lane_ID = np.int(words[12])
        self.pred_veh_ID = np.int(words[13])
        self.follow_veh_ID = np.int(words[14])
        self.shead = np.float(words[15])
        self.thead = np.float(words[16])

    def __str__(self):
        return ("Vehicle record: {}, vehicle ID: {}, unixtime: {}, time: {}, lane: {}, y: {}, x: {}".format(self.ID,
                                                                                                            self.veh_ID,
                                                                                                            self.unixtime,
                                                                                                            self.time.strftime(
                                                                                                                "%Y-%m-%d %H:%M:%S"),
                                                                                                            self.lane_ID,
                                                                                                            self.y,
                                                                                                            self.x))

    def __repr__(self):
        return self.__str__()

    def to_string(self):
        return ','.join([str(e) for e in [self.ID, self.veh_ID, self.unixtime,
                                          self.x, self.y, self.lat, self.lon,
                                          self.len, self.wid, self.cls,
                                          self.spd, self.acc, self.lane_ID,
                                          self.pred_veh_ID, self.follow_veh_ID, self.shead, self.thead]])


class Snapshot:
    def __init__(self, unixtime=None):
        self.unixtime = unixtime
        self.vr_list = list()

    def build_from_processed(self, words, vr_dict):
        assert (len(words) > 1)
        self.unixtime = np.int(words[0])
        self.vr_list = list(map(lambda x: vr_dict[np.int(x)], words[1:]))

    def add_vr(self, vr):
        assert (vr.unixtime == self.unixtime)
        self.vr_list.append(vr)

    def sort_vehs(self, ascending=True):
        self.vr_list = sorted(self.vr_list, key=lambda x: (x.y, x.lane_ID), reverse=(not ascending))

    def __str__(self):
        return ("Snapshot: unixtime: {}, number of vehs: {}".format(self.unixtime, len(self.vr_list)))

    def __repr__(self):
        return self.__str__()

    def to_string(self):
        return ','.join([str(e) for e in [self.unixtime] + list(map(lambda x: x.ID, self.vr_list))])


class VehicleRecordList:
    def __init__(self, veh_ID=None):
        self.veh_ID = veh_ID
        self.vr_list = list()
        self.trajectory = list()

    def build_from_processed(self, words, vr_dict):
        assert (len(words) > 1)
        self.veh_ID = np.int(words[0])
        self.vr_list = list(map(lambda x: vr_dict[np.int(x)], words[1:]))

    def add_vr(self, vr):
        assert (vr.veh_ID == self.veh_ID)
        self.vr_list.append(vr)

    def sort_time(self, ascending=True):
        self.vr_list = sorted(self.vr_list, key=lambda x: (x.unixtime), reverse=(not ascending))

    def __str__(self):
        return ("Vehicle: veh_ID: {}, number of unixtimes: {}".format(self.veh_ID, len(self.vr_list)))

    def __repr__(self):
        return self.__str__()

    def to_string(self):
        return ','.join([str(e) for e in [self.veh_ID] + list(map(lambda x: x.ID, self.vr_list))])

    # downsampl, interval unit: ms
    def down_sample(self, sample_rate):
        self.vr_list = list(filter(lambda x: x.unixtime % sample_rate == 0, self.vr_list))

    def get_stayed_lanes(self):
        return list(set(list(map(lambda x: x.lane_ID, self.vr_list))))

    def build_trajectory(self):
        vr_list = self.vr_list
        assert (len(vr_list) > 0)
        self.trajectory = list()
        cur_time = vr_list[0].unixtime
        tmp_trj = [vr_list[0]]

        for tmp_vr in vr_list[1:]:
            if tmp_vr.unixtime - cur_time > GLB_TIME_THRES:
                if len(tmp_trj) > 1:
                    self.trajectory.append(tmp_trj)
                tmp_trj = [tmp_vr]
            else:
                tmp_trj.append(tmp_vr)
            cur_time = tmp_vr.unixtime

        if len(tmp_trj) > 1:
            self.trajectory.append(tmp_trj)
