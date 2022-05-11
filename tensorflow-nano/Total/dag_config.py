import numpy as np
import threading


PARTITION_INFOS={
    # "VGG-1": ('features1', 'features2', 'features3'),  # partition 1
    # "VGG-2": ('features4', 'features5'),               # partition 2
    # "VGG-3": ('classifier1', 'classifier2', 'classifier3'),  # partition 3
    
    # "AlexNet-1": ('features_1', 'features_2'),
    # "AlexNet-2": ('features_3', 'features_4', 'features_5'),
    # "AlexNet-3": ('classifier_1', 'classifier_2', 'classifier_3'),
    "AlexNet-in": ('input'),
    "AlexNet-1": ('features_1_1', 'features_2_1', 'features_3_1', 'features_4_1', 'features_5_1', 'classifier_1_1', 'classifier_2_1', 'classifier_3_1'),
    "AlexNet-2": ('features_1_2', 'features_2_2', 'features_3_2', 'features_4_2', 'features_5_2', 'classifier_1_2', 'classifier_2_2', 'classifier_3_2'),
    "AlexNet-out": ('output',),
    "VGG": ('features1', 'features2', 'features3', 'features4', 'features5', 'classifier1', 'classifier2', 'classifier3'),
    "NiN": ("features_1", 'features_2', 'features_3'),
    "ResNet-in": ('input',),
    "ResNet-CNN_1_2": ('cnn_1_2',), 
    "ResNet-CNN_2_1": ('cnn_2_1',), 
    "ResNet-CNN_3_2": ('cnn_3_2',), 
    "ResNet-CNN_4_1": ('cnn_4_1',), 
    "ResNet-CNN_5_2": ('cnn_5_2',), 
    "ResNet-CNN_6_1": ('cnn_6_1',), 
    "ResNet-CNN_7_2": ('cnn_7_2',),
    "ResNet-CNN_8_1": ('cnn_8_1',), 
    "ResNet-CNN_9_2": ('cnn_9_2',),
    "ResNet-CNN_10_1": ('cnn_10_1',), 
    "ResNet-CNN_11_2": ('cnn_11_2',),
    "ResNet-CNN_12_1": ('cnn_12_1',), 
    "ResNet-CNN_13_2": ('cnn_13_2',),
    "ResNet-CNN_14_1": ('cnn_14_1',), 
    "ResNet-CNN_15_2": ('cnn_15_2',), 
    "ResNet-CNN_16_1": ('cnn_16_1',), 
    "ResNet-CNN_17": ('cnn_17',),
}


PARTITION_IDX_MAP={
    # "VGG-1": 0,
    # "VGG-2": 1,
    # "VGG-3": 2,
    
    "AlexNet-in": 0,
    "AlexNet-1": 1,
    "AlexNet-2": 2,
    "AlexNet-out": 3,
    "VGG": 4,
    "NiN": 5,
    "ResNet-in": 6,
    "ResNet-CNN_1_2": 7, 
    "ResNet-CNN_2_1": 8, 
    "ResNet-CNN_3_2": 9, 
    "ResNet-CNN_4_1": 10, 
    "ResNet-CNN_5_2": 11, 
    "ResNet-CNN_6_1": 12, 
    "ResNet-CNN_7_2": 13,
    "ResNet-CNN_8_1": 14, 
    "ResNet-CNN_9_2": 15,
    "ResNet-CNN_10_1": 16, 
    "ResNet-CNN_11_2": 17,
    "ResNet-CNN_12_1": 18, 
    "ResNet-CNN_13_2": 19,
    "ResNet-CNN_14_1": 20, 
    "ResNet-CNN_15_2": 21, 
    "ResNet-CNN_16_1": 22, 
    "ResNet-CNN_17": 23,
}


REVERSE_IDX_MAP = {
    # "VGG-1": 0,
    # "VGG-2": 1,
    # "VGG-3": 2,
    
    0: "AlexNet-in",
    1: "AlexNet-1",
    2: "AlexNet-2",
    3: "AlexNet-out",
    4: "VGG",
    5: "NiN",
    6:"ResNet-in",
    7: "ResNet-CNN_1_2",
    8: "ResNet-CNN_2_1",
    9: "ResNet-CNN_3_2",
    10: "ResNet-CNN_4_1",
    11: "ResNet-CNN_5_2",
    12: "ResNet-CNN_6_1",
    13: "ResNet-CNN_7_2",
    14: "ResNet-CNN_8_1",
    15: "ResNet-CNN_9_2",
    16: "ResNet-CNN_10_1",
    17: "ResNet-CNN_11_2",
    18: "ResNet-CNN_12_1",
    19: "ResNet-CNN_13_2",
    20: "ResNet-CNN_14_1",
    21: "ResNet-CNN_15_2",
    22: "ResNet-CNN_16_1",
    23: "ResNet-CNN_17",
}

DAG_SUCCESSORS = {
    PARTITION_IDX_MAP['AlexNet-in']:(PARTITION_IDX_MAP['AlexNet-1'],PARTITION_IDX_MAP['AlexNet-2']),
    PARTITION_IDX_MAP['AlexNet-1']:(PARTITION_IDX_MAP['AlexNet-out'],),
    PARTITION_IDX_MAP['AlexNet-2']:(PARTITION_IDX_MAP['AlexNet-out'],),

    PARTITION_IDX_MAP['ResNet-in']:(PARTITION_IDX_MAP['ResNet-CNN_1_2'], PARTITION_IDX_MAP['ResNet-CNN_2_1']),
    PARTITION_IDX_MAP['ResNet-CNN_1_2']:(PARTITION_IDX_MAP['ResNet-CNN_2_1'], PARTITION_IDX_MAP['ResNet-CNN_3_2']),
    PARTITION_IDX_MAP['ResNet-CNN_2_1']:(PARTITION_IDX_MAP['ResNet-CNN_3_2'], PARTITION_IDX_MAP['ResNet-CNN_4_1'],),
    PARTITION_IDX_MAP['ResNet-CNN_3_2']:(PARTITION_IDX_MAP['ResNet-CNN_4_1'], PARTITION_IDX_MAP['ResNet-CNN_5_2'],),
    PARTITION_IDX_MAP['ResNet-CNN_4_1']:(PARTITION_IDX_MAP['ResNet-CNN_5_2'], PARTITION_IDX_MAP['ResNet-CNN_6_1'],),
    PARTITION_IDX_MAP['ResNet-CNN_5_2']:(PARTITION_IDX_MAP['ResNet-CNN_6_1'], PARTITION_IDX_MAP['ResNet-CNN_7_2'],),
    PARTITION_IDX_MAP['ResNet-CNN_6_1']:(PARTITION_IDX_MAP['ResNet-CNN_7_2'], PARTITION_IDX_MAP['ResNet-CNN_8_1'],),
    PARTITION_IDX_MAP['ResNet-CNN_7_2']:(PARTITION_IDX_MAP['ResNet-CNN_8_1'], PARTITION_IDX_MAP['ResNet-CNN_9_2'],),
    PARTITION_IDX_MAP['ResNet-CNN_8_1']:(PARTITION_IDX_MAP['ResNet-CNN_9_2'], PARTITION_IDX_MAP['ResNet-CNN_10_1'],),
    PARTITION_IDX_MAP['ResNet-CNN_9_2']:(PARTITION_IDX_MAP['ResNet-CNN_10_1'], PARTITION_IDX_MAP['ResNet-CNN_11_2'],),
    PARTITION_IDX_MAP['ResNet-CNN_10_1']:(PARTITION_IDX_MAP['ResNet-CNN_11_2'], PARTITION_IDX_MAP['ResNet-CNN_12_1'],),
    PARTITION_IDX_MAP['ResNet-CNN_11_2']:(PARTITION_IDX_MAP['ResNet-CNN_12_1'], PARTITION_IDX_MAP['ResNet-CNN_13_2'],),
    PARTITION_IDX_MAP['ResNet-CNN_12_1']:(PARTITION_IDX_MAP['ResNet-CNN_13_2'], PARTITION_IDX_MAP['ResNet-CNN_14_1'],),
    PARTITION_IDX_MAP['ResNet-CNN_13_2']:(PARTITION_IDX_MAP['ResNet-CNN_14_1'], PARTITION_IDX_MAP['ResNet-CNN_15_2'],),
    PARTITION_IDX_MAP['ResNet-CNN_14_1']:(PARTITION_IDX_MAP['ResNet-CNN_15_2'], PARTITION_IDX_MAP['ResNet-CNN_16_1'],),
    PARTITION_IDX_MAP['ResNet-CNN_15_2']:(PARTITION_IDX_MAP['ResNet-CNN_16_1'], PARTITION_IDX_MAP['ResNet-CNN_17'],),
    PARTITION_IDX_MAP['ResNet-CNN_16_1']:(PARTITION_IDX_MAP['ResNet-CNN_17'],),
}


MODEL_IDX = {
    'alexnet':0,
    'vggnet':1,
    'nin':2,
    'resnet':3
}

MODEL_START_PARTITION = {
    'alexnet':0,
    'vggnet':4,
    'nin':5,
    'resnet':6
}

MODEL_END_PARTITION = {
    'alexnet':3,
    'vggnet':4,
    'nin':5,
    'resnet':23
}


class DAGManager:
    def __init__(self):
        self.dag_infos = dict()
        self.dag_input_order = dict()
        self.input_num_infos = dict()
        self.partition_lock = list()
        self.recv_data_dict = dict()

        pred_order = dict()
        for partition_name in PARTITION_INFOS.keys():
            part_idx = PARTITION_IDX_MAP[partition_name]
            self.partition_lock.append(threading.Lock())
            if part_idx in DAG_SUCCESSORS:
                for succ in DAG_SUCCESSORS[part_idx]:
                    # self.dag_input_order[part_idx][DAG_SUCCESSORS[part_idx][i]] = i
                    if succ not in pred_order:
                        pred_order[succ] = list()
                    pred_order[succ].append(part_idx)
                    
                    if succ not in self.input_num_infos:
                        self.input_num_infos[succ] = 0
                    self.input_num_infos[succ] += 1
            self.recv_data_dict[part_idx] = dict()
        
        for succ in pred_order:
            pred_order[succ].sort()
            num_inputs = len(pred_order[succ])
            if num_inputs > 1:
                self.dag_input_order[succ] = dict()
                for pred_idx in range(num_inputs):
                    pred = pred_order[succ][pred_idx]
                    self.dag_input_order[succ][pred] = pred_idx

        for start_part in MODEL_START_PARTITION.values():
            if start_part in self.input_num_infos:
                self.input_num_infos[start_part] += 1
            else:
                self.input_num_infos[start_part] = 1

        

    def recv_data(self, inputs, tr_start=0., tr_end=0.):
        req_id = inputs[0]
        source_partition = inputs[1]
        target_partition = inputs[2]
        data = inputs[3]
        pure_tr_time = tr_end - tr_start

        if self.input_num_infos[target_partition] == 1:
            return inputs, pure_tr_time, pure_tr_time
        else:
            with self.partition_lock[target_partition]:   # target partition info updat
                if req_id not in self.recv_data_dict[part_idx]:
                    # self.recv_data_dict[req_id] = [1, np.zeros_like(self.partition_input_sample[target_partition]), tr_start]
                    self.recv_data_dict[part_idx][req_id] = [1, [None for _ in range(self.input_num_infos[target_partition])], tr_start]
                else:
                    self.recv_data_dict[part_idx][req_id][0] += 1
                    
                self.recv_data_dict[part_idx][req_id][1][self.dag_input_order[target_partition][source_partition]] = data
            
                if self.recv_data_dict[part_idx][req_id][0] == self.input_num_infos[target_partition]:
                    result = (req_id, -1, target_partition, self.recv_data_dict[part_idx][req_id][1])
                    tr_time = tr_end - self.recv_data_dict[part_idx][req_id][2]
                    del self.recv_data_dict[part_idx][req_id]
                    return result, tr_time, pure_tr_time
                else:
                    return None

if __name__=="__main__":
    test = DAGManager()
    print(test)