import numpy as np
import threading
import copy


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

    "VGG-1": ('features1', 'features2', ),
    "VGG-2": ('features3', 'features4', ),
    "VGG-3": ('features5', 'classifier1', 'classifier2', 'classifier3',),

    "NiN-1": ("features_1",'features_2',),
    "NiN-2": ('features_3', "feature_4",),

    "ResNet-CNN_1-10": (
        'input', 'cnn_1_2', 'cnn_2_1', 'cnn_3_2', 'cnn_4_1', 'cnn_5_2',
        'cnn_6_1', 'cnn_7_2','cnn_8_1','cnn_9_2', 'cnn_10_1',
        ),
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

    "VGG-1": 4,
    "VGG-2": 5,
    "VGG-3": 6,

    "NiN-1": 7,
    "NiN-2": 8,

    "ResNet-CNN_1-10": 9,
    "ResNet-CNN_11_2": 10,
    "ResNet-CNN_12_1": 11,
    "ResNet-CNN_13_2": 12,
    "ResNet-CNN_14_1": 13,
    "ResNet-CNN_15_2": 14,
    "ResNet-CNN_16_1": 15,
    "ResNet-CNN_17": 16,
}


REVERSE_IDX_MAP = {
    # "VGG-1": 0,
    # "VGG-2": 1,
    # "VGG-3": 2,
    
    0: "AlexNet-in",
    1: "AlexNet-1",
    2: "AlexNet-2",
    3: "AlexNet-out",
    
    4: "VGG-1",
    5: "VGG-2",
    6: "VGG-3",
    
    7: "NiN-1",
    8: "NiN-2",
    
    9: "ResNet-CNN_1-10",
    10: "ResNet-CNN_11_2",
    11: "ResNet-CNN_12_1",
    12: "ResNet-CNN_13_2",
    13: "ResNet-CNN_14_1",
    14: "ResNet-CNN_15_2",
    15: "ResNet-CNN_16_1",
    16: "ResNet-CNN_17",
}

DAG_SUCCESSORS = {
    PARTITION_IDX_MAP['AlexNet-in']:(PARTITION_IDX_MAP['AlexNet-1'],PARTITION_IDX_MAP['AlexNet-2']),
    PARTITION_IDX_MAP['AlexNet-1']:(PARTITION_IDX_MAP['AlexNet-out'],),
    PARTITION_IDX_MAP['AlexNet-2']:(PARTITION_IDX_MAP['AlexNet-out'],),

    PARTITION_IDX_MAP['VGG-1']:(PARTITION_IDX_MAP['VGG-2'],),
    PARTITION_IDX_MAP['VGG-2']:(PARTITION_IDX_MAP['VGG-3'],),

    PARTITION_IDX_MAP['NiN-1']:(PARTITION_IDX_MAP['NiN-2'],),

    PARTITION_IDX_MAP['ResNet-CNN_1-10']:(PARTITION_IDX_MAP['ResNet-CNN_1-10'], PARTITION_IDX_MAP['ResNet-CNN_11_2']),
    PARTITION_IDX_MAP['ResNet-CNN_1-10']:(PARTITION_IDX_MAP['ResNet-CNN_11_2'], PARTITION_IDX_MAP['ResNet-CNN_12_1'],),
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
    'nin':7,
    'resnet':9
}

MODEL_END_PARTITION = {
    'alexnet':3,
    'vggnet':6,
    'nin':8,
    'resnet':16
}

MODEL_OUTPUT_MAP = {  # select output tuple  [source][target][data_tuple_idx] = (output_val_position, input position)
    PARTITION_IDX_MAP['AlexNet-in']: {
        PARTITION_IDX_MAP['AlexNet-1']: ((0, 0),),
        PARTITION_IDX_MAP['AlexNet-2']: ((0, 0),),
        },
    
    PARTITION_IDX_MAP['AlexNet-1']:{PARTITION_IDX_MAP['AlexNet-out']:((0,0),)},
    PARTITION_IDX_MAP['AlexNet-2']:{PARTITION_IDX_MAP['AlexNet-out']:((0,1),)},

    PARTITION_IDX_MAP['ResNet-CNN_1-10']: {
        PARTITION_IDX_MAP["ResNet-CNN_11_2"]: ((0, 1), (2, 0)),
        PARTITION_IDX_MAP["ResNet-CNN_12_1"]: ((0, 0),),
    }, 

    PARTITION_IDX_MAP['ResNet-CNN_11_2']: {
        PARTITION_IDX_MAP["ResNet-CNN_12_1"]: ((0, 1),),
        PARTITION_IDX_MAP["ResNet-CNN_13_2"]: ((0, 0),),
    }, 
    
    PARTITION_IDX_MAP['ResNet-CNN_12_1']: {
        PARTITION_IDX_MAP["ResNet-CNN_13_2"]: ((0, 1),),
        PARTITION_IDX_MAP["ResNet-CNN_14_1"]: ((0, 0),),
        },
    PARTITION_IDX_MAP['ResNet-CNN_13_2']: {
        PARTITION_IDX_MAP["ResNet-CNN_14_1"]: ((0, 1),),
        PARTITION_IDX_MAP['ResNet-CNN_15_2']: ((0, 0),),
        },
    PARTITION_IDX_MAP['ResNet-CNN_14_1']: {
        PARTITION_IDX_MAP['ResNet-CNN_15_2']: ((0, 1),),
        PARTITION_IDX_MAP['ResNet-CNN_16_1']: ((0, 0),),
        },
    PARTITION_IDX_MAP['ResNet-CNN_15_2']: {
        PARTITION_IDX_MAP['ResNet-CNN_16_1']: ((0, 1),),
         PARTITION_IDX_MAP['ResNet-CNN_17']: ((0, 0),),
        },
    PARTITION_IDX_MAP['ResNet-CNN_16_1']: {
        PARTITION_IDX_MAP['ResNet-CNN_17']: ((0, 1),),
        },
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
                if req_id not in self.recv_data_dict[target_partition]:
                    # self.recv_data_dict[req_id] = [1, np.zeros_like(self.partition_input_sample[target_partition]), tr_start]
                    self.recv_data_dict[target_partition][req_id] = [1, [None for _ in range(self.input_num_infos[target_partition])], tr_start]
                else:
                    self.recv_data_dict[target_partition][req_id][0] += 1
                if type(data) in (tuple, list):
                    for data_idx in range(len(data)):
                        _, tuple_idx = MODEL_OUTPUT_MAP[source_partition][target_partition][data_idx]
                        self.recv_data_dict[target_partition][req_id][1][tuple_idx] = data[data_idx]
                else:
                    self.recv_data_dict[target_partition][req_id][1][self.dag_input_order[target_partition][source_partition]] = data
            
                if self.recv_data_dict[target_partition][req_id][0] == self.input_num_infos[target_partition]:
                    result = (req_id, -1, target_partition, self.recv_data_dict[target_partition][req_id][1])
                    tr_time = tr_end - self.recv_data_dict[target_partition][req_id][2]
                    del self.recv_data_dict[target_partition][req_id]
                    return result, tr_time, pure_tr_time
                else:
                    return None
    
    def send_data(self, req_id, source, outputs):  # todo!!!
        targets = DAG_SUCCESSORS[source]
               
        if len(targets) == 1:  # for fast operation            
            new_output = self.convert_output(outputs, source, targets[0])
            return ((req_id, source, targets[0], new_output),)
        else:
            result = list()
            first = True
            for target in targets:
                new_output = self.convert_output(outputs, source, target, first)
                new_output = (req_id, source, target, new_output)
                if first:
                    first = False
                result.append(new_output)                
            return result
        # if type outputs()

    def convert_output(self, outputs, source, target, first=True):
        if first:
            outputs = outputs
        else:
            outputs = copy.deepcopy(outputs)

        if type(outputs) in (tuple, list): # multiple outputs
            result = list()
            for data_idx, _  in MODEL_OUTPUT_MAP[source][target]:
                result.append(outputs[data_idx])
            return result
        else:
            return outputs


if __name__=="__main__":
    test = DAGManager()
    print(test)