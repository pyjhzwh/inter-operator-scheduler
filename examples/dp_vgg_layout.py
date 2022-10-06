import ios
import numpy as np
from ios.utils import get_conv_key, conv_latency, get_transform_key, transform_latency
import argparse
import parser

def get_convandtransform_latency_from_graph(graph: ios.Graph):
    conv_latencies = []
    transform_latencies = []
    conv_nodes = []
    for block in graph.blocks:
        for node in block.inner_nodes + [block.exit_node]:
            if isinstance(node, ios.ir.Conv) and node.groups == 1:
                conv_nodes.append(node)

    for node in conv_nodes:
        conv_key = get_conv_key(node)
        c_latency = conv_latency(conv_key)
        transform_key = get_transform_key(node)
        t_latency = transform_latency(transform_key)

        conv_latencies.append(c_latency)
        transform_latencies.append(t_latency)

    return conv_latencies, transform_latencies


def dp_best_layout(conv_latencies, transform_latencies):
    dp = conv_latencies[0]
    next_dp = [[]] * 4
    local_layout = [[i] for i in range(4)]

    for i in range(1, len(conv_latencies)):
        next_local_layout = [[]] * 4
        for j in range(4):
            curmin = 100.0
            minidx = -1
            for k in range(4):
                if k % 2 == 0 and i % 2 == 1: # NCHW->NHWC
                    cost = transform_latencies[i][0]
                elif k % 2 == 1 and i % 2 == 0: # NHWC->NCHW
                    cost = transform_latencies[i][1]
                else:
                    cost = 0
                cur = dp[k] + conv_latencies[i][k] + cost
                if cur < curmin:
                    curmin = cur
                    minidx = k
            next_dp[j] = curmin
            # append the path for best layout locally
            next_local_layout[j] = [*local_layout[minidx], j]
        dp = next_dp
        local_layout = next_local_layout

    curmin = 100.0
    minidx = -1

    for i in range(len(dp)):
        if dp[i] < curmin:
            curmin = dp[i]
            minidx = i
    return local_layout[minidx]


id2layout = {0: ["NCHW", "NCHW"], 1: ["NCHW", "NHWC"], 2: ["NHWC", "NCHW"], 3: ["NHWC", "NHWC"]}
model2char = {"vgg_11": "A", "vgg_13": "B", "vgg_16": "D", "vgg_19": "E"}
cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def create_vgg_given_layout(model_name:str, best_layout):

    layout = [id2layout[l] for l in best_layout]
    return ios.models.vgg_net_opt_layout(cfgs[model2char[model_name]], layout, model_name)


def main(model_name: str):
    graph0 = getattr(ios.models, model_name)()

    conv_latencies, transform_latencies = get_convandtransform_latency_from_graph(graph0)
    best_layouts = dp_best_layout(conv_latencies, transform_latencies)
    graph1 = create_vgg_given_layout(model_name, best_layouts)

    graph0.sequential_schedule()
    latency0, stage_latency0 = ios.ios_runtime.graph_latency(graph0, batch_size=1, warmup=10, repeat=10, profile_stage=True)

    print(f'original {model_name} Sequential schedule: {np.mean(latency0):.3f} ms')
    print(f'original {model_name} Stage latency: {np.mean(np.array(stage_latency0).reshape(10, -1), axis=0)}\n')

    graph1.sequential_schedule()
    print(graph1)
    latency1, stage_latency1 = ios.ios_runtime.graph_latency(graph1, batch_size=1, warmup=10, repeat=10, profile_stage=True)


    print(f'opt {model_name} Sequential schedule: {np.mean(latency1):.3f} ms')
    print(f'opt {model_name} Stage latency: {np.mean(np.array(stage_latency1).reshape(10, -1), axis=0)}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('use dp to get the best layout for vgg model')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help="name of model")
    args = parser.parse_args()
    if "vgg" not in args.model:
        raise ValueError("{args.model} is not supported, onl/13/16/19 are supported")
    main(args.model)