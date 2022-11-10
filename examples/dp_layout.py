import ios
import numpy as np
from ios.utils import get_conv_key, conv_latency, get_transform_key, transform_latency
import argparse

def print_list_of_list(lists, name=""):
    print(name)
    for line in lists:
        print(", ".join("%.4f" % f for f in line))

def get_convandtransform_latency_from_graph(graph: ios.Graph):
    conv_latencies = []
    transform_latencies = []
    conv_nodes = []
    # id2node = []
    node2id = {}
    dep_all_nodes = {} # map[node] = list of conv node dependencies with propogation

    id = 0
    for block in graph.blocks:
        for node in block.inner_nodes + [block.exit_node]:
            if isinstance(node, ios.ir.Conv):
                node2id[node] = id
                # id2node.append(node)
                conv_nodes.append(node)
                id += 1

            for input in node.inputs[0]:
                if isinstance(input.node, ios.ir.Conv):
                    dep_all_nodes.setdefault(node, []).append(input.node)
                else:
                    # propogate dep
                    if input.node in dep_all_nodes:
                        dep_all_nodes.setdefault(node, []).append(dep_all_nodes[input.node][0])

    # dep of node layout !! (Note it is not the dep of node)
    # c = Conv(a + b); where a and b are conv, dep[c] = a only; because a+b layout use a's layout
    dep = [None] * len(conv_nodes)
    # reverse dep of node layout !! (Note it is not the dep of node)
    reverse_dep = [None] * len(conv_nodes)
    for conv_node in conv_nodes:
        if conv_node in dep_all_nodes.keys():
            cur_deps = dep_all_nodes[conv_node]
            dep[node2id[conv_node]] = []
            for cur_dep in cur_deps:
                dep[node2id[conv_node]].append(node2id[cur_dep])
                if reverse_dep[node2id[cur_dep]] is None:
                    reverse_dep[node2id[cur_dep]] = [node2id[conv_node]]
                else:
                    reverse_dep[node2id[cur_dep]].append(node2id[conv_node])
        else:
            dep[node2id[conv_node]] = [100]

    # print("dep", dep)
    # print("reverse_dep", reverse_dep)

    for node in conv_nodes:
        conv_key = get_conv_key(node)
        c_latency = conv_latency(conv_key)
        transform_key = get_transform_key(node)
        t_latency = transform_latency(transform_key)

        conv_latencies.append(c_latency)
        transform_latencies.append(t_latency)

    # print_list_of_list(conv_latencies, "conv_latencies")
    # print_list_of_list(transform_latencies, "transform_latencies")
    return conv_latencies, transform_latencies, dep


def dp_best_layout(conv_latencies, transform_latencies, dep):
    copy_dep = dep.copy()
    dp = [[0]*4] * len(conv_latencies)
    dp = []
    local_layout = []
    for i in range(len(conv_latencies)):
        tmp = []
        for j in range(4):
            tmp.append(-1)
        dp.append(tmp)
        local_layout.append(tmp.copy())


    for i in range(len(conv_latencies)):
        # cur_node depend on 
        cur_node_dep = copy_dep[i][0]
        
        # for conv that do not dep on other convs
        if cur_node_dep == 100:
            for j in range(4):
                dp[i][j] = conv_latencies[i][j]
                local_layout[i][j] = 100

        else:
            for j in range(4):
                curmin = 100.0
                minidx = -1
                for k in range(4):
                    if k % 2 == 0 and j >= 2: # NCHW->NHWC 00, 10 -> 10, 11
                        cost = transform_latencies[cur_node_dep][0]
                    elif k % 2 == 1 and j <= 1: # NHWC->NCHW 01, 11 -> 00, 01
                        cost = transform_latencies[cur_node_dep][1]
                    else:
                        cost = 0
                    cur = dp[cur_node_dep][k] + conv_latencies[i][j] + cost
                    if cur < curmin:
                        curmin = cur
                        minidx = k
                dp[i][j] = curmin
                # append the path for best layout locally (prev layer's minidx)
                local_layout[i][j] = minidx


    curmin = 100.0
    minidx = -1

    best_layout_list = []
    for i in range(len(dp)):
        best_layout_list.append(None)
    for j in range(len(dp[-1])):
        if dp[-1][j] < curmin:
            curmin = dp[-1][j]
            minidx = j
    best_layout_list[-1] = minidx
    for i in range(len(dp)-1, 0, -1):
        best_layout_list[i-1] = local_layout[i][minidx]
        minidx = local_layout[i][minidx]
    # print("local_layout", local_layout)
    # print("best_layout_list", best_layout_list)
    return best_layout_list


id2layout = {0: ["NCHW", "NCHW"], 1: ["NCHW", "NHWC"], 2: ["NHWC", "NCHW"], 3: ["NHWC", "NHWC"]}
model2char = {"vgg_11": "A", "vgg_13": "B", "vgg_16": "D", "vgg_19": "E"}
cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def create_model_given_layout(model_name:str, best_layout):
    layout = [id2layout[l] for l in best_layout]

    if "vgg" in model_name:
        return ios.models.vgg_net_opt_layout(cfgs[model2char[model_name]], layout, model_name)
    elif model_name == "resnet18":
        return ios.models.resnet18_opt_layout(layout)
    elif model_name == "resnet34":
        return ios.models.resnet34_opt_layout(layout)
    elif model_name == "resnet50":
        return ios.models.resnet50_opt_layout(layout)


def main(model_name: str):
    batch = 1
    warmup = 100
    repeat = 1000

    graph0 = getattr(ios.models, model_name)()
    # graph0 = create_model_given_layout(model_name, [3]*28)
    
    conv_latencies, transform_latencies, dep = get_convandtransform_latency_from_graph(graph0)
    best_layouts = dp_best_layout(conv_latencies, transform_latencies, dep)
    print(len(best_layouts), best_layouts)
    graph1 = create_model_given_layout(model_name, best_layouts)

    graph0.sequential_schedule()
    print(graph0)
    # optimized_graph0 = ios.optimize(graph0, batch_size=1, opt_type='dp_merge_parallel', compute_weight=True)
    # print(optimized_graph0)
    latency0, stage_latency0 = ios.ios_runtime.graph_latency(graph0, batch_size=batch, warmup=warmup, repeat=repeat, profile_stage=True)

    print(f'original {model_name} Sequential schedule: {np.mean(latency0):.3f} ms')
    print(f'original {model_name} Stage latency: {np.mean(np.array(stage_latency0).reshape(repeat, -1), axis=0)}\n')

    graph1.sequential_schedule()
    print(graph1)
    # optimized_graph1 = ios.optimize(graph1, batch_size=1, opt_type='dp_merge_parallel', compute_weight=True)
    # print(optimized_graph1)
    latency1, stage_latency1 = ios.ios_runtime.graph_latency(graph1, batch_size=batch, warmup=warmup, repeat=repeat, profile_stage=True)


    print(f'opt {model_name} Sequential schedule: {np.mean(latency1):.3f} ms')
    print(f'opt {model_name} Stage latency: {np.mean(np.array(stage_latency1).reshape(repeat, -1), axis=0)}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('use dp to get the best layout for vgg model')
    parser.add_argument('-m', '--model', type=str, default="resnet18",
                        help="name of model")
    args = parser.parse_args()
    if "vgg" not in args.model and "resnet" not in args.model:
        raise ValueError("{args.model} is not supported, only vgg/13/16/19 and resnet18/34 are supported")
    main(args.model)