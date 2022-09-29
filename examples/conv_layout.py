import argparse
import ios
import parser
import numpy as np
from ios.models.common import *
from ios.optimizer import *
from ios.ir import *
import itertools
import csv


def get_conv_key(node):
    assert(len(node.weight_shape) == 4)
    assert(len(node.stride) == 2)
    assert(len(node.padding) == 2)
    assert(node.weight_shape[1] == node.input_shape[0]) # in_c matches
    # key0 = str(node.input_shape[0]) + '_' + str(node.input_shape[1]) + '_' + str(node.input_shape[2]) + '_'
    # key1 = str(node.weight_shape[0]) + '_' + str(node.weight_shape[1]) + '_' + str(node.weight_shape[2]) + '_'  + str(node.weight_shape[3]) + '_'
    # key2 = str(node.stride[0]) + '_' + str(node.stride[1]) + '_'
    # key3 = str(node.padding[0]) + '_' + str(node.padding[1])
    param = [node.input_shape[0], node.input_shape[1], node.input_shape[2],  # 0, 1, 2
        node.weight_shape[0], node.weight_shape[1], node.weight_shape[2], node.weight_shape[3], # 3, 4, 5, 6
        node.stride[0], node.stride[1], node.padding[0], node.padding[1], node.act] # 7, 8, 9, 10, 11
    return param

def create_conv_graph_given_layout(conv_param: list, input_layout: str, output_layout: str, transform_conv:bool):
    if transform_conv:
        # default layout is NCHW
        v = ios.placeholder(output_shape=(conv_param[:3]), layout="NCHW")
        block = ios.Block(enter_node=v.node)
        ios.transform_conv2d(block, inputs=[[v]], out_channels=conv_param[0],
            kernel=(conv_param[5:7]), stride=(conv_param[7:9]), padding=(conv_param[9:11]),
            act=conv_param[11], conv_in_layout=input_layout, conv_out_layout=output_layout, is_exit=True)
    else:
        v = ios.placeholder(output_shape=(conv_param[:3]), layout=input_layout)
        block = ios.Block(enter_node=v.node)
        ios.conv2d(block, inputs=[[v]], out_channels=conv_param[0],
            kernel=(conv_param[5:7]), stride=(conv_param[7:9]), padding=(conv_param[9:11]),
            act=conv_param[11], layout=output_layout, is_exit=True)
    
    graph = ios.Graph(name="demo", input=v.node, blocks=[block])
    graph.init_weights()
    return graph

def conv_latency(conv_param: list, transform_conv: bool):
    layouts = ["NCHW", "NHWC"]
    # print(conv_param)
    latencies = []
    for input_layout, output_layout in itertools.product(layouts, layouts):
        graph = create_conv_graph_given_layout(conv_param, input_layout, output_layout, transform_conv)

        graph.sequential_schedule()
        seq_latency = ios.ios_runtime.graph_latency(graph, batch_size=1, repeat=10)
        latencies.append(np.mean(seq_latency))
        # print(graph)
        # print(f'Conv[{input_layout}->{output_layout}]: {np.mean(seq_latency):.3f} ms')
    return latencies

def main(model_name: str, transform_conv: bool):
    graph = getattr(ios.models, model_name)()

    conv_nodes = []
    # results = []
    for block in graph.blocks:
        for node in block.inner_nodes + [block.exit_node]:
            if isinstance(node, Conv) and node.groups == 1:
                conv_nodes.append(node)

    if transform_conv:
        output_file = f"data/transform_{model_name}.csv"
    else:
        output_file = f"data/{model_name}.csv"
    with open(output_file, "w", newline="\n") as f:
        csv_writer = csv.writer(f, delimiter=",")
        csv_writer.writerow(
            [
                "IN_C", "IN_H", "IN_W",
                "OUT_C", "IN_C", "KERNEL_H", "KERNEL_N",
                "STRIDE_H", "STRIDE_W", "PAD_H", "PAD_W",
                "ACT",
                "NCHW->NCHW", "NCHW->NHWC",
                "NHWC->NCHW", "NHWC->NHWC",
            ]
        )
        conv_keys = []
        for node in conv_nodes:
            conv_key = get_conv_key(node)
            if conv_key not in conv_keys:
                latency = conv_latency(conv_key, transform_conv)
                csv_writer.writerow([*conv_key, *latency])
                conv_keys.append(conv_key)

            # results.append(result)
    
    # print(results)

    # graph.sequential_schedule()
    # latency = ios.ios_runtime.graph_latency(graph, batch_size=1, repeat=6)

    # optimized_graph = ios.optimize(graph, batch_size=1)
    # optimized_latency = ios.ios_runtime.graph_latency(optimized_graph, batch_size=1, repeat=6)

    # os.makedirs("./outputs/", exist_ok=True)
    # ios.draw(optimized_graph, fname=f'optimized_{graph.name}.png', label=f'Optimized Graph, Latency = {np.mean(optimized_latency):.3f}')
    # print(f' Sequential schedule: {np.mean(latency):.3f} ms')
    # print(f'  Optimized schedule: {np.mean(optimized_latency):.3f} ms')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('collect conv latency with differnt layout from model')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help="name of model")
    parser.add_argument('-t', '--transform', action="store_true",
                        help="use transform_conv instead of conv")
    args = parser.parse_args()
    main(args.model, args.transform)
