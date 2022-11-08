import argparse
import ios
import numpy as np
from ios.models.common import *
from ios.optimizer import *
from ios.ir import *
import csv


def get_transform_key(node: Transform):
    assert(len(node.weight_shape) == 4)
    assert(len(node.stride) == 2)
    assert(len(node.padding) == 2)
    assert(node.weight_shape[1] == node.input_shape[0]) # in_c matches
    param = [node.output_shape[0], node.output_shape[1], node.output_shape[2]]
    return param

def create_transform_graph_given_layout(transform_param: list, input_layout: str, output_layout: str):
    # default layout is NCHW
    v = ios.placeholder(output_shape=(transform_param[:3]), layout=input_layout)
    block = ios.Block(enter_node=v.node)
    ios.transform(block, inputs=[[v]], dst_layout=output_layout, is_exit=True)
    
    graph = ios.Graph(name="demo", input=v.node, blocks=[block])
    graph.init_weights()
    return graph

def transform_latency(transform_param: list):
    layouts = [["NCHW", "NHWC"], ["NHWC", "NCHW"]]
    # print(transform_param)
    latencies = []
    for input_layout, output_layout in layouts:
        graph = create_transform_graph_given_layout(transform_param, input_layout, output_layout)

        graph.sequential_schedule()
        seq_latency = ios.ios_runtime.graph_latency(graph, batch_size=1, repeat=10)
        latencies.append(np.mean(seq_latency))

    return latencies

def main(model_name: str):
    graph = getattr(ios.models, model_name)()

    transform_nodes = []
    # results = []
    for block in graph.blocks:
        for node in block.inner_nodes + [block.exit_node]:
            if isinstance(node, Conv) and node.groups == 1:
                transform_nodes.append(node)

    output_file = f"data/transform_{model_name}.csv"
    with open(output_file, "w", newline="\n") as f:
        csv_writer = csv.writer(f, delimiter=",")
        csv_writer.writerow(
            [
                "C", "H", "W",
                "NCHW->NHWC", "NHWC->NCHW",
            ]
        )
        transform_keys = []
        for node in transform_nodes:
            transform_key = get_transform_key(node)
            if transform_key not in transform_keys:
                latency = transform_latency(transform_key)
                csv_writer.writerow([*transform_key, *latency])
                transform_keys.append(transform_key)

            # results.append(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('collect transform latency with differnt layout from model')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help="name of model")
    args = parser.parse_args()
    main(args.model)
