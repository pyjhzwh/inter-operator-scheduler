import argparse
import ios
import parser
import numpy as np
from ios.models.common import *
from ios.optimizer import *
from ios.ir import *
import csv


def get_pool_key(node: Pool):
    assert(len(node.output_shape) == 3)
    assert(len(node.input_shape) == 3)
    param = [
        node.input_shape[0], node.input_shape[1], node.input_shape[2], node.pool_type,
        node.kernel[0], node.kernel[1], node.stride[0], node.stride[1],
        node.padding[0], node.padding[1],
        node.output_shape[0], node.output_shape[1], node.output_shape[2],
    ]
    return param

def create_pool_graph_given_layout(param: list, layout: str):
    v = ios.placeholder(output_shape=(param[:3]), layout=layout)
    block = ios.Block(enter_node=v.node)
    ios.pool2d(
        block, inputs=[[v]], pool_type=param[3], kernel=param[4:6],
        stride=param[6:8], padding=param[8:10], is_exit=True
    )
    graph = ios.Graph(name="demo", input=v.node, blocks=[block])
    graph.init_weights()
    return graph


def pool_latency(param: list):
    layouts = ["NCHW", "NHWC"]
    latencies = []
    for layout in layouts:
        graph = create_pool_graph_given_layout(param, layout)
        graph.sequential_schedule()
        seq_latency = ios.ios_runtime.graph_latency(graph, batch_size=1, repeat=10)
        latencies.append(np.mean(seq_latency))

    return latencies

def main(model_name: str):
    graph = getattr(ios.models, model_name)()

    pool_nodes = []
    # results = []
    for block in graph.blocks:
        for node in block.inner_nodes + [block.exit_node]:
            if isinstance(node, Pool):
                pool_nodes.append(node)

    output_file = f"data/pool_{model_name}.csv"
    with open(output_file, "w", newline="\n") as f:
        csv_writer = csv.writer(f, delimiter=",")
        csv_writer.writerow(
            [
                "IN_C", "IN_H", "IN_W",
                "POOL_TYPE",
                "KERNEL_H", "KERNEL_W", "STRIDE_H", "STRIDE_W",
                "PAD_H", "PAD_W",
                "NCHW", "NHWC",
            ]
        )
        pool_keys = []
        for node in pool_nodes:
            pool_key = get_pool_key(node)
            if pool_key not in pool_keys:
                latency = pool_latency(pool_key)
                csv_writer.writerow([*pool_key, *latency])
                pool_keys.append(pool_key)



if __name__ == '__main__':
    parser = argparse.ArgumentParser('collect conv latency with differnt layout from model')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help="name of model")
    args = parser.parse_args()
    main(args.model)
