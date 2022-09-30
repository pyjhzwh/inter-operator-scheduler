import argparse
import ios
import parser
import numpy as np
from ios.models.common import *
from ios.optimizer import *
from ios.ir import *
import csv


def get_elem_key(node: Element):
    assert(len(node.output_shape) == 3)
    param = [
        node.input_shape[0], node.input_shape[1], node.input_shape[2], node.op_type,
    ]
    return param

def get_act_key(node):
    assert(len(node.output_shape) == 3)
    if isinstance(node, Activation):
        param = [
            node.input_shape[0], node.input_shape[1], node.input_shape[2], node.act_type,
        ]
    else:
        assert isinstance(node, Relu)
        param = [
            node.input_shape[0], node.input_shape[1], node.input_shape[2], "relu",
        ]
    return param


def create_elem_graph_given_layout(param: list, layout: str):
    v = ios.placeholder(output_shape=(param[:3]), layout=layout)
    # v2 = ios.placeholder(output_shape=(param[:3]), layout=layout)
    block = ios.Block(enter_node=v.node)
    ios.element(block, inputs=[[v, v]], op_type=param[3], layout=layout, is_exit=True)
    graph = ios.Graph(name="demo", input=v.node, blocks=[block])
    graph.init_weights()
    return graph

def create_act_graph_given_layout(param: list, layout: str):
    v = ios.placeholder(output_shape=(param[:3]), layout=layout)
    block = ios.Block(enter_node=v.node)
    op_type = param[3]
    if op_type == "relu":
        ios.relu(block, inputs=[[v]], is_exit=True, layout=layout)
    else:
        ios.activation(block, inputs=[[v]], act_type=param[3], inplace=True, is_exit=True, layout=layout)
    graph = ios.Graph(name="demo", input=v.node, blocks=[block])
    graph.init_weights()
    return graph


def elem_latency(param: list):
    layouts = ["NCHW", "NHWC"]
    latencies = []
    for layout in layouts:
        graph = create_elem_graph_given_layout(param, layout)
        graph.sequential_schedule()
        seq_latency = ios.ios_runtime.graph_latency(graph, batch_size=1, repeat=10)
        latencies.append(np.mean(seq_latency))

    return latencies

def act_latency(param: list):
    layouts = ["NCHW", "NHWC"]
    latencies = []
    for layout in layouts:
        graph = create_act_graph_given_layout(param, layout)
        graph.sequential_schedule()
        seq_latency = ios.ios_runtime.graph_latency(graph, batch_size=1, repeat=10)
        latencies.append(np.mean(seq_latency))

    return latencies


def main(model_name: str):
    graph = getattr(ios.models, model_name)()

    elem_nodes = []
    act_nodes = []
    # results = []
    for block in graph.blocks:
        for node in block.inner_nodes + [block.exit_node]:
            if isinstance(node, Element):
                elem_nodes.append(node)
            elif isinstance(node, Activation) or isinstance(node, Relu):
                act_nodes.append(node)

    output_file = f"data/elem_act_{model_name}.csv"
    with open(output_file, "w", newline="\n") as f:
        csv_writer = csv.writer(f, delimiter=",")
        csv_writer.writerow(
            [
                "IN_C", "IN_H", "IN_W",
                "OP_TYPE",
                "NCHW", "NHWC",
            ]
        )
        elem_keys = []
        act_keys = []
        for node in elem_nodes:
            elem_key = get_elem_key(node)
            print("elem_key", elem_key)
            if elem_key not in elem_keys:
                latency = elem_latency(elem_key)
                csv_writer.writerow([*elem_key, *latency])
                elem_keys.append(elem_key)

        for node in act_nodes:
            act_key = get_act_key(node)
            print("act_key", act_key)
            if act_key not in act_keys:
                latency = act_latency(act_key)
                csv_writer.writerow([*act_key, *latency])
                act_keys.append(act_key)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('collect conv latency with differnt layout from model')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help="name of model")
    args = parser.parse_args()
    main(args.model)
