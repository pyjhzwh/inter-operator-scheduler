import numpy as np
import ios
from ios.utils import get_conv_key, conv_latency

def get_conv_latency_from_graph(graph: ios.Graph):
    conv_nodes = []
    for block in graph.blocks:
        for node in block.inner_nodes + [block.exit_node]:
            if isinstance(node, ios.ir.Conv) and node.groups == 1:
                conv_nodes.append(node)

    for node in conv_nodes:
        conv_key = get_conv_key(node)
        latency = conv_latency(conv_key)
        print(f"conv_key: {conv_key}, latency: {latency[0]}")
            

def main():
    graph0 = ios.models.vgg_11()

    graph0.sequential_schedule()
    print(graph0)
    latency0, stage_latency0 = ios.ios_runtime.graph_latency(graph0, batch_size=1, repeat=6, profile_stage=True)


    # graph1 = ios.models.vgg_11_opt_layout()

    # graph1.sequential_schedule()
    # print(graph1)
    # latency1, stage_latency1 = ios.ios_runtime.graph_latency(graph1, batch_size=1, repeat=6, profile_stage=True)

    print(f'original vgg_11 Sequential schedule: {np.mean(latency0):.3f} ms')
    print(f'original vgg_11 Stage latency: {np.mean(np.array(stage_latency0).reshape(6, -1), axis=0)}\n')
    # print(f'opt_layout vgg_11 Sequential schedule: {np.mean(latency1):.3f} ms')
    # print(f'opt_layout vgg_11 Stage latency: {np.mean(np.array(stage_latency1).reshape(6, -1), axis=0)}\n')

    get_conv_latency_from_graph(graph0)



if __name__ == '__main__':
    main()