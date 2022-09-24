import numpy as np
import ios

def transform_network():
    v = ios.placeholder(output_shape=(64, 224, 224), layout="NHWC")
    block = ios.Block(enter_node=v.node)
    # v1 = ios.transform(block, inputs=[[v]], dst_layout="NHWC")
    # v2 = ios.transform(block, inputs=[[v1]], dst_layout="NHWC")
    v3 = ios.conv2d(block, inputs=[[v]], out_channels=96, kernel=(1, 1), stride=(1, 1), padding=(1, 1), act='relu', layout="NHWC", is_exit=True)
    graph = ios.Graph(name="demo", input=v.node, blocks=[block])
    graph.init_weights()
    return graph

# define computation graph
graph = transform_network()

# measure latency
graph.sequential_schedule()
seq_latency, stage_latency = ios.ios_runtime.graph_latency(graph, batch_size=1, repeat=6, profile_stage=True)
print(graph)
print(f'Sequential schedule: {np.mean(seq_latency):.3f} ms')
print(f'      Stage latency: {np.mean(np.array(stage_latency).reshape(6, -1), axis=0)}\n')
