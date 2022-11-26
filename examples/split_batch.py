import numpy as np
import ios
from ios import reset_name

def sample_network():
    v = ios.placeholder(output_shape=(512, 32, 32), layout="NCHW")
    block = ios.Block(enter_node=v.node)
    v_split0 = ios.split_batch(block, inputs=[[v]], batch_begin=0, batch_end=1)
    v_split1 = ios.split_batch(block, inputs=[[v]], batch_begin=1, batch_end=5)
    # v0 = ios.conv2d(block, inputs=[[v_split0]], out_channels=512, kernel=(3, 3), stride=(1, 1), padding=(1, 1), act='relu', layout="NHWC")
    # v1 = ios.conv2d(block, inputs=[[v_split1]], out_channels=512, kernel=(3, 3), stride=(1, 1), padding=(1, 1), act='relu', layout="NHWC")
    out = ios.identity(block, inputs=[[v_split0], [v_split1]], is_exit=True)  # concat v1, v2, and v3
    graph = ios.Graph(name="demo", input=v.node, blocks=[block])
    graph.init_weights()
    return graph

# define computation graph
graph = sample_network()

# optimize execution schedule
# optimized_graph = ios.optimize(graph, batch_size=5, opt_type='dp_parallel', compute_weight=True)

# measure latency
graph.sequential_schedule()
seq_latency, stage_latency = ios.ios_runtime.graph_latency(graph, batch_size=5, repeat=6, profile_stage=True)
print(graph)
print(f'Sequential schedule: {np.mean(seq_latency):.3f} ms')
print(f'      Stage latency: {np.mean(np.array(stage_latency).reshape(6, -1), axis=0)}\n')