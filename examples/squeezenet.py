import os
import numpy as np
import ios


def main():
    graph = ios.models.squeezenet()

    graph.sequential_schedule()
    latency = ios.ios_runtime.graph_latency(graph, batch_size=1, repeat=6)

    optimized_graph = ios.optimize(graph, batch_size=1, opt_type='dp_parallel')
    optimized_latency = ios.ios_runtime.graph_latency(optimized_graph, batch_size=1, repeat=6)

    print(f' Sequential schedule: {np.mean(latency):.3f} ms')
    print(f'  Optimized schedule: {np.mean(optimized_latency):.3f} ms')
    print(optimized_graph)


if __name__ == '__main__':
    main()
