import os
import numpy as np
import ios


def main():
    graph = ios.models.inception_v3()

    graph.sequential_schedule()
    latency = ios.ios_runtime.graph_latency(graph, batch_size=1, repeat=6)

    optimized_graph = ios.optimize(graph, batch_size=1, opt_type='dp_merge_parallel_transform')
    optimized_latency = ios.ios_runtime.graph_latency(optimized_graph, batch_size=1, repeat=6)

    # os.makedirs("./outputs/", exist_ok=True)
    # ios.draw(optimized_graph, fname=f'./outputs/optimized_{graph.name}.png', label=f'Optimized Graph, Latency = {np.mean(optimized_latency):.3f}')
    print(optimized_graph)
    print(f' Sequential schedule: {np.mean(latency):.3f} ms')
    print(f'  Optimized schedule: {np.mean(optimized_latency):.3f} ms')


if __name__ == '__main__':
    main()
