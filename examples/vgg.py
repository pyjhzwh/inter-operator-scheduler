import numpy as np
import ios


def main():
    graph0 = ios.models.vgg_11()

    graph0.sequential_schedule()
    latency0 = ios.ios_runtime.graph_latency(graph0, batch_size=1, repeat=6)

    graph1 = ios.models.vgg_11_opt_layout()

    graph1.sequential_schedule()
    latency1 = ios.ios_runtime.graph_latency(graph1, batch_size=1, repeat=6)

    print(f'original vgg_11 Sequential schedule: {np.mean(latency0):.3f} ms')
    print(f'opt_layout vgg_11 Sequential schedule: {np.mean(latency1):.3f} ms')


if __name__ == '__main__':
    main()