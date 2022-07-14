import numpy as np
from paragnizer.test.main import main as test_main, get_nn, get_bilstm

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    bilstm = get_bilstm(6, 6)
    print(bilstm)
    print(count_parameters(bilstm))

    x = [
        [1,2,3,4,4,6],
        [1,2,3,4,4,6],
        [1,2,3,4,4,6],
        [1,2,3,4,4,6],
    ]

    x = np.array(x, dtype=np.float32)
    
    output = bilstm.forward(x)
    print(output)
    for o in output:
        print(o.sum())


if __name__ == '__main__':
    main()