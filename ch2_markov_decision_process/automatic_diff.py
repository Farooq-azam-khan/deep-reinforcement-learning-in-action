import torch as pt

""" calculating the grad/derivative with pytorch"""


def example():
    x = pt.tensor(2.0, requires_grad=True)
    y = 8 * x**4 + 3 * x**3 + 7 * x**2 + 6 * x + 3
    y.backward()

    print(f"derivative at {x=} is {x.grad}")


if __name__ == "__main__":
    example()
