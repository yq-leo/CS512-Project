# P-GNN for Network Alignment

CS512 Projects

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

What things you need to install the software and how to install them:

- numpy
- networkx
- scipy
- torch
- torch_geometric
- tensorboard
- tqdm

### Installing

A step by step series of examples that tell you how to get a development environment running.

1. Clone the repository to your local machine:

```sh
git clone https://github.com/yq-leo/CS512-Project.git
```

2. Navigate to the project directory:

```sh
cd CS512-Project
```

3. Install the required dependencies:
```sh
pip install -r requirements.txt
```

4. To run the application, execute the following command in the terminal:
```sh
python main.py --dataset={dataset}
```

5. After running the application, you can visualize the training runs using TensorBoard. Execute the following command, replacing {dataset} with your dataset's name:
```sh
tensorboard --logdir logs/{dataset}
```



