# MLOps with Azure ML for PyTorch Model

This repository demonstrates an MLOps solution using Azure Machine Learning (Azure ML) for a PyTorch model. Public Walmart weekly sales data is used in this solution. The model is a simple regression model which predicts weekly sales of stores. The repository includes scripts for data preprocessing, training, and various YAML files for Azure ML components, pipelines, environment, data, and endpoints.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Deployment](#deployment)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

Make sure you have the following tools installed:
- [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli)
- [Azure ML CLI](https://docs.microsoft.com/en-us/azure/machine-learning/reference-azure-machine-learning-cli)
- [Python 3.x](https://www.python.org/downloads/)
- [PyTorch](https://pytorch.org/get-started/locally/)

## Setup

1. Clone the repository:
    ```sh
    git clone <repository-url>
    cd <repository-name>
    ```

2. Set up your Azure ML workspace. Follow the [Azure ML documentation](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-setup) for configuring a workspace.


3. Set up a Custom Environment on Azure ML workspace:
    ```sh
    az ml environment create --file ./src/yaml/environment.yml --resource-group $(resource_group) --workspace-name $(workspace_name)
    ```

    
4. Register Command Components to Azure ML Workspace:
   ```sh
    az ml component create --file ./src/yaml/preprocess-component.yml --resource-group $(resource_group) --workspace-name $(workspace_name)
    ```
   ```sh
    az ml component create --file ./src/yaml/train-component.yml --resource-group $(resource_group) --workspace-name $(workspace_name)
    ```


5. Register Data Asset to Azure ML Workspace:
   ```sh
    az ml data create --file ./src/yaml/data.yml --resource-group $(resource_group) --workspace-name $(workspace_name)
    ```


6. Submit a model training job on Azure ML Workspace:
   ```sh
    az ml job create --file ./src/yaml/pipeline-job.yml --resource-group $(resource_group) --workspace-name $(workspace_name)
    ```


7. Create a Real-Time endpoint for inference on Azure ML Workspace:
   ```sh
    az ml online-endpoint create --file ./src/yaml/endpoint.yml --resource-group $(resource_group) --workspace-name $(workspace_name)
    ```


8. Deploy the trained model to the endpoint on Azure ML Workspace:
   ```sh
    az ml online-deployment create --file ./src/yaml/deployment.yml --resource-group $(resource_group) --workspace-name $(workspace_name)
    ```


9. Invoke the endpoint to see predictions:
    ```sh
    az ml online-endpoint invoke --name sales-pytorch-endpoint --request-file ./src/onlinescoring/sample_request.json --resource-group $(resource_group) --workspace-name $(workspace_name)
    ```
