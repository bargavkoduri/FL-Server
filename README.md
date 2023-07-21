# FL-Server
The Federated Learning Server is a centralized component of the federated learning system designed to manage the training process of distributed machine learning models across multiple devices or nodes. 
This server acts as the coordinating entity that facilitates communication, aggregation, and model updates between participating devices without compromising data privacy.
<br/>
You can find the link to the client code <a href="https://github.com/bargavkoduri/FL-Client">here</a>

# Set-Up
To run the server make sure you have python and tensorflow installed on your system.
<br/>
To install other dependencies type
<br/>
``` npm i ```

# Usage
1) Run the below command to initialiaze the model. 
<br/>You can tweek the neural network by editing the init_model.py file.<br/>
The current network accepts monochrome images of 125x125 for 10 class classification. 
```python init_model.py```

2) Run ```node server.js``` to start the server <br/>
   You can change the number of clients required to start the process and client percentage in server.js file. The process stops as soon as Accuracy reaches 95%.
