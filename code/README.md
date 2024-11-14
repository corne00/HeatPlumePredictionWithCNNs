# Physics loss
data requirements: 
- the data need velocity fields as an input
- hence number of inputs needs to be changed to 5

to train with physics loss, 2 adaptations are required:
- add the physics loss to the loss function
- change in the training loop the inputs for the loss calculation, by setting the parameter `energy_loss` to `True`
    ```python
    energy_loss = False
    if energy_loss: 
        l = loss_func(inputs, predictions)
    else:
        l = loss_func(predictions, labels)
                
    ```