## To-do List:
We need to refactor everything due to our main model builder is now completed.
However. There are still a lot of old broken code and problems needed to be fixed.

### Refactoring
  - "model" module
    - Main module to be called and used in the practice
    - Accept basic convolution, residual, and inception formats
    - Consists of X classes:
      - "ModelBuilder.py": Build a CNN model using PyTorch
      - "ModelChroClass.py": Handle and process chromosomes
      - "ModelGeneOperator.py": Handle the crossover and mutation for using with GA.
        - We already have methods. But they need to support for "universal" chromosomes format. 
          - Residual takes the highest priority. But so far we haven't discover its methods yet.
          - Inception shouldn't prone any problems. Since we have specified to perform only at inception section. So all models are safe
          - Normal CNN takes the lowest. And the most problemetic since we would want to perform it in only CNN sections only. Also rewrite to mutate only and ONLY cnn parts. Not inception block + residual connections
        - We should create a way to overload genetic operators from classes then perform them.
          - Need to check for the argument sizes for every operators first.
  - [x] Deprecate "vgg_like" in the future. Move the gene operator + chro class into pure_networks to use for CNN.
  - [x] .. Deprecate "Linear_Net.py" since we now have an universal model builder.
  - [ ] Tidy up other file that aren't associated with model creation
  - [ ] Implement "AutoNN" properly

### To-Do Tasks
- [ ] Inception chromosomes
    - [ ] Implement options for All-Same, N-Same and All-Different in mutation methods
-  [ ] Residual chromosomes
   -  [ ] Implement.. mutation methods. Crossover isn't needed. Need some explorations.
      - Either using dict or go raw with list.
- [ ] "model" modules
  - [ ] "ModelBuilder.py"
    - [ ] Add an option for what to add after the ouput. (Softmax, Sigmoid, .etc .etc). Should be a list
  - [ ] "ModelChroClass.py"
    - [ ] "chro_chunk2str", "chro_checker", and "chro_generate"
      - [ ] "chro_generate" is just dict network generator transformed into chromosomes because it's less complicate.
- [ ] Dictionary networks
  - [ ] Implement model generator
    - Wait. If we had built this method. We could add options whether to generate with residual, inception, linear (before output) or not, we could replace the other generator methods with this one instead.
    - [x] Along with model building in each classes too. We had built an universal model builder. We could just deprecate them for reducing the confusion.
    - What about checkers too? We wouldn't need it since we can use universal model checker for checking every kinds of chromosomes.
- [ ] Proper "Docstring" comments when completed
- [ ] Proper exception messages
- [ ] Clean up the README.md
- [ ] Write the wiki

We could move old code to "legacy" section. But since I'm sure no one cares or even use this (lol), we are just going to deprecate them. :shrug:

## Future Plan
- [ ] Implement **"tactic"** parameters in model creation in which controls how the builder should calculate CNN channels for the model. As current. We use image size for the output channels from the first layers (similar to VGGNet) and multiply the channel by 2 for every pooling channels.
  - [ ] We might need to rewrite "CNN_Net.py" method to be more effective
- [ ] Fully support for residual connections with the same starting point.
  - [ ] As now. You can create such connections but requires the image size for every end connection points to be the same as a starting point. 
- [ ] Allow for activation function specification.
  - Not sure why since ReLU is clearly a choice but eh.
  - Or better. Specify what extra layers should come after for a convolution layer.
- [ ] Support for PyMoo (?)