# ddqn-mario

### Origianl code: <br>
https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html <br>
https://github.com/pytorch/tutorials/blob/master/intermediate_source/mario_rl_tutorial.py <br>

**Things to remember** <br>
-[Gym Wrappers](https://www.gymlibrary.dev/content/wrappers/) <br>
[Chaining previous environment method](https://github.com/pytorch/tutorials/blob/master/intermediate_source/mario_rl_tutorial.py) (chain of responsibility)<br>
-Usage of ```@torch.no_grad()``` <br>

**Things to improve** <br>
-Abuse copying e.g. ```.copy()```, ```__array__()``` <br>
-checking CUDA <br>
-Exploration rate scheduling <br>
Use $\epsilon = \alpha - e^{\beta -x}$ form instead <br>
-Logging with Tensorboard