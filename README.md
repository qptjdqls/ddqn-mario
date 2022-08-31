# ddqn-mario


https://teddylee777.github.io/pytorch/pytorch-tutorial-01

self.net = MarioNet(self.state_dim, self.action_dim).float()

state = state.__array__()

cuda

decrease exploration rate 

state, next_state, action, reward, done = map(torch.stack, zip(*batch))


@torch.no_grad()


if self.curr_step % self.learn_every != 0:
            return None, None