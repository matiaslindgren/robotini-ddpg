# Robotini racing simulator + deep reinforcement learning

Self-driving bots for the [Robotini racing simulator][robotini-simulator] with deep reinforcement learning (RL).
The deep RL algorithm used to train these bots is deep (recurrent) deterministic policy gradients ([DDPG][DDPG]/[RDPG][RDPG]) from the [TensorFlow Agents][tf-agents] framework.

Simulator socket connection and camera frame processing code is partially based on the `codecamp-2021-robotini-car-template` project (not (yet?) public).
However, this project moves all socket communication to separate [OS processes](https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Process) and implements message passing with [concurrent queues](https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Queue).
This is to avoid having the TensorFlow Python API runtime compete for thread time with the socket communication logic (because [GIL](https://docs.python.org/3/c-api/init.html#thread-state-and-the-global-interpreter-lock)).
All custom multiprocessing scribble could probably be replaced with a few lines external library code, e.g. [Celery](https://docs.celeryproject.org/en/stable/getting-started/introduction.html)?.

## Demo

Screen recordings from an 20 hour training session.

### Untrained agent

At the beginning, the agent has no policy for choosing actions based on inputs:

https://user-images.githubusercontent.com/11295697/119233183-63ac6980-bb30-11eb-8f07-fcdde03c210c.mp4

### After 2000 training steps

After 2 hours of training, the agent has learned to avoid the walls but is still making mistakes:

https://user-images.githubusercontent.com/11295697/119233195-7aeb5700-bb30-11eb-9a6c-d744ea498e09.mp4

### After 16000 training steps

After 20 hours of training, the agent is driving somewhat ok:

https://user-images.githubusercontent.com/11295697/119233260-dddcee00-bb30-11eb-9505-9ed36e6d2012.mp4

#### Metrics

TensorBoard provides visualizations for all metrics (only 2 shown here):
![two charts, both with slightly upwards trends][tensorboard-eval-metrics]

Here we can see how the average episode length (number of steps from spawn to crash or full lap) and average return (discounted sum of rewards for one episode) improve over time.
Horizontal axis is the number of training steps.

#### Note

By looking at the metrics at step 16k, we can see a global minimum of approx. 400 avg. episode length and 20 avg. return during evaluation.
Therefore, the agent driving car `env7_step16000_eval` seen in the video under "After 16000 training steps" might not be the optimal choice for a trained agent.

[This repository][trained-agent-repo] contains an agent from training step 11800.

### Live monitoring

https://user-images.githubusercontent.com/11295697/119233343-36ac8680-bb31-11eb-8fbf-e9f71f8e90d6.mp4


### Observation (neural network input)

The neural network sees 8 RGB pixels as input at each time step, also called the observation.
These are computed by averaging 4 non-overlapping row and column groups of the input camera frame:
![4 images, one showing the processed camera frame, two showing how to compute the 8 average values from the frame and one image showing the resulting 8 pixels on one row][explain-observation]

### Exploration

Experience gathering is done in parallel:

https://user-images.githubusercontent.com/11295697/119233302-0f55b980-bb31-11eb-931a-aec0a1a19842.mp4

## Requirements

* Linux or macOS
* [Robotini simulator][robotini-simulator-fork] - fork that exports additional car data over the logging socket.
* Unity - for running the simulator
* [Redis][redis] - Fast in-memory key-value store, used as a concurrent cache for storing car state.
* Python >= 3.6 + all deps from `requirements.txt`

## Running the training from scratch

Training a new agent can take several hours even on fast hardware due to the slow, real-time data collection.
There are no guarantees that an agent trained on one track will work on other tracks.

If you still want to try training a new agent from scratch, here's how to get started:

1. Start the simulator (use the [fork][robotini-simulator-fork]) using [this config file](./RaceParameters.json).
2. If using a single machine for both training and running the simulator, set `SIMULATOR_HOST := localhost` in the Makefile.
Otherwise, set the address of the simulator machine.

On the training machine, run these commands in separate terminals

3. `make start_redis`
4. `make train_ddpg`

### Optional

5. `make start_tensorboard`
6. `make start_monitor`
7. Open two tabs in a browser and go to:
* `localhost:8080` - live feed
* `localhost:6006` - TensorBoard


## Notes and nice-to-know things

* `./tf-data` contains TensorBoard logging data, saved agents/policies, checkpoints, i.e. all training state.
  It can (should) be deleted between unrelated training runs.
  **Note**: If a long training session produced a good agent, remember to backup it from `./tf-data/saved_policies` before deleting.
  Perhaps even back up the whole `tf-data` directory.
* Interrupting or terminating the training script may (rarely) leave processes running.
  Check e.g. with `ps ax | grep 'scripts/train_ddpg.py'` or `pgrep -f train_ddpg.py` and clean unwanted processes with e.g. `pkill -ef 'python3.9 scripts/train_ddpg.py'` (fix pattern depending on `ps ax` output).
* Every N training steps, the complete training state is written to `./tf-data/checkpoints`.
  This makes the training state persistent and allows you to continue training from the saved checkpoint.
    Note that this includes the full, pre-allocated replay buffer, which might make each checkpoint up to 2 GiB in size.
* For end-to-end debugging of the whole system, try `make run_debug_policy`. This will run a trivial hand-written policy that computes the amount of turn from RGB colors (see `robotini_ddpg.features.compute_turn_from_color_mass`).
* If the training log gets filled with `camera frame buffer is empty, skipping step` warnings, then the TensorFlow environment step loop is processing frames faster than we read from the simulator over the socket.
  Some known causes for this is that the simulator is under heavy load or the machine is sleeping.
  Also, the simulator dumps spectator logs to `race.log`, which can grow large if the simulator is running for a long time.
* I have no idea if the hyperparameters in `./scripts/config.yml` are optimal. Some tweaking perhaps could improve the results and training speed drastically.


[DDPG]: https://www.semanticscholar.org/paper/Continuous-control-with-deep-reinforcement-learning-Lillicrap-Hunt/024006d4c2a89f7acacc6e4438d156525b60a98f
[RDPG]: https://rll.berkeley.edu/deeprlworkshop/papers/rdpg.pdf
[explain-observation]: ./media/explain-observation.png
[tensorboard-eval-metrics]: ./media/tensorboard-eval-metrics.png
[redis]: https://redis.io/
[robotini-simulator-fork]: https://github.com/matiaslindgren/Robotini-Racing-Simulator
[robotini-simulator]: https://github.com/mikkomultanen/Robotini-Racing-Simulator
[tf-agents]: https://www.tensorflow.org/agents
[video-eval-step0]: ./media/eval-step0.webm
[video-eval-step2000]: ./media/eval-step2000.webm
[video-eval-step16000]: ./media/eval-step16000.webm
[video-web-ui]: ./media/web-ui.webm
[video-explore-step16000]: ./media/explore-step16000.webm
[trained-agent-repo]: https://github.com/matiaslindgren/robotini-ddpg-agent
