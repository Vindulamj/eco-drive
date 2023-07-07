# MIT License

# Copyright (c) 2023 Vindula Jayawardana and Zhongxia Yan

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from imported_utils.exp_utils import Config

from ut import *
import wandb


class Main(Config):
    flow_base = Path.env("F")._real

    def __init__(c, agent, res, *args, **kwargs):
        super().__init__(agent, res, *args, **kwargs)

    def create_env(c):
        raise NotImplementedError

    @property
    def dist_class(c):
        if "_dist_class" not in c:
            c._dist_class = build_dist(c.action_space)
        return c._dist_class

    @property
    def model_output_size(c):
        if "_model_output_size" not in c:
            c._model_output_size = c.dist_class.model_output_size
        return c._model_output_size

    @property
    def observation_space(c):
        raise NotImplementedError

    @property
    def action_space(c):
        raise NotImplementedError

    def set_model(c):
        c._model = c.get("model_cls", FFN)(c)
        print(c._model)
        return c

    def schedule(c, coef, schedule=None):
        if not schedule and isinstance(coef, (float, int)):
            return coef
        frac = c._i / c.n_steps
        frac_left = 1 - frac
        if callable(coef):
            return coef(frac_left)
        elif schedule == "linear":
            return coef * frac_left
        elif schedule == "cosine":
            return coef * (np.cos(frac * np.pi) + 1) / 2

    @property
    def _lr(c):
        return c.schedule(c.get("lr", 1e-4), c.get("lr_schedule"))

    def log_stats(c, stats, ii=None, n_ii=None, print_time=False):
        if ii is not None:
            assert n_ii is not None and n_ii > 0
        stats = {k: v for k, v in stats.items() if v is not None}
        total_time = time() - c._run_start_time
        if print_time:
            stats["total_time"] = total_time

        prints = []
        prints.extend("{} {:.3g}".format(*kv) for kv in stats.items())

        widths = [len(x) for x in prints]
        line_w = terminal_width()
        prefix = "i {:d}".format(c._i)
        i_start = 0
        curr_w = len(prefix) + 3
        curr_prefix = prefix
        for i, w in enumerate(widths):
            if curr_w + w > line_w:
                c.log(" | ".join([curr_prefix, *prints[i_start:i]]))
                i_start = i
                curr_w = len(prefix) + 3
                curr_prefix = " " * len(prefix)
            curr_w += w + 3
        c.log(" | ".join([curr_prefix, *prints[i_start:]]))
        sys.stdout.flush()

        df = c._results
        for k, v in stats.items():
            if k not in df:
                df[k] = np.nan
            df.loc[c._i, k] = v

    def get_log_ii(c, ii, n_ii, print_time=False):
        return lambda **kwargs: c.log_stats(kwargs, ii, n_ii, print_time=print_time)

    def on_rollout_worker_start(c):
        c._env = c.create_env()
        c.set_model()
        c._model.eval()
        c._i = 0

    def set_weights(c, weights):  # For Ray
        c._model.load_state_dict(
            weights, strict=False
        )  # If c.use_critic, worker may not have critic weights

    def on_train_start(c):
        """
        Run at the begining of the simulation run. Will create the environment and
        initialize the training algorithm.
        """
        c.setdefaults(alg="Algorithm")
        c._env = c.create_env()

        c._alg = (eval(c.alg) if isinstance(c.alg, str) else c.alg)(c)
        c.set_model()
        c._model.train()
        c._model.to(c.device)

        c._i = 0  # for c._lr
        opt = c.get("opt", c.opt)
        if opt == "Adam":
            c._opt = optim.Adam(
                c._model.parameters(),
                lr=c._lr,
                betas=c.get("betas", (0.9, 0.999)),
                weight_decay=c.get("l2", 0),
            )
        elif opt == "RMSprop":
            c._opt = optim.RMSprop(
                c._model.parameters(), lr=c._lr, weight_decay=c.get("l2", 0)
            )
        else:
            raise "No optimizer defined"

        c._run_start_time = time()
        c._i = c.set_state(c._model, opt=c._opt, step="max")

        if c._i:
            c._results = c.load_train_results().loc[: c._i]
            c._run_start_time -= c._results.loc[c._i, "total_time"]
        else:
            c._results = pd.DataFrame(index=pd.Series(name="step"))
        c._i_gd = None

        c.try_save_commit(Main.flow_base)

    def on_step_start(c, stats={}):
        """
        Record training stats and save the training model
        """
        lr = c._lr
        for g in c._opt.param_groups:
            g["lr"] = float(lr)
        c.log_stats(dict(**stats, **c._alg.on_step_start(), lr=lr))
        try:
            if c.wandb:
                # log the training progress in wandb
                wandb.log({"total time": 10}, step=c._i)
        except KeyError:
            print("Wandb recording error!!")
            pass

        c.stats.reset()

        if c._i % c.step_save == 0:
            c.save_train_results(c._results)
            c.save_state(c._i, c.get_state(c._model, c._opt, c._i))

    def rollouts(c):
        """
        Collect a list of rollouts for the training step
        """
        if c.use_ray:
            import ray

            weights_id = ray.put({k: v.cpu() for k, v in c._model.state_dict().items()})
            [w.set_weights.remote(weights_id) for w in c._rollout_workers]
            rollout_stats = flatten(
                ray.get(
                    [
                        w.rollouts_single_process.remote(worker_id=i)
                        for i, w in enumerate(c._rollout_workers)
                    ]
                )
            )
        else:
            rollout_stats = c.rollouts_single_process(worker_id=0)
        # compute advantage estimate from the rollout
        rollouts = [
            c.on_rollout_end(*rollout_stat, ii=ii, n_ii=c.n_rollouts_per_step)
            for ii, rollout_stat in enumerate(rollout_stats)
        ]
        return NamedArrays.concat(rollouts, fn=flatten)

    def rollouts_single_process(c, worker_id=0):
        """
        sub method: collect a rollout for the training step
        """
        rollout_stats = []
        if c.n_rollouts_per_worker > 1:
            rollout_stats = [
                c.var(i_rollout=i).rollout(worker_id=worker_id)
                for i in range(c.n_rollouts_per_worker)
            ]
        else:
            n_steps_total = 0
            rollout_stats = []
            while n_steps_total < c.horizon:
                rollout, stats = c.rollout(worker_id=worker_id)
                rollout_stats.append((rollout, stats))
                n_steps_total += stats.get("steps")
        return rollout_stats

    def rollout(c, worker_id):
        """
        sub method: collect a rollout for the training step
        """
        c.setdefaults(skip_stat_steps=0, rollout_kwargs=None)

        # IMPORTANT: Calls the reset method to reset the environment and all subscription models
        ret = c._env.reset()

        if not isinstance(ret, dict):
            ret = dict(obs=ret)
        rollout = NamedArrays()
        # TODO: What if there are no rl vehicles at the current step. Fix this
        rollout.append(**ret)

        done = False
        a_space = c.action_space
        step = 0
        # rollout unfold
        while step < c.horizon + c.skip_stat_steps and not done:
            pred = from_torch(
                c._model(
                    to_torch(rollout.obs[-1]), value=False, policy=True, argmax=False
                )
            )

            if c.get("aclip", True) and isinstance(a_space, Box):
                pred.action = np.clip(pred.action, a_space.low, a_space.high)

            rollout.append(**pred)
            ret = c._env.step(rollout.action[-1], rollout.id[-1], step)

            if isinstance(ret, tuple):
                obs, reward, done, info = ret
                ret = dict(obs=obs, reward=reward, done=done, info=info)
            done = ret.setdefault("done", False)
            if done:
                ret = {k: v for k, v in ret.items() if k not in ["obs", "id"]}
            rollout.append(**ret)
            step += 1

        stats = {"steps": step}

        speeds = []
        fuel = []
        emission = []

        lane_length = c.lane_length
        intersection_length = c.intersection_length

        for k in c.stats.veh_data.keys():
            data_ary = c.stats.veh_data[k]
            v_speeds = v_fuel = v_emission = []
            is_warmup_vehicle = False
            has_finished_trip = False

            for d in data_ary:
                # only collect non-warmup vehicles stats
                if d[3] == 0:
                    is_warmup_vehicle = True
                    break
                v_speeds.append(d[0])
                v_fuel.append(d[1])
                v_emission.append(d[2])
                if d[4] == 1:
                    has_finished_trip = True

            if not is_warmup_vehicle and has_finished_trip:
                # TODO: this assumes the incoming and outgoing intersections are of the same length
                speeds.append((lane_length * 2 + intersection_length) / len(v_speeds))
                fuel.append(np.sum(np.array(v_fuel)))
                emission.append(np.sum(np.array(v_emission)))

        # record running stats
        c.stats.veh_speed_data_avg.append(np.mean(np.array(speeds)))
        c.stats.veh_speed_data_avg.append(np.mean(np.array(speeds)))
        c.stats.veh_fuel_data_avg.append(np.mean(np.array(fuel)))
        c.stats.veh_emission_data_avg.append(np.mean(np.array(emission)))

        c.stats.print_stats(epi_speeds=speeds, epi_fuel=fuel)

        # reset data strcutures carrying episode data
        c.stats.reset()

        return rollout, stats

    def on_rollout_end(c, rollout, stats, ii=None, n_ii=None):
        """
        Compute value, calculate advantage, log stats
        """
        t_start = time()
        step_id_ = rollout.pop("id", None)
        done = rollout.pop("done", None)
        multi_agent = step_id_ is not None

        step_obs_ = rollout.obs
        step_obs = step_obs_ if done[-1] else step_obs_[:-1]
        assert len(step_obs) == len(rollout.reward)

        value_ = None
        if c.use_critic:
            ((_, mb_),) = rollout.filter("obs").iter_minibatch(
                concat=multi_agent, device=c.device
            )
            value_ = from_torch(c._model(mb_.obs, value=True).value.view(-1))

        if multi_agent:
            step_n = [len(x) for x in rollout.reward]
            reward = (np.concatenate(rollout.reward)).flatten()
            ret, adv = calc_adv_multi_agent(
                np.concatenate(step_id_), reward, c.gamma, value_=value_, lam=c.lam
            )
            rollout.update(obs=step_obs, ret=split(ret, step_n))
            if c.use_critic:
                rollout.update(
                    value=split(value_[: len(ret)], step_n), adv=split(adv, step_n)
                )
        else:
            reward = rollout.reward
            ret, adv = calc_adv(reward, c.gamma, value_, c.lam)
            rollout.update(obs=step_obs, ret=ret)
            if c.use_critic:
                rollout.update(value=value_[: len(ret)], adv=adv)

        log = c.get_log_ii(ii, n_ii)
        log(**stats)
        log(
            reward_mean=np.mean(reward),
            value_mean=np.mean(value_) if c.use_critic else None,
            ret_mean=np.mean(ret),
            adv_mean=np.mean(adv) if c.use_critic else None,
            explained_variance=explained_variance(value_[: len(ret)], ret)
            if c.use_critic
            else None,
        )
        log(rollout_end_time=time() - t_start)
        return rollout

    def on_step_end(c, stats={}):
        """
        Log the stat after one episode
        """
        c.log_stats(stats, print_time=True)
        c.log("")

    def on_train_end(c):
        """
        Save training results.
        """
        if c._results is not None:
            c.save_train_results(c._results)

        if hasattr(c._env, "close"):
            c._env.close()

    def train(c):
        """
        Main training loop
        """
        if c.wandb:
            os.environ["WANDB_API_KEY"] = c.wandb_key
            if c.wandb_dry_run:
                os.environ["WANDB_MODE"] = "dryrun"
            project = c.wandb_proj
            wandb.init(project=project)

        c.on_train_start()
        while c._i < c.n_steps:
            c.on_step_start()
            with torch.no_grad():
                rollouts = c.rollouts()
            gd_stats = {}
            if len(rollouts.obs):
                t_start = time()
                c._alg.optimize(rollouts)
                gd_stats.update(gd_time=time() - t_start)
            c.on_step_end(gd_stats)
            c._i += 1

        # one last step
        c.on_step_start()
        with torch.no_grad():
            rollouts = c.rollouts()
            c.on_step_end()
        c.on_train_end()

    def eval(c):
        """
        Main evaluation loop
        """
        c.setdefaults(alg="PPO")
        c._env = c.create_env()

        c._alg = (eval(c.alg) if isinstance(c.alg, str) else c.alg)(c)
        c.set_model()
        c._model.eval()
        c._results = pd.DataFrame(index=pd.Series(name="step"))

        kwargs = {"step" if isinstance(c.e, int) else "path": c.e}
        step = c.set_state(c._model, opt=None, **kwargs)
        c.log("Loaded model from step %s" % step)

        c._run_start_time = time()
        c._i = 1
        for _ in range(c.n_steps):
            c.rollouts()
            c._i += 1
            c.log("")

        if hasattr(c._env, "close"):
            c._env.close()

    def run(c):
        """
        Selects training or evaluation route based on provided inputs
        """
        c.log(format_yaml({k: v for k, v in c.items() if not k.startswith("_")}))
        c.setdefaults(n_rollouts_per_step=c.per_step_rollouts)
        if c.test_run:
            c.setdefaults(n_workers=1)
            c.setdefaults(
                use_ray=False,
                n_rollouts_per_worker=c.n_rollouts_per_step // c.n_workers,
            )
            print("[INFO] Evaluation mode!!")
            c.eval()
        else:
            if c.get("use_ray", True):
                c.setdefaults(device="cuda" if torch.cuda.is_available() else "cpu")
                c.setdefaults(n_workers=c.n_rollouts_per_step, use_ray=True)
                c.n_rollouts_per_worker = c.n_rollouts_per_step // c.n_workers
                import ray

                try:
                    ray.init(num_cpus=c.n_workers, include_dashboard=False)
                except:
                    ray.init(
                        num_cpus=c.n_workers,
                        include_dashboard=False,
                        _temp_dir=(Path.env("F") / "tmp")._real,
                    )
                RemoteMain = ray.remote(type(c))
                worker_kwargs = c.get("worker_kwargs") or [{}] * c.n_workers
                assert len(worker_kwargs) == c.n_workers
                worker_kwargs = [
                    {**c, "main": False, "device": "cpu", **args}
                    for args in worker_kwargs
                ]
                c._rollout_workers = [
                    RemoteMain.remote(**kwargs, i_worker=i)
                    for i, kwargs in enumerate(worker_kwargs)
                ]
                ray.get(
                    [w.on_rollout_worker_start.remote() for w in c._rollout_workers]
                )
            else:
                c.setdefaults(n_workers=1, n_rollouts_per_worker=c.n_rollouts_per_step)

            assert c.n_workers * c.n_rollouts_per_worker == c.n_rollouts_per_step
            print("[INFO] Training mode!!")
            c.train()
