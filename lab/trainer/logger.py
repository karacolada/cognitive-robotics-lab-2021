import glob
import json
import numpy as np
import os
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
from typing import List, Tuple, Dict


class Logger:
    def __init__(self, log_dir: str, num_runs: int, log_to_json: bool = True, log_to_tensorboard: bool = True,
                 log_to_wandb: bool = False):
        self._step = 0
        self.log_dir = Path(log_dir)
        self.num_runs = num_runs
        self.log_to_tensorboard = log_to_tensorboard
        self.log_to_wandb = log_to_wandb
        self.log_to_json = log_to_json
        self.run_dirs = self._generate_run_dirs()
        self._metrics = []
        self.to_tensorboard = TensorBoardOutput(self.run_dirs)
        if self.log_to_wandb:
            self.to_wandb = WandBOutput(self.run_dirs)
        self.to_json = JSONOutput(self.run_dirs)
        self.to_matplotlib = MatplotlibOutput(self.log_dir)

        self._train_steps = [0] * num_runs
        self._test_steps = [0] * num_runs
        self._train_episodes = [0] * num_runs
        self._test_episodes = [0] * num_runs

    def _generate_run_dirs(self) -> List:
        Path.mkdir(self.log_dir, parents=True, exist_ok=True)
        run_dirs = []
        for i in range(self.num_runs):
            current_dir = self.log_dir / f"run_{i + 1}"
            run_dirs.append(current_dir)
            Path.mkdir(current_dir)
        return run_dirs

    def add(self, name: str, value, step: int, prefix: str = None, run_id: int = None) -> None:
        name = f"{prefix}_{name}" if prefix else name
        if isinstance(value, torch.Tensor):
            value = value.cpu().detach().numpy()
        else:
            value = np.array(value)

        if run_id is None:
            run_id = 0

        if len(value.shape) not in (0, 1, 2, 3, 4):
            raise ValueError(f"Shape {value.shape} for name '{name}' cannot be interpreted as scalar, weights vector, "
                             f"image, or video.")

        self._metrics.append((run_id, step, name, value))
        self._log_to_outputs((run_id, step, name, value), prefix)

    def _log_to_outputs(self, summary: Tuple, prefix: str = None) -> None:
        run_id, step, name, value = summary
        if self.log_to_tensorboard:
            name = f"{prefix}/{name[len(prefix) + 1:]}" if prefix else name  # Change _ to / to create tensorboard dirs
            self.to_tensorboard([(run_id, step, name, value)])
        if self.log_to_wandb:
            name = f"{prefix}/{name[len(prefix) + 1:]}" if prefix else name  # Change _ to / to create wandb dirs
            self.to_wandb([(run_id, step, name, value)])
        if self.log_to_json:
            self.to_json([(run_id, step, name, value)])

    def increment_train_steps(self, increment: int, run_id: int) -> None:
        self._train_steps[run_id - 1] += increment

    def increment_test_steps(self, increment: int, run_id: int) -> None:
        self._test_steps[run_id - 1] += increment

    def increment_train_episodes(self, run_id: int) -> None:
        self._train_episodes[run_id - 1] += 1

    def increment_test_episodes(self, run_id: int) -> None:
        self._test_episodes[run_id - 1] += 1

    def train_episodes(self, run_id: int) -> int:
        return self._train_episodes[run_id - 1]

    def train_steps(self, run_id: int) -> int:
        return self._train_steps[run_id - 1]

    def avg_train_rewards(self, run_id: int) -> int:
        raise NotImplementedError

    def avg_test_rewards(self, run_id: int) -> int:
        raise NotImplementedError

    def load_metrics_from_file(self, path: Path) -> None:
        self._metrics = []
        self.append_metrics_from_file(path)

    def append_metrics_from_file(self, path: Path) -> None:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.rstrip('\n|\r'))
                self._metrics.append((data["run_id"], data["step"], data["name"], data["value"]))
            f.close()


class TensorBoardOutput:
    def __init__(self, run_dirs: List, fps: int = 24):
        self._writers = []
        for i in range(len(run_dirs)):
            self._writers.append(SummaryWriter(run_dirs[i]))
        self.experiment_dir = run_dirs[0].parent
        self._global_writer = SummaryWriter(self.experiment_dir)
        for p in glob.iglob(os.path.join(self.experiment_dir, "events.out.*")):
            os.remove(p)
        self.fps = fps

    def __call__(self, summaries: List) -> None:
        for run_id, step, name, value in summaries:
            if len(value.shape) == 0:
                self._writers[run_id - 1].add_scalar(name, value, global_step=step)
            elif len(value.shape) == 1:
                self._writers[run_id - 1].add_histogram(name, value, global_step=step)
            elif len(value.shape) == 2:
                self._writers[run_id - 1].add_image(name, value, global_step=step, dataformats="HW")
            elif len(value.shape) == 3:
                assert (value.shape[0] == 3 or value.shape[2] == 3), f"Input shape {value.shape} could not be " \
                                                                     f"interpreted as an RGB image."
                if value.shape[0] == 3:
                    self._writers[run_id - 1].add_image(name, value, global_step=step, dataformats="CHW")
                elif value.shape[2] == 3:
                    self._writers[run_id - 1].add_image(name, value, global_step=step, dataformats="HWC")
            elif len(value.shape) == 4:
                value = torch.from_numpy(value)
                value = self.normalize(value)
                value = value.unsqueeze(0)
                self._writers[run_id - 1].add_video(name, value, fps=self.fps, global_step=step)

    def add_hparams(self, hparams: Dict, results: Dict, run_id: int,
                    metric_keys: Tuple = ("test_rewards",)) -> None:
        assert all(key in results for key in metric_keys), "Desired metric keys are missing in results."
        metrics = {}
        for key in metric_keys:
            metrics["metrics/" + key] = results[key][-1]
        flattened_hparams = self.flatten_dict(hparams)
        processed_hparams = self.adapt_dict_values(flattened_hparams)
        self._global_writer.add_hparams(processed_hparams, metrics, run_name="run_" + str(run_id))

    @staticmethod
    def normalize(value: torch.Tensor) -> torch.Tensor:
        normalized_value = value
        minimum = torch.min(value)
        maximum = torch.max(value)
        if minimum < 0:
            normalized_value -= minimum
        if maximum - minimum > 1.:
            normalized_value = normalized_value / (maximum - minimum)
        return normalized_value

    @staticmethod
    def flatten_dict(input_dict: Dict, parent_key: str = "", separator: str = "/") -> Dict:
        import collections
        items = []
        for key, value in input_dict.items():
            new_key = parent_key + separator + key if parent_key else key
            if isinstance(value, collections.MutableMapping):
                items.extend(TensorBoardOutput.flatten_dict(value, key, separator).items())
            else:
                items.append((new_key, value))
        return dict(items)

    @staticmethod
    def adapt_dict_values(input_dict: Dict) -> Dict:
        processed_dict = input_dict.copy()
        for key, value in input_dict.items():
            if not isinstance(value, (int, float, str, bool, torch.Tensor)):
                processed_dict[key] = str(value)
        return processed_dict


class JSONOutput:
    def __init__(self, run_dirs: List) -> None:
        self.run_dirs = run_dirs

    def __call__(self, summaries: List) -> None:
        for run_id, step, name, value in summaries:
            # Log only scalar values to JSON.
            if len(value.shape) == 0:
                with (self.run_dirs[run_id - 1] / "metrics.jsonl").open("a") as f:
                    f.write(json.dumps({"run_id": run_id, "step": step, "name": name, "value": float(value)}) + "\n")


class MatplotlibOutput:
    def __init__(self, log_dir: Path) -> None:
        self.log_dir = log_dir
        self.plt = __import__("matplotlib.pyplot", fromlist=[None])

    def __call__(self, summaries: List, name: str = None) -> None:
        plot_values = []
        for run_id, step, n, value in summaries:
            # Log only scalar values to matplotlib figure.
            if len(value.shape) == 0:
                # Filter scalars by name.
                if name is not None:
                    if n == name:
                        plot_values.append((run_id, step, value))
                else:
                    assert False, "No name of the desired scalar given."

        self._sort_plot_values(plot_values)
        self._create_lineplots(plot_values, title=name)

    def _sort_plot_values(self, plot_values: List) -> None:
        plot_values.sort(key=self._take_second_element)

    @staticmethod
    def _take_second_element(input_tuple: Tuple) -> int:
        return input_tuple[1]

    def _create_lineplots(self, plot_values: List, title: str) -> None:
        self._avg_lineplot(plot_values, title)
        self._multiline_plot(plot_values, title)

    def _avg_lineplot(self, plot_values: List, title: str = "", color: str = "#3333FF",
                      font: str = "Helvetica", fontsize: int = 12, x_label: str = "Train episodes",
                      y_label: str = "Episode return", hide_splines: bool = True):
        default_font = self.plt.rcParams["font.family"]
        self.plt.rcParams["font.family"] = font
        self.plt.rcParams.update({"font.size": fontsize})
        steps, mean, std_dev = self._get_mean_and_std_dev(plot_values)
        assert steps.shape == mean.shape == std_dev.shape, "Steps, mean and std_dev shapes do not match."
        fig = self.plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(steps, mean, color=color)
        ax.fill_between(steps, mean - std_dev, mean + std_dev, facecolor=color, alpha=0.5)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        #ax.set_title(title)
        if hide_splines:
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
        self.plt.savefig(self.log_dir / f"avg_lineplot_{title}.svg", format="svg")
        self.plt.rcParams["font.family"] = default_font

    def _multiline_plot(self, plot_values: List, title: str = "", start_color: np.array = np.array([0.9, 0.58, 0.66]),
                        end_color: np.array = np.array([0.2, 0.0, 0.2]), font: str = "Helvetica", fontsize: int = 12,
                        x_label: str = "Train episodes", y_label: str = "Episode return", hide_splines: bool = True):
        default_font = self.plt.rcParams["font.family"]
        self.plt.rcParams["font.family"] = font
        self.plt.rcParams.update({"font.size": fontsize})
        steps, values = self._get_multirun_values(plot_values)
        fig = self.plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for run in range(values.shape[0]):
            color = start_color + (run / (values.shape[0] - 1)) * (end_color - start_color) if values.shape[0] > 1 \
                else start_color
            ax.plot(steps, values[run], color=tuple(color))
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        # ax.set_title(title)
        if hide_splines:
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
        self.plt.savefig(self.log_dir / f"multiline_plot_{title}.svg", format="svg")
        self.plt.rcParams["font.family"] = default_font

    def _prediction_rollouts_fig(self, rollouts: Dict, context_len: int, run_id: int, train_episodes: int,
                                 nth_step: int = 5, font: str = "Helvetica", fontsize: int = 12):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from torchvision.utils import make_grid
        default_font = self.plt.rcParams["font.family"]
        self.plt.rcParams["font.family"] = font
        self.plt.rcParams.update({"font.size": fontsize})

        context = {}
        prediction = {}
        for key, value in rollouts.items():
            context[key] = make_grid(torch.from_numpy(value[0:context_len]), nrow=context_len).numpy()
            array = np.arange(-1, value.shape[0])
            num_steps_in_context = int(np.floor(context_len / nth_step))
            first_step = np.array([array[context_len + 1]])
            latter_steps = array[0::nth_step][(num_steps_in_context + 1):]
            idxs = np.concatenate([first_step, latter_steps])
            numbering = idxs + 1
            prediction[key] = make_grid(torch.from_numpy(np.take(value, idxs, axis=0)), nrow=idxs.shape[0]).numpy()

        context = np.concatenate([context["true"], context["model"]], axis=1)
        prediction = np.concatenate([prediction["true"], prediction["model"]], axis=1)

        fig, axs = self.plt.subplots(1, 2, figsize=(8, 3),
                                     gridspec_kw=dict(width_ratios=[context.shape[2], prediction.shape[2]]))
        canvas = FigureCanvasAgg(fig)

        axs[0].imshow(context.transpose(1, 2, 0))
        axs[0].set_title("Context")
        axs[1].imshow(prediction.transpose(1, 2, 0))
        axs[1].xaxis.set_label_position('top')
        axs[1].xaxis.tick_top()

        xticks, xlabels = [], []
        for i in range(idxs.shape[0]):
            xticks.append((0.5 * prediction.shape[2] / idxs.shape[0]) + (i * prediction.shape[2] / idxs.shape[0]))
            xlabels.append(str(numbering[i]))
        axs[1].set_yticks([])
        axs[1].set_xticks(xticks)
        axs[1].set_xticklabels(xlabels)

        yticks, ylabels = [], ["True", "Model"]
        for j in range(2):
            yticks.append((0.5 * prediction.shape[1] / 2) + (j * prediction.shape[1] / 2))
        axs[0].set_xticks([])
        axs[0].tick_params(axis="y", pad=8)
        axs[0].set_yticks(yticks)
        axs[0].set_yticklabels(ylabels, rotation="vertical", ha="center", rotation_mode="anchor",
                               fontsize=(fontsize - 2))
        for i in range(2):
            axs[i].tick_params(axis=u'both', which=u'both', length=0)
            for position in ["left", "right", "top", "bottom"]:
                axs[i].spines[position].set_visible(False)
        fig.tight_layout()
        self.plt.savefig(self.log_dir /
                         f"prediction_rollouts_fig_run={run_id}_episode={train_episodes}.svg", format="svg", dpi=500)
        self.plt.rcParams["font.family"] = default_font
        canvas.draw()
        prediction_rollouts_fig = np.array(canvas.renderer.buffer_rgba())[:, :, 0:3]
        return torch.from_numpy(prediction_rollouts_fig).permute(2, 0, 1)



    @staticmethod
    def _get_mean_and_std_dev(plot_values: List) -> Tuple:
        steps = []
        mean = []
        std_dev = []
        prev_step = None
        for run_id, step, value in plot_values:
            # New step after all runs for previous step have been considered.
            if step != prev_step:
                if prev_step is not None:
                    steps.append(prev_step)
                    mean.append(np.mean(np.array(current_values)))
                    std_dev.append(np.std(np.array(current_values)))
                current_values = []
                prev_step = step
            current_values.append(value)
        # Append values for last step.
        steps.append(step)
        mean.append(np.mean(np.array(current_values)))
        std_dev.append(np.std(np.array(current_values)))
        return np.array(steps), np.array(mean), np.array(std_dev)

    @staticmethod
    def _get_multirun_values(plot_values: List) -> Tuple:
        steps = []
        values = []
        prev_step = None
        for run_id, step, value in plot_values:
            # New step after all runs for previous step have been considered.
            if step != prev_step:
                if prev_step is not None:
                    steps.append(prev_step)
                    values.append(np.array(current_values))
                current_values = []
                prev_step = step
            current_values.append(value)
        # Append values for last step.
        steps.append(step)
        values.append(np.array(current_values))
        return np.array(steps), np.stack(values, axis=1)


class WandBOutput:
    def __init__(self, run_dirs: List, fps: int = 24, silent: bool = True) -> None:
        self.run_dirs = run_dirs
        self.fps = fps
        os.environ["WANDB_SILENT"] = "true" if silent else "false"
        self.wandb = __import__("wandb")
        self._prev_run_id = -1
        self.wandb_run = None

    def __call__(self, summaries: List) -> None:
        for run_id, step, name, value in summaries:
            self._init_wandb_run(run_id)
            if len(value.shape) == 0:
                self.wandb.log({name: value}, step=step)
            elif len(value.shape) == 1:
                self.wandb.log({name: self.wandb.Histogram(value)}, step=step)
            elif len(value.shape) == 2:
                self.wandb.log({name: self.wandb.Image(value)}, step=step)
            elif len(value.shape) == 3:
                assert (value.shape[0] == 3 or value.shape[2] == 3), f"Input shape {value.shape} could not be " \
                                                                     f"interpreted as an RGB image."
                self.wandb.log({name: self.wandb.Image(value)}, step=step)
            elif len(value.shape) == 4:
                value = torch.from_numpy(value)
                value = TensorBoardOutput.normalize(value)
                value = value * 255.
                value = value.type(torch.uint8)
                self.wandb.log({name: self.wandb.Video(value, fps=self.fps)}, step=step)

    def watch(self, agent, run_id: int) -> None:
        self._init_wandb_run(run_id)
        self.wandb.watch(agent)

    def _init_wandb_run(self, run_id: int, project: str = "rl-lab") -> None:
        if run_id != self._prev_run_id:
            if self.wandb_run:
                self.wandb_run.finish()
                self.wandb_run = self.wandb.init(reinit=True, dir=self.run_dirs[run_id - 1],
                                                 name=str(self.run_dirs[run_id - 1]), project=project)
            else:
                self.wandb_run = self.wandb.init(dir=self.run_dirs[run_id - 1], name=str(self.run_dirs[run_id - 1]),
                                                 project=project)
            self._prev_run_id = run_id
