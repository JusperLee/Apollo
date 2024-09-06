###
# Author: Kai Li
# Date: 2022-05-27 10:27:56
# Email: lk21@mails.tsinghua.edu.cn
# LastEditTime: 2022-06-13 12:11:15
###
from rich import print
from dataclasses import dataclass
from pytorch_lightning.utilities import rank_zero_only
from typing import Union
from pytorch_lightning.callbacks.progress.rich_progress import *
from rich.console import Console, RenderableType
from rich.progress_bar import ProgressBar
from rich.style import Style
from rich.text import Text
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
    ProgressColumn
)
from rich import print, reconfigure

@rank_zero_only
def print_only(message: str):
    print(message)

@dataclass
class RichProgressBarTheme:
    """Styles to associate to different base components.

    Args:
        description: Style for the progress bar description. For eg., Epoch x, Testing, etc.
        progress_bar: Style for the bar in progress.
        progress_bar_finished: Style for the finished progress bar.
        progress_bar_pulse: Style for the progress bar when `IterableDataset` is being processed.
        batch_progress: Style for the progress tracker (i.e 10/50 batches completed).
        time: Style for the processed time and estimate time remaining.
        processing_speed: Style for the speed of the batches being processed.
        metrics: Style for the metrics

    https://rich.readthedocs.io/en/stable/style.html
    """

    description: Union[str, Style] = "#FF4500"
    progress_bar: Union[str, Style] = "#f92672"
    progress_bar_finished: Union[str, Style] = "#b7cc8a"
    progress_bar_pulse: Union[str, Style] = "#f92672"
    batch_progress: Union[str, Style] = "#fc608a"
    time: Union[str, Style] = "#45ada2"
    processing_speed: Union[str, Style] = "#DC143C"
    metrics: Union[str, Style] = "#228B22"

class BatchesProcessedColumn(ProgressColumn):
    def __init__(self, style: Union[str, Style]):
        self.style = style
        super().__init__()

    def render(self, task) -> RenderableType:
        total = task.total if task.total != float("inf") else "--"
        return Text(f"{int(task.completed)}/{int(total)}", style=self.style)

class MyMetricsTextColumn(ProgressColumn):
    """A column containing text."""

    def __init__(self, style):
        self._tasks = {}
        self._current_task_id = 0
        self._metrics = {}
        self._style = style
        super().__init__()

    def update(self, metrics):
        # Called when metrics are ready to be rendered.
        # This is to prevent render from causing deadlock issues by requesting metrics
        # in separate threads.
        self._metrics = metrics

    def render(self, task) -> Text:
        text = ""
        for k, v in self._metrics.items():
            text += f"{k}: {round(v, 3) if isinstance(v, float) else v} "
        return Text(text, justify="left", style=self._style)

class MyRichProgressBar(RichProgressBar):
    """A progress bar prints metrics at the end of each epoch
    """

    def _init_progress(self, trainer):
        if self.is_enabled and (self.progress is None or self._progress_stopped):
            self._reset_progress_bar_ids()
            reconfigure(**self._console_kwargs)
            # file = open("/home/likai/data/Look2Hear/Experiments/run_logs/EdgeFRCNN-Noncausal.log", 'w')
            self._console: Console = Console(force_terminal=True)
            self._console.clear_live()
            self._metric_component = MetricsTextColumn(trainer, self.theme.metrics)
            self.progress = CustomProgress(
                *self.configure_columns(trainer),
                self._metric_component,
                auto_refresh=False,
                disable=self.is_disabled,
                console=self._console,
            )
            self.progress.start()
            # progress has started
            self._progress_stopped = False