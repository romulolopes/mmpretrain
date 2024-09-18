import os
from typing import Dict, Optional

from mmengine.dist import master_only
from mmdet.registry import HOOKS

from mmengine.hooks import LoggerHook

os.environ['MLFLOW_TRACKING_USERNAME'] = os.environ.get('DAGSHUB_USER')
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.environ.get('DAGSHUB_TOKEN')


@HOOKS.register_module()
class MlflowLoggerHook(LoggerHook):
    """Class to log metrics and (optionally) a trained model to MLflow.

    It requires `MLflow`_ to be installed.

    Args:
        exp_name (str, optional): Name of the experiment to be used.
            Default None. If not None, set the active experiment.
            If experiment does not exist, an experiment with provided name
            will be created.
        tags (Dict[str], optional): Tags for the current run.
            Default None. If not None, set tags for the current run.
        params (Dict[str], optional): Params for the current run.
            Default None. If not None, set params for the current run.
        log_model (bool, optional): Whether to log an MLflow artifact.
            Default True. If True, log runner.model as an MLflow artifact
            for the current run.
        interval (int): Logging interval (every k iterations). Default: 10.
        ignore_last (bool): Ignore the log of last iterations in each epoch
            if less than `interval`. Default: True.
        reset_flag (bool): Whether to clear the output buffer after logging.
            Default: False.
        by_epoch (bool): Whether EpochBasedRunner is used. Default: True.

    .. _MLflow:
        https://www.mlflow.org/docs/latest/index.html
    """

    def __init__(self,
                 exp_name: Optional[str] = None,
                 tags: Optional[Dict] = None,
                 params: Optional[Dict] = None,
                 log_model: bool = True,
                 interval: int = 50,
                 ignore_last: bool = True,
                 reset_flag: bool = False,
                 by_epoch: bool = True):
        super().__init__(interval=interval, ignore_last=ignore_last,log_metric_by_epoch=by_epoch)
        #super().__init__(interval, ignore_last, reset_flag, by_epoch)
        self.import_mlflow()
        self.exp_name = exp_name
        self.tags = tags
        self.params = params
        self.log_model = log_model
        self.uri = os.environ.get('DAGSHUB_MLFLOW')

    
    def import_mlflow(self) -> None:
        try:
            import mlflow
            import mlflow.pytorch as mlflow_pytorch
            
        except ImportError:
            raise ImportError(
                'Please run "pip install mlflow" to install mlflow')
        self.mlflow = mlflow
        self.mlflow_pytorch = mlflow_pytorch

    @master_only
    def before_run(self, runner) -> None:
        super().before_run(runner)
        self.mlflow.set_tracking_uri(self.uri)

        if self.exp_name is not None:
            experiment_exists = self.mlflow.get_experiment_by_name(self.exp_name) 
            if not experiment_exists: 
                self.mlflow.create_experiment(self.exp_name)
            self.mlflow.set_experiment(self.exp_name)

        if self.tags is not None:
            self.mlflow.set_tags(self.tags)
        if self.params is not None:
            self.mlflow.log_params(self.params)

    @master_only
    def after_val_epoch(self, runner, metrics) -> None:
        _metrics = {key.replace('coco/', ''): value for key, value in metrics.items()}
        self.mlflow.log_metrics(_metrics, step=runner.epoch+1)

    @master_only
    def after_run(self, runner) -> None:
        if self.log_model:
            self.mlflow_pytorch.log_model(runner.model,'models')
