import mlflow
import time


class BaseClassWithLogging:
    def __init__(self, class_name):
        self.class_name = class_name

    def start_logging(self):
        mlflow.start_run(run_name=self.class_name, nested=True)

        mlflow.set_tag("class_name", self.class_name)

    def end_logging(self):
        mlflow.end_run()
